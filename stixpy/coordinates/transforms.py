import warnings
from functools import lru_cache

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.io import fits
from astropy.table import QTable, vstack
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.net import Fido
from sunpy.net import attrs as a

from stixpy.coordinates.frames import STIXImaging
from stixpy.utils.logging import get_logger

STIX_X_SHIFT = 26.1 * u.arcsec  # fall back to this when non sas solution available
STIX_Y_SHIFT = 58.2 * u.arcsec  # fall back to this when non sas solution available
STIX_X_OFFSET = 60.0 * u.arcsec  # remaining offset after SAS solution
STIX_Y_OFFSET = 8.0 * u.arcsec  # remaining offset after SAS solution

logger = get_logger(__name__)

__all__ = ["get_hpc_info", "stixim_to_hpc", "hpc_to_stixim"]


def _get_rotation_matrix_and_position(obstime, obstime_end=None):
    r"""
    Return rotation matrix STIX Imaging to SOLO HPC and position of SOLO in HEEQ.

    Parameters
    ----------
    obstime : `astropy.time.Time`
        Time
    Returns
    -------
    """
    roll, solo_position_heeq, spacecraft_pointing = get_hpc_info(obstime, obstime_end)

    # Generate the rotation matrix using the x-convention (see Goldstein)
    # Rotate +90 clockwise around the first axis
    # STIX is mounted on the -Y panel (SOLO_SRF) need to rotate to from STIX frame to SOLO_SRF
    rot_to_solo = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    logger.debug("Pointing: %s.", spacecraft_pointing)
    A = rotation_matrix(-1 * roll, "x")
    B = rotation_matrix(spacecraft_pointing[..., 1], "y")
    C = rotation_matrix(-1 * spacecraft_pointing[..., 0], "z")

    # Will be applied right to left
    rmatrix = A @ B @ C @ rot_to_solo

    return rmatrix, solo_position_heeq


def get_hpc_info(times, end_time=None):
    r"""
    Get STIX pointing and SO location from L2 aspect files.

    Parameters
    ----------
    times : `astropy.time.Time`
        Time/s to get hpc info for at.

    Returns
    -------

    """
    aux = _get_aux_data(times.min(), end_time or times.max())

    indices = np.argwhere((aux["time"] >= times.min()) & (aux["time"] <= times.max()))
    if end_time is not None:
        indices = np.argwhere((aux["time"] >= times.min()) & (aux["time"] <= end_time))
    indices = indices.flatten()

    if end_time is not None and times.size == 1 and indices.size >= 2:
        # mean
        aux = aux[indices]

        roll, pitch, yaw = np.mean(aux["roll_angle_rpy"], axis=0)
        solo_heeq = np.mean(aux["solo_loc_heeq_zxy"], axis=0)

        good_solution = np.where(aux["sas_ok"] == 1)
        good_sas = aux[good_solution]

        if len(good_sas) == 0:
            warnings.warn(f"No good SAS solution found for time range: {times} to {end_time}.")
            sas_x = 0
            sas_y = 0
        else:
            sas_x = np.mean(good_sas["y_srf"])
            sas_y = np.mean(good_sas["z_srf"])
            sigma_x = np.std(good_sas["y_srf"])
            sigma_y = np.std(good_sas["z_srf"])
            tolerance = 3 * u.arcsec
            if sigma_x > tolerance or sigma_y > tolerance:
                warnings.warn(f"Pointing unstable: StD(X) = {sigma_x}, StD(Y) = {sigma_y}.")
    else:
        if end_time is not None and indices.size < 2:
            times = times + (end_time - times) * 0.5

        # Interpolate all times
        x = (times - aux["time"][0]).to_value(u.s)
        xp = (aux["time"] - aux["time"][0]).to_value(u.s)

        roll = np.interp(x, xp, aux["roll_angle_rpy"][:, 0].value) << aux["roll_angle_rpy"].unit
        pitch = np.interp(x, xp, aux["roll_angle_rpy"][:, 1].value) << aux["roll_angle_rpy"].unit
        yaw = np.interp(x, xp, aux["roll_angle_rpy"][:, 2].value) << aux["roll_angle_rpy"].unit

        solo_heeq = (
            np.vstack(
                [
                    np.interp(x, xp, aux["solo_loc_heeq_zxy"][:, 0].value),
                    np.interp(x, xp, aux["solo_loc_heeq_zxy"][:, 1].value),
                    np.interp(x, xp, aux["solo_loc_heeq_zxy"][:, 2].value),
                ]
            ).T
            << aux["solo_loc_heeq_zxy"].unit
        )

        sas_x = np.interp(x, xp, aux["y_srf"])
        sas_y = np.interp(x, xp, aux["z_srf"])
        if x.size == 1:
            good_sas = [True] if np.interp(x, xp, aux["sas_ok"]).astype(bool) else []
        else:
            sas_ok = np.interp(x, xp, aux["sas_ok"]).astype(bool)
            good_sas = sas_ok[sas_ok == True]  # noqa E712

    # Convert the spacecraft pointing to STIX frame
    rotated_yaw = -yaw * np.cos(roll) + pitch * np.sin(roll)
    rotated_pitch = yaw * np.sin(roll) + pitch * np.cos(roll)
    spacecraft_pointing = np.vstack([STIX_X_SHIFT + rotated_yaw, STIX_Y_SHIFT + rotated_pitch]).T
    stix_pointing = spacecraft_pointing
    sas_pointing = np.vstack([sas_x + STIX_X_OFFSET, -1 * sas_y + STIX_Y_OFFSET]).T

    pointing_diff = np.sqrt(np.sum((spacecraft_pointing - sas_pointing) ** 2, axis=0))
    if np.all(np.isfinite(sas_pointing)) and len(good_sas) > 0:
        if np.max(pointing_diff) < 200 * u.arcsec:
            logger.info(f"Using SAS pointing: {sas_pointing}")
            stix_pointing = np.where(pointing_diff < 200 * u.arcsec, sas_pointing, spacecraft_pointing)
        else:
            warnings.warn(
                f"Using spacecraft pointing: {spacecraft_pointing}" f" large difference between SAS and spacecraft."
            )
    else:
        warnings.warn(f"SAS solution not available using spacecraft pointing: {stix_pointing}.")

    if end_time is not None or times.ndim == 0:
        solo_heeq = solo_heeq.squeeze()
        stix_pointing = stix_pointing.squeeze()

    return roll, solo_heeq, stix_pointing


@lru_cache
def _get_aux_data(start_time, end_time=None):
    r"""
    Search, download and read L2 pointing data.

    Parameters
    ----------
    start_time : `astropy.time.Time`
        Time or start of a time interval.

    Returns
    -------

    """
    if end_time is None:
        end_time = start_time
    # Find, download, read aux file with pointing, sas and position information
    logger.debug(f"Searching for AUX data: {start_time} - {end_time}")
    query = Fido.search(
        a.Time(start_time, end_time),
        a.Instrument.stix,
        a.Level.l2,
        a.stix.DataType.aux,
        a.stix.DataProduct.aux_ephemeris,
    )
    if len(query["stix"]) == 0:
        raise ValueError(f"No STIX pointing data found for time range {start_time} to {end_time}.")

    logger.debug(f"Downloading {len(query['stix'])} AUX files")
    aux_files = Fido.fetch(query["stix"])
    if len(aux_files.errors) > 0:
        raise ValueError("There were errors downloading the data.")
    # Read and extract data
    logger.debug("Loading and extracting AUX data")

    aux_data = []
    for aux_file in aux_files:
        hdu = fits.getheader(aux_file, ext=0)
        aux = QTable.read(aux_file, hdu=2)
        date_beg = Time(hdu.get("DATE-BEG"))
        aux["time"] = (
            date_beg + aux["time"] - 32 * u.s
        )  # Shift AUX data by half a time bin (starting time vs. bin centre)
        aux_data.append(aux)

    aux = vstack(aux_data)
    aux.sort(keys=["time"])

    return aux


@frame_transform_graph.transform(coord.FunctionTransform, STIXImaging, Helioprojective)
def stixim_to_hpc(stxcoord, hpcframe):
    r"""
    Transform STIX Imaging coordinates to given HPC frame
    """
    logger.debug("STIX: %s", stxcoord)

    rot_matrix, solo_pos_heeq = _get_rotation_matrix_and_position(stxcoord.obstime, stxcoord.obstime_end)

    obstime = stxcoord.obstime
    if stxcoord.obstime_end is not None:
        obstime = np.mean(Time([stxcoord.obstime, stxcoord.obstime_end]))

    solo_heeq = HeliographicStonyhurst(solo_pos_heeq.T, representation_type="cartesian", obstime=obstime)

    # Transform from STIX imaging to SOLO HPC
    newrepr = stxcoord.cartesian.transform(rot_matrix)

    # Create SOLO HPC
    solo_hpc = Helioprojective(
        newrepr,
        obstime=obstime,
        observer=solo_heeq.transform_to(HeliographicStonyhurst(obstime=obstime)),
    )
    logger.debug("SOLO HPC: %s", solo_hpc)

    # Transform from SOLO HPC to input HPC
    if hpcframe.observer is None:
        hpcframe = type(hpcframe)(obstime=hpcframe.obstime, observer=solo_heeq)
    hpc = solo_hpc.transform_to(hpcframe)
    logger.debug("Target HPC: %s", hpc)
    return hpc


@frame_transform_graph.transform(coord.FunctionTransform, Helioprojective, STIXImaging)
def hpc_to_stixim(hpccoord, stxframe):
    r"""
    Transform HPC coordinate to STIX Imaging frame.
    """
    logger.debug("Input HPC: %s", hpccoord)
    rmatrix, solo_pos_heeq = _get_rotation_matrix_and_position(stxframe.obstime, stxframe.obstime_end)

    obstime = stxframe.obstime
    if stxframe.obstime_end is not None:
        obstime = np.mean(Time([stxframe.obstime, stxframe.obstime_end]))
    solo_hgs = HeliographicStonyhurst(solo_pos_heeq.T, representation_type="cartesian", obstime=obstime)

    # Create SOLO HPC
    solo_hpc_frame = Helioprojective(obstime=obstime, observer=solo_hgs)
    # Transform input to SOLO HPC
    solo_hpc_coord = hpccoord.transform_to(solo_hpc_frame)
    logger.debug("SOLO HPC: %s", solo_hpc_coord)

    # Transform from SOLO HPC to STIX imaging
    newrepr = solo_hpc_coord.cartesian.transform(matrix_transpose(rmatrix))
    stx_coord = stxframe.realize_frame(newrepr)
    logger.debug("STIX: %s", newrepr)
    return stx_coord
