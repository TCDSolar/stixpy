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


def _get_rotation_matrix_and_position(obstime):
    r"""
    Return rotation matrix STIX Imaging to SOLO HPC and position of SOLO in HEEQ.

    Parameters
    ----------
    obstime : `astropy.time.Time`
        Time
    Returns
    -------
    """
    roll, solo_position_heeq, spacecraft_pointing = get_hpc_info(obstime)

    # Generate the rotation matrix using the x-convention (see Goldstein)
    # Rotate +90 clockwise around the first axis
    # STIX is mounted on the -Y panel (SOLO_SRF) need to rotate to from STIX frame to SOLO_SRF
    rot_to_solo = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    logger.debug("Pointing: %s.", spacecraft_pointing)
    A = rotation_matrix(-1 * roll, "x")
    B = rotation_matrix(spacecraft_pointing[1], "y")
    C = rotation_matrix(-1 * spacecraft_pointing[0], "z")

    # Will be applied right to left
    rmatrix = A @ B @ C @ rot_to_solo

    return rmatrix, solo_position_heeq


def get_hpc_info(start_time, end_time=None):
    r"""
    Get STIX pointing and SO location from L2 aspect files.

    Parameters
    ----------
    start_time : `astropy.time.Time`
        Time or start of a time interval.
    end_time : `astropy.time.Time`, optional
        End of time interval

    Returns
    -------

    """
    aux = _get_aux_data(start_time, end_time=end_time)
    if end_time is None:
        end_time = start_time
    indices = np.argwhere((aux["time"] >= start_time) & (aux["time"] <= end_time))
    indices = indices.flatten()

    if indices.size < 2:
        logger.info("Only one data point found interpolating between two closest times.")
        # Times contained in one time bin have to interpolate
        time_center = start_time + (end_time - start_time) * 0.5
        diff_center = (aux["time"] - time_center).to("s")
        closest_index = np.argmin(np.abs(diff_center))

        if diff_center[closest_index] > 0:
            start_ind = closest_index - 1
            end_ind = closest_index + 1
        else:
            start_ind = closest_index
            end_ind = closest_index + 2

        interp = slice(start_ind, end_ind)
        dt = np.diff(aux[start_ind:end_ind]["time"])[0].to(u.s)

        roll, pitch, yaw = (
            aux[start_ind : start_ind + 1]["roll_angle_rpy"]
            + (np.diff(aux[interp]["roll_angle_rpy"], axis=0) / dt) * diff_center[closest_index]
        )[0]
        solo_heeq = (
            aux[start_ind : start_ind + 1]["solo_loc_heeq_zxy"]
            + (np.diff(aux[interp]["solo_loc_heeq_zxy"], axis=0) / dt) * diff_center[closest_index]
        )[0]
        sas_x = (
            aux[start_ind : start_ind + 1]["y_srf"] + (np.diff(aux[interp]["y_srf"]) / dt) * diff_center[closest_index]
        )[0]
        sas_y = (
            aux[start_ind : start_ind + 1]["z_srf"] + (np.diff(aux[interp]["z_srf"]) / dt) * diff_center[closest_index]
        )[0]

        good_solution = np.where(aux[interp]["sas_ok"] == 1)
        good_sas = aux[good_solution]
    else:
        # average over
        aux = aux[indices]

        roll, pitch, yaw = np.mean(aux["roll_angle_rpy"], axis=0)
        solo_heeq = np.mean(aux["solo_loc_heeq_zxy"], axis=0)

        good_solution = np.where(aux["sas_ok"] == 1)
        good_sas = aux[good_solution]

        if len(good_sas) == 0:
            warnings.warn(f"No good SAS solution found for time range: {start_time} to {end_time}.")
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

    # Convert the spacecraft pointing to STIX frame
    rotated_yaw = -yaw * np.cos(roll) + pitch * np.sin(roll)
    rotated_pitch = yaw * np.sin(roll) + pitch * np.cos(roll)
    spacecraft_pointing = np.hstack([STIX_X_SHIFT + rotated_yaw, STIX_Y_SHIFT + rotated_pitch])
    stix_pointing = spacecraft_pointing
    sas_pointing = np.hstack([sas_x + STIX_X_OFFSET, -1 * sas_y + STIX_Y_OFFSET])

    pointing_diff = np.linalg.norm(spacecraft_pointing - sas_pointing)
    if np.all(np.isfinite(sas_pointing)) and len(good_sas) > 0:
        if pointing_diff < 200 * u.arcsec:
            logger.info(f"Using SAS pointing: {sas_pointing}")
            stix_pointing = sas_pointing
        else:
            warnings.warn(
                f"Using spacecraft pointing: {spacecraft_pointing}" f" large difference between SAS and spacecraft."
            )
    else:
        warnings.warn(f"SAS solution not available using spacecraft pointing: {stix_pointing}.")

    return roll, solo_heeq, stix_pointing


@lru_cache
def _get_aux_data(start_time, end_time=None):
    r"""
    Search, download and read L2 pointing data.

    Parameters
    ----------
    start_time : `astropy.time.Time`
        Time or start of a time interval.
    end_time : `astropy.time.Time`, optional
        End of time interval

    Returns
    -------

    """
    # Find, download, read aux file with pointing, sas and position information
    if end_time is None:
        end_time = start_time
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

    rot_matrix, solo_pos_heeq = _get_rotation_matrix_and_position(stxcoord.obstime)
    solo_heeq = HeliographicStonyhurst(solo_pos_heeq, representation_type="cartesian", obstime=stxcoord.obstime)

    # Transform from STIX imaging to SOLO HPC
    newrepr = stxcoord.cartesian.transform(rot_matrix)

    # Create SOLO HPC
    solo_hpc = Helioprojective(
        newrepr,
        obstime=stxcoord.obstime,
        observer=solo_heeq.transform_to(HeliographicStonyhurst(obstime=stxcoord.obstime)),
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
    rmatrix, solo_pos_heeq = _get_rotation_matrix_and_position(stxframe.obstime)
    solo_hgs = HeliographicStonyhurst(solo_pos_heeq, representation_type="cartesian", obstime=stxframe.obstime)

    # Create SOLO HPC
    solo_hpc_frame = Helioprojective(obstime=stxframe.obstime, observer=solo_hgs)
    # Transform input to SOLO HPC
    solo_hpc_coord = hpccoord.transform_to(solo_hpc_frame)
    logger.debug("SOLO HPC: %s", solo_hpc_coord)

    # Transform from SOLO HPC to STIX imaging
    newrepr = solo_hpc_coord.cartesian.transform(matrix_transpose(rmatrix))
    stx_coord = stxframe.realize_frame(newrepr, obstime=solo_hpc_frame.obstime)
    logger.debug("STIX: %s", newrepr)
    return stx_coord
