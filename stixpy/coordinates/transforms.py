import warnings

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_product, matrix_transpose, rotation_matrix
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.net import Fido
from sunpy.net import attrs as a

from stixpy.coordinates.frames import STIX_X_OFFSET, STIX_X_SHIFT, STIX_Y_OFFSET, STIX_Y_SHIFT, STIXImaging, logger

__all__ = ['get_hpc_info', 'stixim_to_hpc', 'hpc_to_stixim']

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
    rmatrix = matrix_product(A, B, C, rot_to_solo)

    return rmatrix, solo_position_heeq


def get_hpc_info(obstime):
    roll, pitch, yaw, sas_x, sas_y, solo_position_heeq = _get_aux_data(obstime)
    if np.isnan(sas_x) or np.isnan(sas_y):
        warnings.warn("SAS solution not available using spacecraft pointing and average SAS offset")
        sas_x = yaw
        sas_y = pitch
    # Convert the spacecraft pointing to STIX frame
    rotated_yaw = -yaw * np.cos(roll) + pitch * np.sin(roll)
    rotated_pitch = yaw * np.sin(roll) + pitch * np.cos(roll)
    spacecraft_pointing = np.hstack([STIX_X_SHIFT + rotated_yaw, STIX_Y_SHIFT + rotated_pitch])
    sas_pointing = np.hstack([sas_x + STIX_X_OFFSET, -1 * sas_y + STIX_Y_OFFSET])
    pointing_diff = np.linalg.norm(spacecraft_pointing - sas_pointing)
    if pointing_diff < 200 * u.arcsec:
        logger.debug(f"Using SAS pointing {sas_pointing}")
        spacecraft_pointing = sas_pointing
    else:
        logger.debug(f"Using spacecraft pointing {spacecraft_pointing}")
        warnings.warn(
            f"Using spacecraft pointing large difference between "
            f"SAS {sas_pointing} and spacecraft {spacecraft_pointing}."
        )
    return roll, solo_position_heeq, spacecraft_pointing


def _get_aux_data(obstime):
    # Find, download, read aux file with pointing, sas and position information
    logger.debug("Searching for AUX data")
    query = Fido.search(
        a.Time(obstime, obstime), a.Instrument.stix, a.Level.l2, a.stix.DataType.aux, a.stix.DataProduct.aux_ephemeris
    )
    if len(query["stix"]) != 1:
        raise ValueError("Exactly one AUX file should be found but %d were found.", len(query["stix"]))
    logger.debug("Downloading AUX data")
    files = Fido.fetch(query)
    if len(files.errors) > 0:
        raise ValueError("There were errors downloading the data.")
    # Read and extract data
    logger.debug("Loading and extracting AUX data")
    hdu = fits.getheader(files[0], ext=0)
    aux = QTable.read(files[0], hdu=2)
    start_time = Time(hdu.get("DATE-BEG"))
    rel_date = (obstime - start_time).to("s")
    idx = np.argmin(np.abs(rel_date - aux["time"]))
    sas_x, sas_y = aux[idx][["y_srf", "z_srf"]]
    roll, pitch, yaw = aux[idx]["roll_angle_rpy"]
    solo_position_heeq = aux[idx]["solo_loc_heeq_zxy"]
    logger.debug(
        "SAS: %s, %s, Orientation: %s, %s, %s, Position: %s",
        sas_x,
        sas_y,
        roll,
        yaw.to("arcsec"),
        pitch.to("arcsec"),
        solo_position_heeq,
    )

    # roll, pitch, yaw = np.mean(aux[1219:1222]['roll_angle_rpy'], axis=0)
    # sas_x = np.mean(aux[1219:1222]['y_srf'])
    # sas_y = np.mean(aux[1219:1222]['z_srf'])

    return roll, pitch, yaw, sas_x, sas_y, solo_position_heeq


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
        newrepr, obstime=stxcoord.obstime, observer=solo_heeq.transform_to(HeliographicStonyhurst)
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
    logger.debug("STIX: %s", stx_coord)
    return stx_coord
