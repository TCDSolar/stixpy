import astropy.coordinates as coord
import astropy.units as u
import numpy as np

from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from sunpy.coordinates.frames import HeliographicStonyhurst, Helioprojective, \
    SunPyBaseCoordinateFrame
from sunpy.coordinates.transformations import _check_observer_defined
from sunpy.net import Fido, attrs as a

from stixcore.util.logging import get_logger
from stixpy.net.client import STIXClient  # noqa

logger = get_logger(__name__, 'DEBUG')

__all__ = ['STIXImagingFrame', '_get_rotation_matrix_and_position', 'hpc_to_stixim', 'hpc_to_stixim']

STIX_X_SHIFT = 26.1 * u.arcsec  # fall back to this when non sas solution available
STIX_Y_SHIFT = 58.2 * u.arcsec  # fall back to this when non sas solution available
STIX_X_OFFSET = 60.0 * u.arcsec  # remaining offset after SAS solution
STIX_Y_OFFSET = 8.0 * u.arcsec  # remaining offset after SAS solution


class STIXImagingFrame(SunPyBaseCoordinateFrame):
    r"""
    A coordinate or frame in the STIX Imaging system

    - ``x`` (aka "theta_x") is the angle relative to the plane containing the Sun-observer line
      and the Sun's rotation axis, with positive values in the direction of the Sun's west limb.
    - ``Ty`` (aka "theta_y") is the angle relative to the Sun's equatorial plane, with positive
      values in the direction of the Sun's north pole.
    - ``distance`` is the Sun-observer distance.
    """

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'x', u.arcsec),
            coord.RepresentationMapping('lat', 'y', u.arcsec),
            coord.RepresentationMapping('distance', 'distance')]

    }


def _get_rotation_matrix_and_position(obstime):
    r"""
    Return the rotation matrix for the STIX Imaging coordinate system and position of SOLO

    Parameters
    ----------
    obstime : `astropy.time.Time`

    Returns
    -------

    """
    # Find, download, read aux file with pointing, sas and position information
    logger.debug('Searching for AUX data')
    query = Fido.search(a.Time(obstime, obstime), a.Instrument.stix, a.Level.l2,
                        a.stix.DataType.aux, a.stix.DataProduct.aux_auxiliary)
    if len(query['stix']) != 1:
        raise ValueError('Exactly one AUX file should be found but %d were found.', len(query['stix']))
    logger.debug('Downloading AUX data')
    files = Fido.fetch(query)
    if len(files.errors) > 0:
        raise ValueError('There were errors downloading the data.')

    # Read and extract data
    logger.debug('Loading and extracting AUX data')
    hdu = fits.getheader(files[0], ext=0)
    aux = QTable.read(files[0], hdu=2)
    start_time = Time(hdu.get('DATE-BEG'))
    rel_date = (obstime - start_time).to('s')
    idx = np.argmin(np.abs(rel_date - aux['time']))
    sas_x, sas_y = aux[idx][['y_srf', 'z_srf']]
    roll, yaw, pitch = aux[idx]['roll_angle_rpy']
    solo_position_heeq = aux[idx]['solo_loc_heeq_zxy']
    if np.isnan(sas_x) or np.isnan(sas_y):
        logger.warning('SAS solution not available using spacecraft pointing and average SAS offset')
        sas_x = yaw
        sas_y = pitch

    # Generate the rotation matrix using the x-convention (see Goldstein)
    C = rotation_matrix(sas_x + STIX_X_OFFSET, "z")
    B = rotation_matrix(-1*sas_y + STIX_Y_OFFSET, "y")
    A = rotation_matrix(-1*roll, 'x')
    rmatrix = matrix_product(A, B, C)

    return rmatrix, solo_position_heeq


@frame_transform_graph.transform(coord.FunctionTransform, STIXImagingFrame, Helioprojective)
def stixim_to_hpc(stxcoord, hpcframe):
    r"""
    Transform from stix imaging coordiantes to HPC
    """
    logger.debug('STIX: %s', stxcoord)
    _check_observer_defined(hpcframe)

    rot_matrix, solo_pos_heeq = _get_rotation_matrix_and_position(stxcoord.obstime)

    solo_heeq = HeliographicStonyhurst(solo_pos_heeq, representation='cartesian',
                                       obstime=stxcoord.obstime)

    # Transform from STIX imaging to SOLO HPC
    newrepr = stxcoord.cartesian.transform(rot_matrix)

    # Create SOLO HPC
    solo_hpc = Helioprojective(newrepr, obstime=stxcoord.obstime,
                               observer=solo_heeq.transform_to(HeliographicStonyhurst))
    logger.debug('SOLO HPC: %s', solo_hpc)

    # Transform from SOLO HPC to input HPC
    hpc = solo_hpc.transform_to(hpcframe)
    logger.debug('Target HPC: %s', hpc)
    return hpc


@frame_transform_graph.transform(coord.FunctionTransform, Helioprojective, STIXImagingFrame)
def hpc_to_stixim(hpccoord, stxframe):
    r"""
    Transform from HPC to stix imaging coordiantes.
    """
    logger.debug('Input HPC: %s', hpccoord)
    rmatrix, solo_pos_heeq = _get_rotation_matrix_and_position(hpccoord.obstime)

    solo_heeq = HeliographicStonyhurst(solo_pos_heeq, representation='cartesian',
                                       obstime=stxframe.obstime)

    # Create SOLO HPC
    solo_hpc_frame = Helioprojective(obstime=stxframe.obstime,
                                     observer=solo_heeq.transform_to(HeliographicStonyhurst))
    solo_hpc_coord = hpccoord.transform_to(solo_hpc_frame)
    logger.debug('SOLO HPC: %s', solo_hpc_coord)

    newrepr = solo_hpc_coord.cartesian.transform(matrix_transpose(rmatrix))

    stx_coord = stxframe.realize_frame(newrepr, obstime=solo_hpc_frame.obstime)
    logger.debug('STIX: %s', stx_coord)
    return stx_coord
