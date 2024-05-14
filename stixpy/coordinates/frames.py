import astropy
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import QuantityAttribute
from astropy.wcs import WCS
from sunpy.coordinates.frameattributes import ObserverCoordinateAttribute
from sunpy.coordinates.frames import HeliographicStonyhurst, SunPyBaseCoordinateFrame
from sunpy.sun.constants import radius as _RSUN

from stixpy.net.client import STIXClient  # noqa
from stixpy.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["STIXImaging"]


STIX_X_CTYPE = "SXLN-TAN"
STIX_Y_CTYPE = "SXLT-TAN"


class STIXImaging(SunPyBaseCoordinateFrame):
    r"""
    STIX Imaging Frame

    - ``Tx`` (aka "theta_x") along the pixel rows (e.g. 0, 1, 2, 3; 4, 5, 6, 8).
    - ``Ty`` (aka "theta_y") along the pixel columns (e.g. A, B, C, D).
    - ``distance`` is the Sun-observer distance.

    Aligned with STIX 'pixels' +X corresponds direction along pixel row toward pixels
    0, 4 and +Y corresponds direction along columns towards pixels 0, 1, 2, 3.

    .. code-block:: text

      Col\Row
            ---------
           | 3  | 7  |
        D  |   _|_   |
           |  |11 |  |
            ---------
           | 2  | 6  |
        C  |   _|_   |
           |  |10 |  |
            ---------
           | 1  | 5  |
        B  |   _|_   |
           |  | 9 |  |
            ---------
           | 0  | 4  |
        A  |   _|_   |
           |  | 8 |  |
            ---------

    """
    observer = ObserverCoordinateAttribute(HeliographicStonyhurst)
    rsun = QuantityAttribute(default=_RSUN, unit=u.km)

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "Tx", u.arcsec),
            coord.RepresentationMapping("lat", "Ty", u.arcsec),
            coord.RepresentationMapping("distance", "distance"),
        ],
        coord.UnitSphericalRepresentation: [coord.RepresentationMapping('lon', 'Tx', u.arcsec),
                                            coord.RepresentationMapping('lat', 'Ty', u.arcsec)]
    }


def stix_wcs_to_frame(wcs):
    r"""
    This function registers the coordinate frames to their FITS-WCS coordinate
    type values in the `astropy.wcs.utils.wcs_to_celestial_frame` registry.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`

    Returns
    -------
    `astropy.coordinates.BaseCoordinateFrame`
    """
    if hasattr(wcs, "coordinate_frame"):
        return wcs.coordinate_frame

    # Not a STIX wcs bail out early
    if set(wcs.wcs.ctype) != {STIX_X_CTYPE, STIX_Y_CTYPE}:
        return None

    dateobs = wcs.wcs.dateobs

    rsun = wcs.wcs.aux.rsun_ref
    if rsun is not None:
        rsun *= u.m

    hgs_longitude = wcs.wcs.aux.hgln_obs
    hgs_latitude = wcs.wcs.aux.hglt_obs
    hgs_distance = wcs.wcs.aux.dsun_obs

    observer = HeliographicStonyhurst(lat=hgs_latitude * u.deg,
                                      lon=hgs_longitude * u.deg,
                                      radius=hgs_distance * u.m,
                                      obstime=dateobs,
                                      rsun=rsun)

    frame_args = {'obstime': dateobs,
                  'observer': observer,
                  'rsun': rsun}

    return STIXImaging(**frame_args)


def stix_frame_to_wcs(frame, projection='TAN'):
    r"""
    For a given frame, this function returns the corresponding WCS object.

    It registers the WCS coordinates types from their associated frame in the
    `astropy.wcs.utils.celestial_frame_to_wcs` registry.

    Parameters
    ----------
    frame : `astropy.coordinates.BaseCoordinateFrame`
    projection : `str`, optional

    Returns
    -------
    `astropy.wcs.WCS`
    """
    # Bail out early if not STIXImaging frame
    if not isinstance(frame, STIXImaging):
        return None

    wcs = WCS(naxis=2)
    wcs.wcs.aux.rsun_ref = frame.rsun.to_value(u.m)

    # Sometimes obs_coord can be a SkyCoord, so convert down to a frame
    obs_frame = frame.observer
    if hasattr(obs_frame, 'frame'):
        obs_frame = frame.observer.frame

    wcs.wcs.aux.hgln_obs = obs_frame.lon.to_value(u.deg)
    wcs.wcs.aux.hglt_obs = obs_frame.lat.to_value(u.deg)
    wcs.wcs.aux.dsun_obs = obs_frame.radius.to_value(u.m)

    wcs.wcs.dateobs = frame.obstime.utc.iso
    wcs.wcs.cunit = ['arcsec', 'arcsec']
    wcs.wcs.ctype = [STIX_X_CTYPE, STIX_Y_CTYPE]

    return wcs


# Remove once min version of sunpy has https://github.com/sunpy/sunpy/pull/7594
astropy.wcs.utils.WCS_FRAME_MAPPINGS.insert(1, [stix_wcs_to_frame])
astropy.wcs.utils.FRAME_WCS_MAPPINGS.insert(1, [stix_frame_to_wcs])

STIX_CTYPE_TO_UCD1 = {
    "SXLT": "custom:pos.stiximaging.lat",
    "SXLN": "custom:pos.stiximaging.lon",
    "SXRZ": "custom:pos.stiximaging.z"
}
astropy.wcs.wcsapi.fitswcs.CTYPE_TO_UCD1.update(STIX_CTYPE_TO_UCD1)
