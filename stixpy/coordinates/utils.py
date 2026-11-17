import numpy as np

import astropy.units as u
from astropy.coordinates import CartesianRepresentation

__all__ = ["get_heliocentric_angle"]


def _get_heliocentric_angle(coordinate_on_solar_disk):
    r"""
    Returns the heliocentric angle, the angle between the observer look direction
    for a point on the surface of the Sun and the local vertical for that point.

    If a point is on the visible side of the Sun for the observer, the angle ranges
    between 0 :math:`^\circ` (at disk center) and 90 :math:`^\circ` (at the solar
    limb). A point on the far side of the Sun has to be provided as a 3D coordinate,
    and then the angle ranges from 90 :math:`^\circ` to 180 :math:`^\circ`.

    The heliocentric angle is related to the parameter :math:`\mu` commonly used for
    limb-darkening calculations:

    .. math::

        \mu = \cos(heliocentric\_angle)

    Copy of `sunpy.coordinates.utils.get_heliocentric_angle`, which was added in sunpy 7.1.

    Parameters
    ----------
    coordinate_on_solar_disk : `astropy.coordinates.SkyCoord`
        A coordinate on the solar disk, requires the observer and obstime to be set.

    Returns
    -------
    heliocentric_angle : `~astropy.units.Quantity`
        The angle between the local solar vertical and the line of sight from
        the observer to a point on the solar disk.

    Examples
    --------
    >>> import numpy as np

    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord

    >>> from stixpy.coordinates.utils import get_heliocentric_angle

    >>> # At the center of the solar disk
    >>> hpc_coord_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame='helioprojective', observer="earth", obstime="2017-07-26")
    >>> get_heliocentric_angle(hpc_coord_center)
    <Quantity 0. deg>
    >>> # mu
    >>> np.cos(get_heliocentric_angle(hpc_coord_center))
    <Quantity 1.>

    >>> # Almost at the limb
    >>> hpc_coord_limb = SkyCoord(944.35*u.arcsec, 0*u.arcsec, frame='helioprojective', observer="earth", obstime="2017-07-26")
    >>> get_heliocentric_angle(hpc_coord_limb)
    <Quantity 89.26429919 deg>
    >>> # mu
    >>> np.cos(get_heliocentric_angle(hpc_coord_limb))
    <Quantity 0.01284005>
    """
    hcc = coordinate_on_solar_disk.heliocentric
    normal = hcc.cartesian
    to_observer = CartesianRepresentation(0, 0, 1) * hcc.observer.radius - normal
    return np.arctan2(normal.cross(to_observer).norm(), normal.dot(to_observer)).to(u.deg)


# TODO: Once stixpy requires sunpy>=7.1, import get_heliocentric_angle from sunpy.coordinates.utils and delete this module.
try:
    from sunpy.coordinates.utils import get_heliocentric_angle
except ImportError:
    get_heliocentric_angle = _get_heliocentric_angle
