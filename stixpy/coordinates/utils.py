import warnings

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import HeliographicStonyhurst

__all__ = [
    "heliocentric_angle",
]


# TODO: Remove in favour of sunpy.coordinates.utils.get_heliocentric_angle once require sunpy>=7.1.
@u.quantity_input
def heliocentric_angle(
    coord: SkyCoord,
    observer: SkyCoord | None = None,
    obstime_tolerance: u.Quantity[u.s] = 1 * u.min,
) -> u.deg:
    r"""
    Calculate the heliocentric angle of a point on the solar surface as seen by an observer.

    The heliocentric angle is the angle between the local radial direction (the outward unit vector
    from Sun centre through the point) and the direction from the point to the observer. An angle of
    0 deg means the observer is directly above the point and 90 deg means the observer sees it on the
    limb. It is often given as :math:`\mu = \cos\theta`, e.g. for albedo corrections of X-ray spectra.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
        Position on the Sun, e.g. a flare location, in any frame that can be transformed to
        `~sunpy.coordinates.HeliographicStonyhurst`. A 2D `~sunpy.coordinates.Helioprojective`
        coordinate is projected onto the solar surface, so it must lie on the disk as seen by its
        observer.
    observer : `~astropy.coordinates.SkyCoord`, optional
        Observer position, usually Solar Orbiter, in any frame that can be transformed to
        `~sunpy.coordinates.HeliographicStonyhurst`. Defaults to the observer of ``coord``, which is
        correct when the position was derived from the observer's own data, e.g. by
        `~stixpy.imaging.flare_location.estimate_flare_location`.
    obstime_tolerance : `~astropy.units.Quantity`, optional
        Warn if ``observer`` and ``coord`` have observation times further apart than this. The
        observer position is not propagated in time, so it should be given for the time of ``coord``.

    Returns
    -------
    `~astropy.units.Quantity`
        The angle in degrees, or NaN deg (with a warning) if ``coord`` is off the limb.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
    >>> from stixpy.coordinates import heliocentric_angle
    >>> solo = SkyCoord(0 * u.deg, 0 * u.deg, 0.5 * u.AU, frame=HeliographicStonyhurst(obstime="2022-03-30"))
    >>> flare = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=solo, obstime="2022-03-30"))
    >>> heliocentric_angle(flare)  # doctest: +FLOAT_CMP
    <Quantity 0. deg>
    """
    if observer is None:
        observer = getattr(coord, "observer", None)
        if observer is None:
            raise ValueError("observer must be given if coord has no observer.")

    if observer.obstime is not None and abs(observer.obstime - coord.obstime) > obstime_tolerance:
        warnings.warn(
            f"Observer obstime {observer.obstime.isot} differs from coord obstime "
            f"{coord.obstime.isot} by more than {obstime_tolerance}.",
            stacklevel=2,
        )

    coord_hgs = coord.transform_to(HeliographicStonyhurst(obstime=coord.obstime))
    if np.isnan(coord_hgs.lat):
        warnings.warn(
            "Coordinate is off the limb so no angle can be calculated.",
            stacklevel=2,
        )
        return np.nan * u.deg

    point = coord_hgs.cartesian
    los = observer.transform_to(coord_hgs.frame).cartesian - point
    return np.arctan2(point.cross(los).norm(), point.dot(los)).to(u.deg)
