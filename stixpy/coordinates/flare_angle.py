import warnings

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import HeliographicStonyhurst

__all__ = [
    "flare_spacecraft_angle",
]


@u.quantity_input
def flare_spacecraft_angle(
    flare_hpc: SkyCoord,
    spacecraft_coord: SkyCoord | None = None,
    obstime_tolerance: u.Quantity[u.s] = 1 * u.min,
) -> u.deg:
    r"""
    Calculate the angle between the flare's surface normal and the direction to the spacecraft.

    The surface normal is taken to be the local radial direction, the outward unit vector from Sun
    centre through the flare. An angle of 0 deg means the spacecraft is directly above the flare and
    90 deg means the spacecraft sees the flare on the limb.

    Parameters
    ----------
    flare_hpc : `~astropy.coordinates.SkyCoord`
        Flare position in a 2D `~sunpy.coordinates.Helioprojective` frame. It is projected onto the
        solar surface, so it must lie on the disk as seen by its observer.
    spacecraft_coord : `~astropy.coordinates.SkyCoord`, optional
        Spacecraft position in any frame that can be transformed to
        `~sunpy.coordinates.HeliographicStonyhurst`. Defaults to the observer of ``flare_hpc``, which
        is correct when the flare position was derived from the spacecraft's own data, e.g. by
        `~stixpy.imaging.flare_location.estimate_flare_location`.
    obstime_tolerance : `~astropy.units.Quantity`, optional
        Warn if ``spacecraft_coord`` and ``flare_hpc`` have observation times further apart than
        this. The spacecraft position is not propagated in time, so it should be given for the time
        of the flare position.

    Returns
    -------
    `~astropy.units.Quantity`
        The angle in degrees, or NaN deg (with a warning) if the flare is off the limb.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
    >>> from stixpy.coordinates.flare_angle import flare_spacecraft_angle
    >>> solo = SkyCoord(0 * u.deg, 0 * u.deg, 0.5 * u.AU, frame=HeliographicStonyhurst(obstime="2022-03-30"))
    >>> flare = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=solo, obstime="2022-03-30"))
    >>> flare_spacecraft_angle(flare)  # doctest: +FLOAT_CMP
    <Quantity 0. deg>
    """
    if spacecraft_coord is None:
        spacecraft_coord = flare_hpc.observer
        if spacecraft_coord is None:
            raise ValueError("spacecraft_coord must be given if flare_hpc has no observer.")

    if spacecraft_coord.obstime is not None and abs(spacecraft_coord.obstime - flare_hpc.obstime) > obstime_tolerance:
        warnings.warn(
            f"Spacecraft obstime {spacecraft_coord.obstime.isot} differs from flare obstime "
            f"{flare_hpc.obstime.isot} by more than {obstime_tolerance}.",
            stacklevel=2,
        )

    flare_hgs = flare_hpc.transform_to(HeliographicStonyhurst(obstime=flare_hpc.obstime))
    if np.isnan(flare_hgs.lat):
        warnings.warn(
            "Flare location is off the limb so no angle can be calculated.",
            stacklevel=2,
        )
        return np.nan * u.deg

    flare = flare_hgs.cartesian
    los = spacecraft_coord.transform_to(flare_hgs.frame).cartesian - flare
    return np.arctan2(flare.cross(los).norm(), flare.dot(los)).to(u.deg)
