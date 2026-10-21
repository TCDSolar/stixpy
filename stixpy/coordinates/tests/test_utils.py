import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.util.exceptions import SunpyUserWarning

from stixpy.coordinates.utils import _get_heliocentric_angle, get_heliocentric_angle

OBSTIME = Time("2022-03-30T17:30:00")

# In the equatorial plane, with the flare at lon 60 deg and the spacecraft at lon 0 and distance d (in R_sun),
# the normal is n = (cos 60, sin 60) and the flare-to-spacecraft vector is (d - cos 60, -sin 60).
_D = (0.5 * u.AU).to_value(u.R_sun)
_ANALYTIC = np.arccos((_D * np.cos(np.pi / 3) - 1) / np.hypot(_D - np.cos(np.pi / 3), np.sin(np.pi / 3))) << u.rad
# A spacecraft at angular separation arccos(R/d) from the flare sees it exactly on the limb (90 deg).
_HORIZON = np.arccos(1 / (0.4 * u.AU).to_value(u.R_sun)) << u.rad

# The local copy is what sunpy<7.1 uses, so test it as well as whichever version is imported.
implementations = pytest.mark.parametrize(
    "func", [get_heliocentric_angle, _get_heliocentric_angle], ids=["default", "fallback"]
)


def _spacecraft(lon, lat, distance):
    return SkyCoord(lon, lat, distance, frame=HeliographicStonyhurst(obstime=OBSTIME))


@implementations
@pytest.mark.parametrize(
    "flare_lonlat, spacecraft_lonlat, distance, expected",
    [
        pytest.param((0, 0) * u.deg, (0, 0) * u.deg, 1 * u.AU, 0 * u.deg, id="disk-centre"),
        pytest.param((60, 20) * u.deg, (60, 20) * u.deg, 0.3 * u.AU, 0 * u.deg, id="directly-above"),
        pytest.param((60, 0) * u.deg, (0, 0) * u.deg, 0.5 * u.AU, _ANALYTIC, id="analytic"),
        pytest.param((30, 0) * u.deg, (30 * u.deg + _HORIZON, 0 * u.deg), 0.4 * u.AU, 90 * u.deg, id="horizon"),
    ],
)
def test_known_angles(func, flare_lonlat, spacecraft_lonlat, distance, expected):
    spacecraft = _spacecraft(*spacecraft_lonlat, distance)
    flare = SkyCoord(*flare_lonlat, 1 * u.R_sun, frame=HeliographicStonyhurst(obstime=OBSTIME))
    flare_hpc = flare.transform_to(Helioprojective(observer=spacecraft, obstime=OBSTIME))
    angle = func(flare_hpc)
    assert angle.unit == u.deg
    assert_quantity_allclose(angle, expected, atol=1e-5 * u.deg)


@implementations
def test_off_limb_is_nan(func):
    observer = _spacecraft(0 * u.deg, 0 * u.deg, 1 * u.AU)
    flare = SkyCoord(2000 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=observer, obstime=OBSTIME))
    with pytest.warns(SunpyUserWarning, match="all NaNs"):
        angle = func(flare)
    assert np.isnan(angle)
    assert angle.unit == u.deg
