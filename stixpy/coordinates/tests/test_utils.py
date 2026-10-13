from contextlib import nullcontext

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.util.exceptions import SunpyUserWarning

from stixpy.coordinates.utils import heliocentric_angle

OBSTIME = Time("2022-03-30T17:30:00")


@pytest.fixture
def observer():
    return SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU, frame=HeliographicStonyhurst(obstime=OBSTIME))


def _flare_hpc(lon, lat, observer):
    """Flare on the solar surface at the given HGS position, as seen in HPC by ``observer``."""
    flare_hgs = SkyCoord(lon, lat, 1 * u.R_sun, frame=HeliographicStonyhurst(obstime=OBSTIME))
    return flare_hgs.transform_to(Helioprojective(observer=observer, obstime=OBSTIME))


def _spacecraft(lon, lat, distance, obstime=OBSTIME):
    return SkyCoord(lon, lat, distance, frame=HeliographicStonyhurst(obstime=obstime))


# In the equatorial plane, with the flare at lon 60 deg and the spacecraft at lon 0 and distance d (in R_sun),
# the normal is n = (cos 60, sin 60) and the flare-to-spacecraft vector is (d - cos 60, -sin 60).
_D = (0.5 * u.AU).to_value(u.R_sun)
_ANALYTIC = np.arccos((_D * np.cos(np.pi / 3) - 1) / np.hypot(_D - np.cos(np.pi / 3), np.sin(np.pi / 3))) << u.rad
# A spacecraft at angular separation arccos(R/d) from the flare sees it exactly on the limb (90 deg).
_HORIZON = np.arccos(1 / (0.4 * u.AU).to_value(u.R_sun)) << u.rad


@pytest.mark.parametrize(
    "flare_lonlat, spacecraft_lonlat, distance, expected",
    [
        pytest.param((0, 0) * u.deg, (0, 0) * u.deg, 1 * u.AU, 0 * u.deg, id="disk-centre"),
        pytest.param((60, 20) * u.deg, (60, 20) * u.deg, 0.3 * u.AU, 0 * u.deg, id="directly-above"),
        pytest.param((60, 0) * u.deg, (0, 0) * u.deg, 0.5 * u.AU, _ANALYTIC, id="analytic"),
        pytest.param((30, 0) * u.deg, (30 * u.deg + _HORIZON, 0 * u.deg), 0.4 * u.AU, 90 * u.deg, id="horizon"),
    ],
)
def test_known_angles(observer, flare_lonlat, spacecraft_lonlat, distance, expected):
    flare = _flare_hpc(*flare_lonlat, observer)
    spacecraft = _spacecraft(*spacecraft_lonlat, distance)
    angle = heliocentric_angle(flare, spacecraft)
    assert angle.unit == u.deg
    assert_quantity_allclose(angle, expected, atol=1e-5 * u.deg)


def test_defaults_to_coord_observer(observer):
    spacecraft = _spacecraft(-30 * u.deg, 5 * u.deg, 0.3 * u.AU)
    flare = _flare_hpc(10 * u.deg, -15 * u.deg, spacecraft)
    assert_quantity_allclose(heliocentric_angle(flare), heliocentric_angle(flare, spacecraft))


def test_no_observer_raises():
    flare = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(obstime=OBSTIME))
    with pytest.raises(ValueError, match="no observer"):
        heliocentric_angle(flare)


@pytest.mark.parametrize(
    "tolerance, context",
    [
        # pytest.ini turns warnings into errors, so any warning in the nullcontext case fails the test.
        pytest.param(1 * u.h, nullcontext(), id="within-tolerance"),
        pytest.param(1 * u.min, pytest.warns(UserWarning, match="differs from coord obstime"), id="mismatch"),
    ],
)
def test_obstime_tolerance(observer, tolerance, context):
    flare = _flare_hpc(10 * u.deg, 10 * u.deg, observer)
    spacecraft = _spacecraft(0 * u.deg, 0 * u.deg, 0.5 * u.AU, obstime=OBSTIME - 30 * u.min)
    with context:
        heliocentric_angle(flare, spacecraft, obstime_tolerance=tolerance)


def test_off_limb_flare_warns_and_returns_nan(observer):
    flare = SkyCoord(2000 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer=observer, obstime=OBSTIME))
    with (
        pytest.warns(SunpyUserWarning, match="all NaNs"),
        pytest.warns(UserWarning, match="off the limb"),
    ):
        angle = heliocentric_angle(flare)
    assert np.isnan(angle)
    assert angle.unit == u.deg


def test_coord_without_observer_attribute():
    flare = SkyCoord(30 * u.deg, 0 * u.deg, 1 * u.R_sun, frame=HeliographicStonyhurst(obstime=OBSTIME))
    with pytest.raises(ValueError, match="no observer"):
        heliocentric_angle(flare)
    observer = _spacecraft(30 * u.deg, 0 * u.deg, 0.5 * u.AU)
    assert_quantity_allclose(heliocentric_angle(flare, observer), 0 * u.deg, atol=1e-5 * u.deg)
