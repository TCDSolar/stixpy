import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

import sunpy.map
from sunpy.coordinates import Helioprojective, get_earth, sun
from sunpy.time import TimeRange

from stixpy.imaging.flare_location import calculate_sidelobes_ratio, estimate_flare_location, get_rsun_obs
from stixpy.product import Product


@pytest.fixture
def flare_cpd():
    return Product(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2022/08/28/SCI/"
        "solo_L1_stix-sci-xray-cpd_20220828T154401-20220828T161600_V02_2208284257-61808.fits"
    )


def test_get_rsun_obs():
    # cross-check against sunpy's own Earth-based angular radius calculation
    t = Time("2022-08-28T15:44:00")
    earth = get_earth(t)
    assert_quantity_allclose(get_rsun_obs(earth), sun.angular_radius(t), rtol=1e-5)


def test_calculate_sidelobes_ratio():
    data = np.zeros((50, 50))
    data[25, 25] = 10.0  # main peak
    data[5, 5] = 3.0  # sidelobe far outside the exclusion threshold
    ref_coord = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective(observer="earth", obstime="2022-08-28T15:44:00")
    )
    header = sunpy.map.make_fitswcs_header(data, ref_coord, scale=[20, 20] * u.arcsec / u.pix)
    test_map = sunpy.map.Map((data, header))

    ratio = calculate_sidelobes_ratio(test_map, threshold=50 * u.arcsec)
    assert ratio == pytest.approx(0.3)


# This short time bin doesn't give enough signal for a reliable back-projection image
# these tests only check the API (e.g. return keys/types, determinism, accepted time_range forms)
@pytest.mark.filterwarnings("ignore:Flare location may be unreliable")
@pytest.mark.remote_data
def test_estimate_flare_location(flare_cpd):
    time_range = TimeRange(flare_cpd.times[0], flare_cpd.times[2])
    energy_range = [6, 15] * u.keV

    results = estimate_flare_location(flare_cpd, time_range=time_range, energy_range=energy_range)

    assert set(results.keys()) == {"stx", "hpc", "sidelobes_ratio", "vis_tr"}
    assert isinstance(results["stx"], SkyCoord)
    assert isinstance(results["hpc"], SkyCoord)
    assert isinstance(results["vis_tr"], TimeRange)
    assert 0 <= results["sidelobes_ratio"] <= 1

    # result should be deterministic given the same inputs
    repeat = estimate_flare_location(flare_cpd, time_range=time_range, energy_range=energy_range)
    assert_quantity_allclose(results["stx"].data.lon, repeat["stx"].data.lon)
    assert_quantity_allclose(results["stx"].data.lat, repeat["stx"].data.lat)


@pytest.mark.filterwarnings("ignore:Flare location may be unreliable")
@pytest.mark.remote_data
def test_estimate_flare_location_accepts_list_time_range(flare_cpd):
    # time_range should also accept a plain [start, end] list, not just `TimeRange`
    time_range = [flare_cpd.times[0], flare_cpd.times[2]]
    energy_range = [6, 15] * u.keV

    results = estimate_flare_location(flare_cpd, time_range=time_range, energy_range=energy_range)

    assert isinstance(results["stx"], SkyCoord)
