import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from stixpy.calibration.visibility import create_meta_pixels, get_uv_points_data
from stixpy.coordinates.frames import STIXImaging
from stixpy.product import Product


@pytest.fixture
@pytest.mark.remote_data
def flare_cpd():
    return Product(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2022/08/28/SCI/"
        "solo_L1_stix-sci-xray-cpd_20220828T154401-20220828T161600_V02_2208284257-61808.fits"
    )


@pytest.fixture
@pytest.mark.remote_data
def background_cpd():
    return Product(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2022/08/24/SCI/"
        "solo_L1_stix-sci-xray-cpd_20220824T140037-20220824T145017_V02_2208242079-61784.fits"
    )


def test_get_uv_points_data():
    uv_data = get_uv_points_data()
    assert uv_data["u"][0] == -0.03333102271709666 / u.arcsec
    assert uv_data["v"][0] == 0.005908224739704219 / u.arcsec
    assert uv_data["isc"][0] == 1
    assert uv_data["label"][0] == "3c"


def test_create_meta_pixels(background_cpd):
    time_range = Time(["2022-08-24T14:00:37.271", "2022-08-24T14:50:17.271"])
    energy_range = [20, 76] * u.keV
    meta_pixels = create_meta_pixels(
        background_cpd,
        time_range=time_range,
        energy_range=energy_range,
        flare_location=STIXImaging(0 * u.arcsec, 0 * u.arcsec),
        no_shadowing=True,
    )

    assert_quantity_allclose(
        [0.17628464, 0.17251869, 0.16275495, 0.19153585] * u.ct / (u.keV * u.cm**2 * u.s),
        meta_pixels["abcd_rate_kev_cm"][0, :],
    )

    assert_quantity_allclose(
        [0.18701911, 0.18205339, 0.18328145, 0.17945563] * u.ct / (u.keV * u.cm**2 * u.s),
        meta_pixels["abcd_rate_kev_cm"][-1, :],
    )
