import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from numpy.ma.testutils import assert_equal
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.time import TimeRange

from stixpy.calibration.visibility import (
    calibrate_visibility,
    create_meta_pixels,
    create_visibility,
    get_uv_points_data,
)
from stixpy.coordinates.frames import STIXImaging
from stixpy.io.readers import read_det_adc_mapping
from stixpy.product import Product


@pytest.fixture
def flare_cpd():
    return Product(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2022/08/28/SCI/"
        "solo_L1_stix-sci-xray-cpd_20220828T154401-20220828T161600_V02_2208284257-61808.fits"
    )


@pytest.fixture
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


def test_get_read_det_adc_mapping():
    det_map = np.array(read_det_adc_mapping()["Adc #"])
    assert (
        det_map
        == [0, 0, 7, 7, 2, 1, 1, 6, 6, 5, 2, 3, 3, 4, 4, 5, 13, 12, 12, 11, 11, 10, 13, 14, 14, 9, 9, 10, 15, 15, 8, 8]
    ).all()


@pytest.mark.parametrize(
    "pixel_set,expected_abcd_rate_kev,expected_abcd_rate_kev_cm0,expected_abcd_rate_kev_cm1,areas",
    [
        (
            "all",
            [0.03509001, 0.03432438, 0.03248172, 0.03821136] * u.ct / u.keV / u.s,
            [0.17336964, 0.16958685, 0.1604828, 0.1887913] * u.ct / u.keV / u.s / u.cm**2,
            [0.18532299, 0.17855843, 0.1802053, 0.17609046] * u.ct / u.keV / u.s / u.cm**2,
            [0.20239999, 0.20239999, 0.20239999, 0.20239999] * u.cm**2,
        ),
        (
            "top+bot",
            [0.0339154, 0.03319087, 0.03131242, 0.03684958] * u.ct / u.keV / u.s,
            [0.17628464, 0.17251869, 0.16275495, 0.19153585] * u.ct / u.keV / u.s / u.cm**2,
            [0.18701911, 0.18205339, 0.18328145, 0.17945563] * u.ct / u.keV / u.s / u.cm**2,
            [0.192389994, 0.192389994, 0.192389994, 0.192389994] * u.cm**2,
        ),
        (
            "small",
            [0.00117461, 0.00113351, 0.00116929, 0.00136178] * u.ct / u.keV / u.s,
            [0.1173439, 0.11323753, 0.11681252, 0.13604156] * u.ct / u.keV / u.s / u.cm**2,
            [0.15272384, 0.11138607, 0.12108232, 0.11141279] * u.ct / u.keV / u.s / u.cm**2,
            [0.010009999, 0.010009999, 0.010009999, 0.010009999] * u.cm**2,
        ),
        (
            "top",
            [0.01742041, 0.01738642, 0.01624934, 0.01833627] * u.ct / u.keV / u.s,
            [0.18109474, 0.18074145, 0.16892087, 0.19061566] * u.ct / u.keV / u.s / u.cm**2,
            [0.18958941, 0.17299885, 0.17864632, 0.17571344] * u.ct / u.keV / u.s / u.cm**2,
            [0.096194997, 0.096194997, 0.096194997, 0.096194997] * u.cm**2,
        ),
        (
            "bot",
            [0.01649499, 0.01580445, 0.01506308, 0.01851331] * u.ct / u.keV / u.s,
            [0.17147454, 0.16429592, 0.15658903, 0.19245605] * u.ct / u.keV / u.s / u.cm**2,
            [0.18444881, 0.19110794, 0.18791658, 0.18319781] * u.ct / u.keV / u.s / u.cm**2,
            [0.096194997, 0.096194997, 0.096194997, 0.096194997] * u.cm**2,
        ),
    ],
)
def test_create_meta_pixels(
    background_cpd, pixel_set, expected_abcd_rate_kev, expected_abcd_rate_kev_cm0, expected_abcd_rate_kev_cm1, areas
):
    time_range = Time(["2022-08-24T14:00:37.271", "2022-08-24T14:50:17.271"])
    energy_range = [20, 76] * u.keV
    meta_pixels = create_meta_pixels(
        background_cpd,
        time_range=time_range,
        energy_range=energy_range,
        flare_location=SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=STIXImaging),
        pixels=pixel_set,
        no_shadowing=True,
    )

    assert_quantity_allclose(expected_abcd_rate_kev, meta_pixels["abcd_rate_kev"][0, :], atol=1e-7 * u.ct / u.keV / u.s)

    assert_quantity_allclose(
        expected_abcd_rate_kev_cm0, meta_pixels["abcd_rate_kev_cm"][0, :], atol=1e-7 * expected_abcd_rate_kev_cm0.unit
    )

    assert_quantity_allclose(
        expected_abcd_rate_kev_cm1, meta_pixels["abcd_rate_kev_cm"][-1, :], atol=1e-7 * expected_abcd_rate_kev_cm1.unit
    )


@pytest.mark.remote_data
def test_create_meta_pixels_timebins(flare_cpd):
    energy_range = [6, 12] * u.keV

    # check time_range within one bin (i.e. timerange passed is within one bin)
    time_range = [flare_cpd.times[0], flare_cpd.times[0] + flare_cpd.durations / 4]
    meta_pixels = create_meta_pixels(
        flare_cpd,
        time_range=time_range,
        energy_range=energy_range,
        flare_location=SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=STIXImaging),
        no_shadowing=True,
    )

    assert_quantity_allclose(meta_pixels["time_range"].dt.to(u.s), flare_cpd.durations[0].to(u.s))

    # check time_range fully outside bins (i.e. all bins contained with timerange)
    time_range = [flare_cpd.times[0] - 10 * u.s, flare_cpd.times[-1] + 10 * u.s]
    meta_pixels = create_meta_pixels(
        flare_cpd,
        time_range=time_range,
        energy_range=energy_range,
        flare_location=SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=STIXImaging),
        no_shadowing=True,
    )

    assert_quantity_allclose(np.sum(flare_cpd.durations), meta_pixels["time_range"].dt.to(u.s))

    # check time_range start and end are within bins
    time_range = [flare_cpd.times[0], flare_cpd.times[2]]
    meta_pixels = create_meta_pixels(
        flare_cpd,
        time_range=time_range,
        energy_range=energy_range,
        flare_location=SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=STIXImaging),
        no_shadowing=True,
    )

    assert_quantity_allclose(np.sum(flare_cpd.durations[0:3]), meta_pixels["time_range"].dt.to(u.s))


def test_create_meta_pixels_shadow(flare_cpd):
    energy_range = [6, 12] * u.keV
    time_range = [flare_cpd.times[0], flare_cpd.times[2]]
    create_meta_pixels(
        flare_cpd,
        time_range=time_range,
        energy_range=energy_range,
        flare_location=SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=STIXImaging),
        no_shadowing=False,
    )


@pytest.mark.parametrize(
    "pix_set, real_comp",
    [
        ("blahblah", None),
        ("top+bot", np.cos(46.1 * u.deg)),
        ("all", np.cos(45 * u.deg)),
        ("small", np.cos(22.5 * u.deg)),
    ],
)
def test_create_visibility(pix_set, real_comp):
    # counts chosen to make real component 0 so real comp will only be phase from pixel combination
    fake_meta_pixels = {
        "abcd_rate_kev_cm": np.repeat([[0, 0, 1, 0]], 32, axis=0) * u.ct,
        "abcd_rate_error_kev_cm": np.repeat([[0, 0, 0, 0]], 32, axis=0) * u.ct,
        "energy_range": [4, 10] * u.keV,
        "time_range": TimeRange("2025-01-30", "2025-01-31"),
        "pixels": pix_set,
    }
    if pix_set == "blahblah":
        with pytest.raises(ValueError):
            create_visibility(fake_meta_pixels)
    else:
        vis = create_visibility(fake_meta_pixels)
        assert_equal(np.real(vis[0].visibilities.value), real_comp)


@pytest.mark.remote_data
def test_calibrate_visibility(flare_cpd):
    energy_range = [6, 12] * u.keV
    time_range = [flare_cpd.times[0], flare_cpd.times[2]]
    meta_pixels = create_meta_pixels(
        flare_cpd,
        time_range=time_range,
        energy_range=energy_range,
        no_shadowing=True,
    )
    vis = create_visibility(meta_pixels)
    time_centre = time_range[0] + np.diff(time_range) / 2
    obs = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime=time_centre)
    coord = SkyCoord(0 * u.deg, 0 * u.deg, frame=Helioprojective(obstime=time_centre, observer=obs))
    cal_vis = calibrate_visibility(vis, flare_location=coord)
    assert_quantity_allclose(cal_vis.meta["offset"].data.xyz, coord.transform_to(STIXImaging).data.xyz)
