import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from stixpy.data import test
from stixpy.product import Product
from stixpy.product.sources.science import (
    CompressedPixelData,
    RawPixelData,
    Spectrogram,
    SummedCompressedPixelData,
    Visibility,
)


@pytest.fixture
def rpd():
    return Product(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-rpd_20200505T235959-20200506T000019_V02_0087031808-50882.fits"
    )


@pytest.fixture
def cpd():
    return Product(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-cpd_20200505T235959-20200506T000019_V02_0087031809-50883.fits"
    )


@pytest.fixture
def scpd():
    return Product(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-scpd_20200505T235959-20200506T000019_V02_0087031810-50884.fits"
    )


@pytest.fixture
def spec():
    return Product(
        r"https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-spec_20200505T235959-20200506T000019_V02_0087031812-50886.fits"
    )


@pytest.mark.remote_data
def test_sciencedata_get_data(cpd):
    tot = cpd.data["counts"]
    norm = cpd.data["timedel"].reshape(5, 1, 1, 1) * cpd.dE
    rate = tot / norm
    error = np.sqrt(tot * u.ct + cpd.data["counts_comp_err"] ** 2) / norm
    r, re, t, dt, e = cpd.get_data()
    assert_allclose(rate, r)
    assert_allclose(error, re)

    # Detector sum
    tot = cpd.data["counts"][:, 0:32, ...].sum(axis=1, keepdims=True)
    norm = cpd.data["timedel"].reshape(5, 1, 1, 1) * cpd.dE
    rate = tot / norm
    error = np.sqrt(tot * u.ct + cpd.data["counts_comp_err"][:, 0:32, ...].sum(axis=1, keepdims=True) ** 2) / norm
    r, re, t, dt, e = cpd.get_data(detector_indices=[[0, 31]])
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Pixel sum
    tot = cpd.data["counts"][..., 0:12, :].sum(axis=2, keepdims=True)
    norm = cpd.data["timedel"].reshape(5, 1, 1, 1) * cpd.dE
    rate = tot / norm
    error = np.sqrt(tot * u.ct + cpd.data["counts_comp_err"][..., 0:12, :].sum(axis=2, keepdims=True) ** 2) / norm
    r, re, t, dt, e = cpd.get_data(pixel_indices=[[0, 11]])
    assert_allclose(rate, r)
    assert_allclose(error, re)

    # Detector and Pixel sum
    tot = cpd.data["counts"][:, 0:32, 0:12, :].sum(axis=(1, 2), keepdims=True)
    norm = cpd.data["timedel"].reshape(5, 1, 1, 1) * cpd.dE
    rate = tot / norm
    error = (
        np.sqrt(tot * u.ct + cpd.data["counts_comp_err"][:, 0:32, 0:12, :].sum(axis=(1, 2), keepdims=True) ** 2) / norm
    )
    r, re, t, dt, e = cpd.get_data(pixel_indices=[[0, 11]], detector_indices=[[0, 31]])
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Energy sum
    tot = cpd.data["counts"][..., 1:31].sum(axis=3, keepdims=True)
    norm = cpd.data["timedel"].reshape(5, 1, 1, 1) * (cpd.energies[30]["e_high"] - cpd.energies[1]["e_low"])
    rate = tot / norm
    error = np.sqrt(tot * u.ct + cpd.data["counts_comp_err"][..., 1:31].sum(axis=3, keepdims=True) ** 2) / norm
    r, re, t, dt, e = cpd.get_data(energy_indices=[[1, 30]])
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Time sum
    tot = cpd.data["counts"][:, ...].sum(axis=0, keepdims=True)
    norm = cpd.data["timedel"].sum() * cpd.dE
    rate = tot / norm
    error = np.sqrt(tot * u.ct + cpd.data["counts_comp_err"][:, ...].sum(axis=0, keepdims=True) ** 2) / norm
    r, re, t, dt, e = cpd.get_data(time_indices=[[0, 4]])
    assert_allclose(rate, r)
    assert_allclose(error, re)

    # Sum everything down to one number
    tot = cpd.data["counts"][..., 1:31].sum(keepdims=True)
    norm = cpd.data["timedel"].sum() * (cpd.energies[30]["e_high"] - cpd.energies[1]["e_low"])
    rate = tot / norm
    error = np.sqrt(tot * u.ct + cpd.data["counts_comp_err"][..., 1:31].sum(keepdims=True) ** 2) / norm
    r, re, t, dt, e = cpd.get_data(
        time_indices=[[0, 4]], energy_indices=[[1, 30]], pixel_indices=[[0, 11]], detector_indices=[[0, 31]]
    )
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Overwrite data with known values
    # 5 seconds, with dE 1/30keV across 30 channels
    cpd.data.remove_column("counts")  # need to remove and add to avoid error regarding casting
    cpd.data["counts"] = np.full((5, 32, 12, 32), 1 / (5 * 30)) * u.ct
    cpd.data.remove_column("counts_comp_err")
    cpd.data["counts_comp_err"] = np.full_like(cpd.data["counts"], np.sqrt(1 / (5 * 30)) * u.ct)
    cpd.data["timedel"] = 1 / 5 * u.s
    cpd.dE[:] = ((1 / 30) * u.keV).astype(np.float32)
    cpd.energies["e_high"][1:-1] = ((np.arange(31) / 30)[1:] * u.keV).astype(np.float32)
    cpd.energies["e_low"][1:-1] = ((np.arange(31) / 30)[:-1] * u.keV).astype(np.float32)
    cpd.data["time"] = cpd.time_range.start + np.arange(5) / 5 * u.s
    count, count_err, times, timedel, energies = cpd.get_data(time_indices=[[0, 4]], energy_indices=[[1, 30]])
    assert_allclose(count, 1 * u.ct / (u.s * u.keV))
    assert_allclose(count_err, np.sqrt(2) * u.ct / (u.s * u.keV))  # 1 + 1


# Raw Pixel Data (rpd) or l0
@pytest.mark.remote_data
def test_science_rpd(rpd):
    type_, num_times, num_detectors, num_pixels, num_energies = RawPixelData, 5, 32, 12, 33
    assert isinstance(rpd, type_)
    assert len(rpd.times) == num_times
    assert np.array_equal(rpd.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(rpd.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(rpd.energy_masks.masks, np.ones((1, num_energies)))


# Compresses Pixel data (cpd) or cpd
@pytest.mark.remote_data
def test_science_l1(cpd):
    type_, num_times, num_detectors, num_pixels, num_energies = CompressedPixelData, 5, 32, 12, 33
    assert isinstance(cpd, type_)
    assert len(cpd.times) == num_times
    assert np.array_equal(cpd.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(cpd.pixels.masks, np.ones(12)[None, ...])
    assert np.array_equal(cpd.energy_masks.masks, np.ones((1, num_energies)))
    assert cpd.data["counts"].shape == (num_times, num_detectors, num_pixels, num_energies - 1)


# Summed Compresses Pixel data (scpd) or l2
@pytest.mark.remote_data
def test_science_l2(scpd):
    type_, num_times, num_detectors, num_pixels, num_energies = SummedCompressedPixelData, 5, 32, 4, 33
    assert isinstance(scpd, type_)
    assert len(scpd.times) == num_times
    assert np.array_equal(scpd.detectors.masks, np.ones((1, num_detectors)))
    # assert np.array_equal(scpd.pixels.masks.astype(int), np.ones((1, num_pixels)))
    assert np.array_equal(scpd.energy_masks.masks, np.ones((1, num_energies)))
    assert scpd.data["counts"].shape == (num_times, num_detectors, num_pixels, num_energies - 1)


# Visibility (vis) or l3
@pytest.mark.skip(reason="Don't really use this data type")
def test_science_l3():
    type_, num_times, num_detectors, num_pixels, num_energies = Visibility, 5, 32, 12, 32  # noqa: F841
    res = Product(test.STIX_SCI_XRAY_VIS)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    # assert np.array_equal(scpd.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


# Spectrogram (spec) or l4
@pytest.mark.remote_data
def test_spectrogram(spec):
    type_, num_times, num_detectors, num_pixels, num_energies = Spectrogram, 5, 32, 12, 33
    assert isinstance(spec, type_)
    assert len(spec.times) == num_times
    assert np.array_equal(spec.detectors.masks, np.ones((1, num_detectors)))
    # assert np.array_equal(scpd.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(spec.energy_masks.masks, np.ones((1, num_energies)))
    assert np.array_equal(spec.pixel_masks.masks, np.zeros((1, num_pixels)))
    assert spec.data["counts"].shape == (num_times, num_energies - 1)


@pytest.mark.remote_data
def test_cpd_plot_pixel(cpd):
    cpd.plot_pixels()
    cpd.plot_pixels(kind="errorbar")
    cpd.plot_pixels(kind="config")


@pytest.mark.remote_data
def test_cpd_plot_timeseries(cpd):
    cpd.plot_timeseries()


@pytest.mark.remote_data
def test_cpd_plot_spectrogram(cpd):
    cpd.plot_spectrogram()


@pytest.mark.remote_data
def test_spec_plot_timeseries(spec):
    spec.plot_timeseries()


@pytest.mark.remote_data
def test_spec_plot_spectrogram(spec):
    spec.plot_spectrogram()
