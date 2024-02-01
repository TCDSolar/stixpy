import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from stixpy.data import test
from stixpy.product.sources import CompressedPixelData, SummedCompressedPixelData, Spectrogram, RawPixelData
from stixpy.product import Product


@pytest.mark.remote_data
def test_sciencedata_get_data():
    l1 = Product("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-cpd_20200505T235959-20200506T000019_V02_0087031809-50883.fits")
    tot = l1.data['counts']
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot*u.ct+l1.data['counts_comp_err']**2) / norm
    r, re, t, dt, e = l1.get_data()
    assert_allclose(rate, r)
    assert_allclose(error, re)

    # Detector sum
    tot = l1.data['counts'][:, 0:32, ...].sum(axis=1, keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot*u.ct+l1.data['counts_comp_err'][:, 0:32, ...].sum(axis=1, keepdims=True)**2)/norm
    r, re, t, dt, e = l1.get_data(detector_indices=[[0, 31]])
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Pixel sum
    tot = l1.data['counts'][..., 0:12, :].sum(axis=2, keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot * u.ct
                    + l1.data['counts_comp_err'][..., 0:12, :].sum(axis=2, keepdims=True)**2) / norm
    r, re, t, dt, e = l1.get_data(pixel_indices=[[0, 11]])
    assert_allclose(rate, r)
    assert_allclose(error, re)

    # Detector and Pixel sum
    tot = l1.data['counts'][:, 0:32, 0:12, :].sum(axis=(1, 2), keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot*u.ct + l1.data['counts_comp_err'][:, 0:32, 0:12, :].sum(axis=(1, 2),
                                                                           keepdims=True)**2) / norm
    r, re, t, dt, e = l1.get_data(pixel_indices=[[0, 11]], detector_indices=[[0, 31]])
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Energy sum
    tot = l1.data['counts'][..., 1:31].sum(axis=3, keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1)
            * (l1.energies[30]['e_high']-l1.energies[1]['e_low']))
    rate = tot / norm
    error = np.sqrt(tot*u.ct + l1.data['counts_comp_err'][..., 1:31].sum(axis=3, keepdims=True)**2)/norm
    r, re, t, dt, e = l1.get_data(energy_indices=[[1, 30]])
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Time sum
    tot = l1.data['counts'][:, ...].sum(axis=0, keepdims=True)
    norm = (l1.data['timedel'].sum() * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot * u.ct + l1.data['counts_comp_err'][:, ...].sum(axis=0, keepdims=True) ** 2)/norm
    r, re, t, dt, e = l1.get_data(time_indices=[[0, 4]])
    assert_allclose(rate, r)
    assert_allclose(error, re)

    # Sum everything down to one number
    tot = l1.data['counts'][..., 1:31].sum(keepdims=True)
    norm = (l1.data['timedel'].sum() * (l1.energies[30]['e_high'] - l1.energies[1]['e_low']))
    rate = tot/norm
    error = np.sqrt(tot * u.ct + l1.data['counts_comp_err'][..., 1:31].sum(keepdims=True) ** 2) / norm
    r, re, t, dt, e = l1.get_data(time_indices=[[0, 4]], energy_indices=[[1, 30]],
                                  pixel_indices=[[0, 11]], detector_indices=[[0, 31]])
    assert_allclose(rate, r)
    assert_allclose(error, re, atol=1e-3)

    # Overwrite data with known values
    # 5 seconds, with dE 1/30keV across 30 channels
    l1.data.remove_column('counts') # need to remove and add to avoid error regarding casting
    l1.data['counts'] = np.full((5, 32, 12, 32), 1/(5*30))*u.ct
    l1.data.remove_column('counts_comp_err')
    l1.data['counts_comp_err'] = np.full_like(l1.data['counts'], np.sqrt(1/(5*30))*u.ct)
    l1.data['timedel'] = 1/5*u.s
    l1.dE[:] = ((1/30)*u.keV).astype(np.float32)
    l1.energies['e_high'][1:-1] = ((np.arange(31) / 30)[1:] * u.keV).astype(np.float32)
    l1.energies['e_low'][1:-1] = ((np.arange(31) / 30)[:-1] * u.keV).astype(np.float32)
    l1.data['time'] = l1.time_range.start + np.arange(5)/5*u.s
    count, count_err, times, timedel, energies = l1.get_data(time_indices=[[0, 4]],
                                                             energy_indices=[[1, 30]])
    assert_allclose(count, 1 * u.ct / (u.s * u.keV))
    assert_allclose(count_err, np.sqrt(2) * u.ct / (u.s * u.keV))  # 1 + 1


@pytest.mark.remote_data
def test_science_l0():
    type_, num_times, num_detectors, num_pixels, num_energies = RawPixelData, 5, 32, 12, 33
    res = Product("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-rpd_20200505T235959-20200506T000019_V02_0087031808-50882.fits")
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


@pytest.mark.remote_data
def test_science_l1():
    type_, num_times, num_detectors, num_pixels, num_energies = CompressedPixelData, 5, 32, 12, 33
    res = Product("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-cpd_20200505T235959-20200506T000019_V02_0087031809-50883.fits")
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(res.pixels.masks, np.ones(12)[None, ...])
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


@pytest.mark.remote_data
def test_science_l2():
    type_, num_times, num_detectors, num_pixels, num_energies = SummedCompressedPixelData, 5, 32, 4, 33
    res = Product("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-scpd_20200505T235959-20200506T000019_V02_0087031810-50884.fits")
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    # assert np.array_equal(res.pixels.masks.astype(int), np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


# def test_science_l3():
#     type_, num_times, num_detectors, num_pixels, num_energies = Visibility, 5, 32, 12, 32
#     res = ScienceData.from_fits(test.STIX_SCI_XRAY_VIS)
#     assert isinstance(res, type_)
#     assert len(res.times) == num_times
#     assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
#     # assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
#     assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))

@pytest.mark.remote_data
def test_spectrogram():
    type_, num_times, num_detectors, num_pixels, num_energies = Spectrogram, 5, 32, 12, 33
    res = Product(r"https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/05/SCI/solo_L1_stix-sci-xray-spec_20200505T235959-20200506T000019_V02_0087031812-50886.fits")
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    # assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))
