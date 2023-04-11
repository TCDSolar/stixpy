import astropy.units as u
import numpy as np

from stixpy.data import test
from stixpy.product.sources import CompressedPixelData, SummedCompressedPixelData, Spectrogram, RawPixelData
from stixpy.science import ScienceData


def test_sciencedata_get_data():
    l1 = ScienceData.from_fits(test.STIX_SCI_XRAY_CPD)
    tot = l1.data['counts']
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot*u.ct+l1.data['counts_comp_err']**2) / norm
    r, re, t, dt, e = l1.get_data()
    assert np.allclose(rate, r)
    assert np.allclose(error, re)

    # Detector sum
    tot = l1.data['counts'][:, 0:32, ...].sum(axis=1, keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot*u.ct+l1.data['counts_comp_err'][:, 0:32, ...].sum(axis=1, keepdims=True)**2)/norm
    r, re, t, dt, e = l1.get_data(detector_indices=[[0, 31]])
    assert np.allclose(rate, r)
    assert np.allclose(error, re, atol=1e-3)

    # Pixel sum
    tot = l1.data['counts'][..., 0:12, :].sum(axis=2, keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot * u.ct
                    + l1.data['counts_comp_err'][..., 0:12, :].sum(axis=2, keepdims=True)**2) / norm
    r, re, t, dt, e = l1.get_data(pixel_indices=[[0, 11]])
    assert np.allclose(rate, r)
    assert np.allclose(error, re)

    # Detector and Pixel sum
    tot = l1.data['counts'][:, 0:32, 0:12, :].sum(axis=(1, 2), keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1) * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot*u.ct + l1.data['counts_comp_err'][:, 0:32, 0:12, :].sum(axis=(1, 2),
                                                                           keepdims=True)**2) / norm
    r, re, t, dt, e = l1.get_data(pixel_indices=[[0, 11]], detector_indices=[[0, 31]])
    assert np.allclose(rate, r)
    assert np.allclose(error, re, atol=1e-3)

    # Energy sum
    tot = l1.data['counts'][..., 1:31].sum(axis=3, keepdims=True)
    norm = (l1.data['timedel'].reshape(5, 1, 1, 1)
            * (l1.energies[30]['e_high']-l1.energies[1]['e_low']))
    rate = tot / norm
    error = np.sqrt(tot*u.ct + l1.data['counts_comp_err'][..., 1:31].sum(axis=3, keepdims=True)**2)/norm
    r, re, t, dt, e = l1.get_data(energy_indices=[[1, 30]])
    assert np.allclose(rate, r)
    assert np.allclose(error, re, atol=1e-3)

    # Time sum
    tot = l1.data['counts'][:, ...].sum(axis=0, keepdims=True)
    norm = (l1.data['timedel'].sum() * l1.dE)
    rate = tot / norm
    error = np.sqrt(tot * u.ct + l1.data['counts_comp_err'][:, ...].sum(axis=0, keepdims=True) ** 2)/norm
    r, re, t, dt, e = l1.get_data(time_indices=[[0, 4]])
    assert np.allclose(rate, r)
    assert np.allclose(error, re)

    # Sum everything down to one number
    tot = l1.data['counts'][..., 1:31].sum(keepdims=True)
    norm = (l1.data['timedel'].sum() * (l1.energies[30]['e_high'] - l1.energies[1]['e_low']))
    rate = tot/norm
    error = np.sqrt(tot * u.ct + l1.data['counts_comp_err'][..., 1:31].sum(keepdims=True) ** 2) / norm
    r, re, t, dt, e = l1.get_data(time_indices=[[0, 4]], energy_indices=[[1, 30]],
                                  pixel_indices=[[0, 11]], detector_indices=[[0, 31]])
    assert np.allclose(rate, r)
    assert np.allclose(error, re, atol=1e-3)

    # Overwrite data with known values
    # 5 seconds, with dE 1/30keV across 30 channels
    l1.data.remove_column('counts') # need to remove and add to avoid error regarding casting
    l1.data['counts'] = np.full((5, 32, 12, 32), 146/(5*30))*u.ct
    l1.data.remove_column('counts_comp_err')
    l1.data['counts_comp_err'] = np.full_like(l1.data['counts'], np.sqrt(146*145/(5*30))*u.ct)
    l1.data['timedel'] = 1/5*u.s
    l1.dE[:] = ((1/30)*u.keV).astype(np.float32)
    l1.energies['e_high'][1:-1] = ((np.arange(31) / 30)[1:] * u.keV).astype(np.float32)
    l1.energies['e_low'][1:-1] = ((np.arange(31) / 30)[:-1] * u.keV).astype(np.float32)
    l1.data['time'] = l1.time_range.start + np.arange(5)/5*u.s
    count, count_err, times, timedel, energies = l1.get_data(time_indices=[[0, 4]],
                                                             energy_indices=[[1, 30]])
    assert np.allclose(count, 1 * u.ct / (u.s * u.keV))
    assert np.allclose(count_err, 1 * u.ct / (u.s * u.keV))


def test_science_l0():
    type_, num_times, num_detectors, num_pixels, num_energies = RawPixelData, 5, 32, 12, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_RPD)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


def test_science_l1():
    type_, num_times, num_detectors, num_pixels, num_energies = CompressedPixelData, 5, 32, 12, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_CPD)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(res.pixels.masks, np.ones(12)[None, ...])
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


def test_science_l2():
    type_, num_times, num_detectors, num_pixels, num_energies = SummedCompressedPixelData, 5, 32, 4, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_SCPD)
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


def test_spectrogram():
    type_, num_times, num_detectors, num_pixels, num_energies = Spectrogram, 5, 32, 12, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_SPEC)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    # assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))
