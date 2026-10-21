import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.table import QTable

from stixpy.data import test
from stixpy.product import Product
from stixpy.product.sources.science import (
    CompressedPixelData,
    RawPixelData,
    SelectedData,
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
    # Without corrections get_data returns the stored counts, less the 0 keV bottom and open top energy bins
    uncorrected = {"vtype": "dcr", "livetime_correction": False, "elut_correction": False}
    counts = cpd.data["counts"][..., 1:31]
    var = counts * u.ct + cpd.data["counts_comp_err"][..., 1:31] ** 2
    timedel = cpd.data["timedel"].reshape(5, 1, 1, 1).to(u.s)
    de = cpd.dE[1:31]

    data = cpd.get_data(**uncorrected)
    assert_allclose(data.counts, counts / (timedel * de))
    assert_allclose(data.counts_err, np.sqrt(var) / (timedel * de), rtol=1e-6)

    # Detector sum
    data = cpd.get_data(detector_indices=[[0, 31]], **uncorrected)
    assert_allclose(data.counts, counts.sum(axis=1, keepdims=True) / (timedel * de))
    assert_allclose(data.counts_err, np.sqrt(var.sum(axis=1, keepdims=True)) / (timedel * de), rtol=1e-6)

    # Pixel sum
    data = cpd.get_data(pixel_indices=[[0, 11]], **uncorrected)
    assert_allclose(data.counts, counts.sum(axis=2, keepdims=True) / (timedel * de))
    assert_allclose(data.counts_err, np.sqrt(var.sum(axis=2, keepdims=True)) / (timedel * de), rtol=1e-6)

    # Detector and Pixel sum
    data = cpd.get_data(pixel_indices=[[0, 11]], detector_indices=[[0, 31]], **uncorrected)
    assert_allclose(data.counts, counts.sum(axis=(1, 2), keepdims=True) / (timedel * de))
    assert_allclose(data.counts_err, np.sqrt(var.sum(axis=(1, 2), keepdims=True)) / (timedel * de), rtol=1e-6)

    # Energy sum: rows 1 to 30 of the energy table are every bin get_data returns
    de_sum = cpd.energies[30]["e_high"] - cpd.energies[1]["e_low"]
    data = cpd.get_data(energy_indices=[[1, 30]], **uncorrected)
    assert_allclose(data.counts, counts.sum(axis=3, keepdims=True) / (timedel * de_sum))
    assert_allclose(data.counts_err, np.sqrt(var.sum(axis=3, keepdims=True)) / (timedel * de_sum), rtol=1e-6)
    assert_allclose(data.e_norm.reshape(-1), de_sum)

    # Time sum
    data = cpd.get_data(time_indices=[[0, 4]], **uncorrected)
    assert_allclose(data.counts, counts.sum(axis=0, keepdims=True) / (timedel.sum() * de))
    assert_allclose(data.counts_err, np.sqrt(var.sum(axis=0, keepdims=True)) / (timedel.sum() * de), rtol=1e-6)
    assert_allclose(data.t_norm.reshape(-1), timedel.sum())

    # Sum everything down to one number
    data = cpd.get_data(
        time_indices=[[0, 4]],
        energy_indices=[[1, 30]],
        pixel_indices=[[0, 11]],
        detector_indices=[[0, 31]],
        **uncorrected,
    )
    assert_allclose(data.counts, counts.sum(keepdims=True) / (timedel.sum() * de_sum))
    assert_allclose(data.counts_err, np.sqrt(var.sum(keepdims=True)) / (timedel.sum() * de_sum), rtol=1e-6)

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
    data = cpd.get_data(time_indices=[[0, 4]], energy_indices=[[1, 30]], **uncorrected)
    assert_allclose(data.counts, 1 * u.ct / (u.s * u.keV))
    assert_allclose(data.counts_err, np.sqrt(2) * u.ct / (u.s * u.keV))  # 1 + 1


@pytest.mark.remote_data
def test_cpd_get_data_pixel_indices(cpd):
    counts, *_ = cpd.get_data(pixel_indices=[8, 9], livetime_correction=False)
    assert counts.shape[2] == 2
    # the livetime correction sums the selected pixels
    counts, *_ = cpd.get_data(pixel_indices=[8, 9])
    assert counts.shape[2] == 1


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
@pytest.mark.parametrize("vtype, unit", [("c", "ct"), ("cr", "ct / s"), ("dcr", "ct / (s keV)")])
def test_get_data_vtype(cpd, vtype, unit):
    data, *_ = cpd.get_data(vtype=vtype)
    assert data.unit == u.Unit(unit)


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


# synthetic products have no triggers or ELUT channels, so get_data is called without the corrections
UNCORRECTED = {"livetime_correction": False, "elut_correction": False}


def _make_product(cls, counts, detector_masks, pixel_masks, err_name="counts_comp_err"):
    """Build a synthetic science product from a counts array and per-time masks."""
    nt, ne = counts.shape[0], counts.shape[-1]
    data = QTable()
    data["time"] = np.arange(nt) * u.s
    data["timedel"] = np.ones(nt) * u.s
    data["pixel_masks"] = pixel_masks
    data["rcr"] = np.zeros(nt, dtype=int)
    data["counts"] = counts * u.ct
    data[err_name] = np.sqrt(counts) * u.ct
    control = QTable()
    control["energy_bin_edge_mask"] = np.ones((1, ne + 1), dtype=int)
    if cls is Spectrogram:
        control["detector_masks"] = detector_masks[:1]
    else:
        data["detector_masks"] = detector_masks
    energies = QTable()
    # start above 0 keV, as get_data drops a 0 keV bottom bin
    energies["e_low"] = (np.arange(ne) + 1) * u.keV
    energies["e_high"] = (np.arange(ne) + 2) * u.keV
    return cls(
        meta={"date-obs": "2024-01-01T00:00:00"}, control=control, data=data, energies=energies, idb_versions=None
    )


@pytest.fixture
def cpd_partial():
    """Compressed pixel data with detectors 9, 10 and pixels 3, 7 switched off, so only 30 x 10 are stored."""
    nt, ne = 3, 4
    dmask = np.ones(32, dtype=int)
    dmask[[9, 10]] = 0
    pmask = np.ones(12, dtype=int)
    pmask[[3, 7]] = 0
    counts = np.arange(nt * 30 * 10 * ne, dtype=float).reshape(nt, 30, 10, ne) + 1
    return _make_product(CompressedPixelData, counts, np.tile(dmask, (nt, 1)), np.tile(pmask, (nt, 1)))


def test_get_data_full_layout(cpd_partial):
    stored = cpd_partial.data["counts"]
    stored_err = cpd_partial.data["counts_comp_err"]
    dets = np.flatnonzero(cpd_partial.detector_masks.masks[0])
    pixs = np.flatnonzero(cpd_partial.pixel_masks.masks[0])

    counts, errors, *_ = cpd_partial.get_data(vtype="c", **UNCORRECTED)
    assert counts.shape == (3, 32, 12, 4)

    # Stored position (i, j) lands at real detector dets[i] and pixel pixs[j]
    assert_allclose(counts[:, dets[:, None], pixs, :], stored)
    assert_allclose(counts[:, 11, 8, :], stored[:, 9, 6, :])
    assert_allclose(errors[:, dets[:, None], pixs, :], np.sqrt(stored * u.ct + stored_err**2))

    # Detectors and pixels that are off are zero
    assert np.all(counts[:, [9, 10]] == 0)
    assert np.all(counts[:, :, [3, 7]] == 0)
    assert np.all(errors[:, [9, 10]] == 0)
    assert np.all(errors[:, :, [3, 7]] == 0)

    # Indices refer to real detectors and pixels
    sel, *_ = cpd_partial.get_data(vtype="c", detector_indices=[11], pixel_indices=[8], **UNCORRECTED)
    assert_allclose(sel[:, 0, 0, :], stored[:, 9, 6, :])
    total, *_ = cpd_partial.get_data(vtype="c", detector_indices=[[0, 31]], pixel_indices=[[0, 11]], **UNCORRECTED)
    assert_allclose(total[:, 0, 0, :], stored.sum(axis=(1, 2)))


def test_get_data_full_layout_comp_comp_err():
    nt = 2
    dmask = np.ones(32, dtype=int)
    dmask[0] = 0
    counts = np.ones((nt, 31, 12, 3))
    prod = _make_product(
        CompressedPixelData,
        counts,
        np.tile(dmask, (nt, 1)),
        np.ones((nt, 12), dtype=int),
        err_name="counts_comp_comp_err",
    )
    counts, errors, *_ = prod.get_data(vtype="c", **UNCORRECTED)
    assert counts.shape == errors.shape == (nt, 32, 12, 3)
    assert np.all(errors[:, 0] == 0)
    assert np.all(errors[:, 1:] > 0)


def test_get_data_full_layout_changing_detector_mask():
    nt = 2
    dmasks = np.ones((nt, 32), dtype=int)
    dmasks[1, 5] = 0
    prod = _make_product(CompressedPixelData, np.ones((nt, 31, 12, 3)), dmasks, np.ones((nt, 12), dtype=int))
    with pytest.raises(ValueError, match="detector mask changes"):
        prod.get_data(**UNCORRECTED)


def test_get_data_full_layout_shape_mismatch():
    nt = 2
    dmask = np.ones(32, dtype=int)
    dmask[0] = 0
    prod = _make_product(
        CompressedPixelData, np.ones((nt, 30, 12, 3)), np.tile(dmask, (nt, 1)), np.ones((nt, 12), dtype=int)
    )
    with pytest.raises(ValueError, match="does not match the masks"):
        prod.get_data(**UNCORRECTED)


def test_get_data_full_layout_summed_unchanged():
    nt = 2
    pmask = np.zeros((4, 12), dtype=int)
    for i in range(4):
        pmask[i, i::4] = 1
    counts = np.arange(nt * 32 * 4 * 3, dtype=float).reshape(nt, 32, 4, 3)
    prod = _make_product(SummedCompressedPixelData, counts, np.ones((nt, 32), dtype=int), np.tile(pmask, (nt, 1, 1)))
    out, *_ = prod.get_data(vtype="c", **UNCORRECTED)
    assert_allclose(out, counts * u.ct)


def test_get_data_full_layout_spectrogram_unchanged():
    nt = 2
    counts = np.arange(nt * 3, dtype=float).reshape(nt, 3)
    dmask = np.ones(32, dtype=int)
    dmask[[9, 10]] = 0
    prod = _make_product(Spectrogram, counts, dmask[None, :], np.zeros((nt, 12), dtype=int))
    out, *_ = prod.get_data(vtype="c", **UNCORRECTED)
    assert_allclose(out, counts.reshape(nt, 1, 1, 3) * u.ct)


def test_plot_pixels_full_layout(cpd_partial):
    plotter = cpd_partial.plot_pixels()
    assert plotter.counts.shape == (3, 32, 12, 4)
    assert np.isnan(plotter.counts[:, [9, 10]]).all()
    assert np.isnan(plotter.counts[:, :, [3, 7]]).all()
    assert_allclose(plotter.counts[0, 11, 8, :].value, cpd_partial.data["counts"][0, 9, 6, :].value)


def test_get_data_returns_selected_data(cpd_partial):
    data = cpd_partial.get_data(vtype="c", time_indices=[[0, 1], [2, 2]], **UNCORRECTED)
    assert isinstance(data, SelectedData)
    assert data.counts is data[0]
    assert data.counts.shape == (2, 32, 12, 4)
    assert data.livefrac is None
    assert data.elut_cor_fac is None
    # one RCR state per summed time range
    assert data.rcr.shape == (2,)


def test_plot_spectrogram_all_energies(cpd_partial):
    im = cpd_partial.plot_spectrogram()
    # every energy bin get_data returns is plotted
    assert im.get_array().shape == (4, 3)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"energy_indices": [0, 1]}, ValueError, "energy_indices can not be used"),
        ({"livetime_correction": False}, ValueError, "needs livetime_correction=True"),
        ({"flare_location": {"stx": None, "hpc": None}}, TypeError, "flare_location must be a SkyCoord"),
    ],
)
def test_get_data_sunkit_spex_errors(cpd_partial, kwargs, error, match):
    pytest.importorskip("sunkit_spex")
    with pytest.raises(error, match=match):
        cpd_partial.get_data(sunkit_spex_spectrum=True, **kwargs)


def test_get_data_sunkit_spex_not_installed(cpd_partial, monkeypatch):
    import sys

    # a None entry makes the import fail, as if sunkit-spex were not installed
    for name in [name for name in sys.modules if name.split(".")[0] == "sunkit_spex"] + ["sunkit_spex"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "stixpy.product.sources._spectrum", raising=False)
    with pytest.raises(ImportError, match=r"stixpy\[spectroscopy\]"):
        cpd_partial.get_data(sunkit_spex_spectrum=True)
