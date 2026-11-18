import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.table import QTable

from stixpy.product.sources.science import (
    CompressedPixelData,
    Spectrogram,
    SummedCompressedPixelData,
    subtract_background,
)

# synthetic products have no ELUT channels, so get_data is called without the ELUT correction
NO_ELUT = {"elut_correction": False}


def _make_product(cls, counts, *, timedel=1, triggers=1000.0, detector_off=(), pixel_off=(), energies=None):
    """
    Build a synthetic science product with triggers, so the livetime correction can be applied.

    Pixel data `counts` are (time, detector, pixel, energy) for the detectors and pixels that are on, spectrogram
    `counts` are (time, energy).
    """
    nt, ne = counts.shape[0], counts.shape[-1]
    dmask = np.ones(32, dtype=int)
    dmask[list(detector_off)] = 0
    pmask = np.ones(12, dtype=int)
    pmask[list(pixel_off)] = 0

    data = QTable()
    data["time"] = np.arange(nt) * timedel * u.s
    data["timedel"] = np.full(nt, timedel, dtype=float) * u.s
    data["pixel_masks"] = np.tile(pmask, (nt, 1))
    data["rcr"] = np.zeros(nt, dtype=int)
    data["counts"] = counts * u.ct
    data["counts_comp_err"] = np.zeros(counts.shape) * u.ct
    control = QTable()
    control["energy_bin_edge_mask"] = np.ones((1, ne + 1), dtype=int)
    if cls is Spectrogram:
        data["triggers"] = np.full(nt, 16 * triggers * timedel)
        data["triggers_comp_err"] = np.zeros(nt)
        control["detector_masks"] = dmask[None, :]
    else:
        data["triggers"] = np.full((nt, 16), triggers * timedel)
        data["triggers_comp_err"] = np.zeros((nt, 16))
        data["detector_masks"] = np.tile(dmask, (nt, 1))
    if energies is None:
        energies = QTable()
        energies["e_low"] = (np.arange(ne) + 4) * u.keV
        energies["e_high"] = (np.arange(ne) + 5) * u.keV
    return cls(
        meta={"date-obs": "2024-01-01T00:00:00"}, control=control, data=data, energies=energies, idb_versions=None
    )


@pytest.fixture
def cpd():
    """Pixel data with 3 time bins, detectors 10, 11 and pixel 7 off."""
    rng = np.random.default_rng(0)
    counts = rng.integers(50, 100, size=(3, 30, 11, 4)).astype(float)
    return _make_product(CompressedPixelData, counts, detector_off=(10, 11), pixel_off=(7,))


@pytest.fixture
def bkg():
    """A background with one 100 s time bin of 2 counts per detector, pixel and energy bin."""
    return _make_product(CompressedPixelData, np.full((1, 32, 12, 4), 2.0), timedel=100)


@pytest.fixture
def bkg_zero():
    return _make_product(CompressedPixelData, np.zeros((1, 32, 12, 4)), timedel=100)


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"detector_indices": [0, 3, 5]},
        {"detector_indices": [[0, 15], [16, 31]], "pixel_indices": [[0, 3], [4, 11]]},
        {"detector_indices": [[0, 31]], "time_indices": [[0, 2]]},
        {"pixel_indices": [1, 2]},
        {"detector_indices": [0, 3, 5], "sum_detectors": True},
    ],
)
@pytest.mark.parametrize("vtype", ["c", "cr", "dcr"])
def test_zero_background_unchanged(cpd, bkg_zero, selection, vtype):
    without = cpd.get_data(vtype=vtype, **selection, **NO_ELUT)
    with_bkg = subtract_background(without, bkg_zero)

    assert with_bkg.counts.shape == without.counts.shape
    assert with_bkg.counts.unit == without.counts.unit
    assert_allclose(with_bkg.counts, without.counts)
    assert_allclose(with_bkg.counts_err, without.counts_err)
    assert with_bkg.selection is without.selection


def test_background_scaled_and_subtracted(cpd, bkg):
    without = cpd.get_data(vtype="c", detector_indices=[0, 1], pixel_indices=[0], **NO_ELUT)
    with_bkg = subtract_background(without, bkg)

    # both livetime fractions are the same, so the subtracted background is 2 ct / 100 s for each 1 s science bin
    assert_allclose(with_bkg.counts, without.counts - 0.02 * u.ct)
    assert np.all(with_bkg.counts_err > without.counts_err)


@pytest.mark.parametrize("vtype", ["c", "cr", "dcr"])
def test_background_only_science_detectors_and_pixels(cpd, bkg, vtype):
    # the background has every detector and pixel on, the science data not detectors 10 and 11 or pixel 7: only the
    # 11 pixels of the detectors that are on in the science data are subtracted
    without = cpd.get_data(vtype=vtype, **NO_ELUT)
    with_bkg = subtract_background(without, bkg)
    # a livetime corrected rate; the background has the same trigger rate, so the same livetime fraction
    rate = np.where(np.isin(np.arange(32), [10, 11]), 0, 11 * 0.02)[None, :, None, None] * u.ct / u.s
    rate = rate / without.livefrac
    scale = {"c": without.t_norm * without.livefrac, "cr": 1, "dcr": 1 / without.e_norm}[vtype]
    assert_allclose(with_bkg.counts, without.counts - scale * rate)


def test_background_spectrogram(bkg_zero, bkg):
    rng = np.random.default_rng(1)
    spec = _make_product(Spectrogram, rng.integers(500, 1000, size=(3, 4)).astype(float), detector_off=(9,))

    without = spec.get_data(vtype="cr", **NO_ELUT)
    zero = subtract_background(without, bkg_zero)
    assert zero.counts.shape == without.counts.shape == (3, 1, 1, 4)
    assert_allclose(zero.counts, without.counts)
    assert_allclose(zero.counts_err, without.counts_err)

    # the background of the 31 detectors and 12 pixels summed on board into the spectrogram
    counts = spec.get_data(vtype="c", **NO_ELUT)
    assert_allclose(subtract_background(counts, bkg).counts, counts.counts - 31 * 12 * 0.02 * u.ct)


def test_background_negative_kept(cpd):
    big = _make_product(CompressedPixelData, np.full((1, 32, 12, 4), 1e6), timedel=100)
    data = subtract_background(cpd.get_data(vtype="c", detector_indices=[[0, 31]], **NO_ELUT), big)
    assert np.all(data.counts < 0)
    assert np.all(data.counts_err > 0)


def test_background_livetime_false(cpd, bkg):
    with pytest.raises(ValueError, match="needs the livetime correction"):
        subtract_background(cpd.get_data(livetime_correction=False, **NO_ELUT), bkg)


def test_background_not_a_product(cpd):
    with pytest.raises(TypeError, match="bkg must be a STIX science product"):
        subtract_background(cpd.get_data(**NO_ELUT), np.zeros(3))


def test_background_spectrogram_rejected(cpd):
    spec = _make_product(Spectrogram, np.ones((1, 4)), timedel=100)
    with pytest.raises(ValueError, match="must be pixel data"):
        subtract_background(cpd.get_data(**NO_ELUT), spec)


def test_background_summed_pixels_rejected(cpd, bkg):
    pmask = np.zeros((4, 12), dtype=int)
    for i in range(4):
        pmask[i, i::4] = 1
    scpd = _make_product(SummedCompressedPixelData, np.ones((2, 32, 4, 4)))
    scpd.data["pixel_masks"] = np.tile(pmask, (2, 1, 1))
    scpd.pixel_masks = type(cpd.pixel_masks)(scpd.data["pixel_masks"])
    with pytest.raises(ValueError, match="summed pixel data"):
        subtract_background(scpd.get_data(livetime_correction=False, **NO_ELUT), bkg)


def test_background_several_time_bins(cpd):
    two = _make_product(CompressedPixelData, np.ones((2, 32, 12, 4)), timedel=100)
    with pytest.raises(ValueError, match="single time bin, but it has 2"):
        subtract_background(cpd.get_data(**NO_ELUT), two)


def test_background_energy_mismatch(cpd):
    energies = QTable()
    energies["e_low"] = [4, 5, 6, 8] * u.keV
    energies["e_high"] = [5, 6, 8, 10] * u.keV
    other = _make_product(CompressedPixelData, np.ones((1, 32, 12, 4)), timedel=100, energies=energies)
    with pytest.raises(ValueError, match=r"science energy bins \['6-7', '7-8'\] keV are not in the background"):
        subtract_background(cpd.get_data(**NO_ELUT), other)
    # a science range that starts and ends on background edges can be subtracted
    data = subtract_background(cpd.get_data(vtype="c", energy_indices=[[0, 1], [2, 3]], **NO_ELUT), other)
    assert data.counts.shape[-1] == 2


def test_background_elut_mismatch(cpd, bkg, monkeypatch):
    from dataclasses import replace

    data = cpd.get_data(**NO_ELUT)
    data = data._replace(selection=replace(data.selection, elut_file="elut_table_20200519.csv"))
    with pytest.raises(ValueError, match="recorded with the ELUT"):
        subtract_background(data, bkg)


def test_background_detector_off(cpd):
    other = _make_product(CompressedPixelData, np.ones((1, 31, 11, 4)), timedel=100, detector_off=(3,), pixel_off=(2,))
    with pytest.raises(ValueError, match=r"Detectors \[3\] and pixels \[2\] are selected but are off"):
        subtract_background(cpd.get_data(**NO_ELUT), other)
    # detectors and pixels that are not selected need not be in the background
    subtract_background(cpd.get_data(detector_indices=[0, 1], pixel_indices=[0, 1], **NO_ELUT), other)


def test_background_detector_off_spectrogram():
    spec = _make_product(Spectrogram, np.ones((2, 4)))
    other = _make_product(CompressedPixelData, np.ones((1, 31, 12, 4)), timedel=100, detector_off=(5,))
    with pytest.raises(ValueError, match=r"Detectors \[5\] and pixels \[\] were summed on board"):
        subtract_background(spec.get_data(**NO_ELUT), other)


def test_selected_detector_ids_follow_count_order():
    counts = np.ones((1, 32, 12, 4))
    counts[:, 2] *= 3
    product = _make_product(CompressedPixelData, counts, triggers=0)
    selected = product.get_data(vtype="c", detector_indices=[2, 0], pixel_indices=[0], **NO_ELUT)
    np.testing.assert_array_equal(np.nonzero(selected.selection.detectors)[1], [2, 0])
    assert_allclose(selected.counts[0, :, 0, 0].to_value(u.ct), [3, 1])


def test_correlated_matches_legacy_without_combining(cpd, bkg):
    # with nothing combined over detectors or time, the parts of the uncertainty add up the same way
    kwargs = {"vtype": "c", "detector_indices": [0], **NO_ELUT}
    legacy = subtract_background(cpd.get_data(**kwargs), bkg)
    correlated = subtract_background(cpd.get_data(error_model="correlated", **kwargs), bkg)
    assert_allclose(correlated.counts, legacy.counts)
    assert_allclose(correlated.counts_err, legacy.counts_err)


@pytest.mark.parametrize("error_model", ["legacy", "correlated"])
def test_background_error_linear_over_time(bkg, error_model):
    # no science counts, so the background is the whole uncertainty: one measurement scaled to all three time bins
    empty = _make_product(CompressedPixelData, np.zeros((3, 32, 12, 4)))
    kwargs = {"vtype": "c", "detector_indices": [0], "pixel_indices": [0], "error_model": error_model, **NO_ELUT}
    one = subtract_background(empty.get_data(time_indices=[0], **kwargs), bkg)
    three = subtract_background(empty.get_data(time_indices=[[0, 2]], **kwargs), bkg)
    assert_allclose(three.counts_err, 3 * one.counts_err)
    assert_allclose(three.counts, -3 * 0.02 * u.ct)


def test_error_model_unknown(cpd):
    with pytest.raises(ValueError, match="error_model must be one of"):
        cpd.get_data(error_model="independent", **NO_ELUT)


@pytest.mark.parametrize("ssw_compatible", [False, True])
def test_spectrogram_background_with_unequal_live_exposures(ssw_compatible):
    science = _make_product(Spectrogram, np.array([[10.0], [1000.0]]), detector_off=(8, 9))
    science.data["triggers"][:] = [16_000, 160_000]
    background = _make_product(CompressedPixelData, np.ones((1, 32, 12, 1)), triggers=0)
    options = dict(vtype="c", ssw_compatible=ssw_compatible, **NO_ELUT)

    per_bin = science.get_data(**options)
    subtracted = subtract_background(per_bin, background)
    exposure = per_bin.t_norm * per_bin.livefrac
    expected_rate = (32 if ssw_compatible else 30) * 12 * u.ct / u.s
    assert_allclose(per_bin.counts - subtracted.counts, expected_rate * exposure)
    assert subtracted.counts[0, 0, 0, 0] < 0 * u.ct

    combined = science.get_data(time_indices=[[0, 1]], **options)
    combined_subtracted = subtract_background(combined, background)
    assert_allclose(combined_subtracted.counts, subtracted.counts.sum(axis=0, keepdims=True))
    if ssw_compatible:
        expected_variance = (subtracted.counts_err**2).sum(axis=0, keepdims=True)
    else:
        background_error = np.sqrt(subtracted.counts_err**2 - per_bin.counts_err**2)
        expected_variance = combined.counts_err**2 + background_error.sum(axis=0, keepdims=True) ** 2
    assert_allclose(combined_subtracted.counts_err**2, expected_variance)
