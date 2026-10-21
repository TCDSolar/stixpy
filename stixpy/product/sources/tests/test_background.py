import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.table import QTable

from stixpy.product.sources.science import CompressedPixelData, Spectrogram, SummedCompressedPixelData

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
    ],
)
@pytest.mark.parametrize("vtype", ["c", "dcr"])
def test_zero_background_unchanged(cpd, bkg_zero, selection, vtype):
    without = cpd.get_data(vtype=vtype, **selection, **NO_ELUT)
    with_bkg = cpd.get_data(vtype=vtype, bkg=bkg_zero, **selection, **NO_ELUT)

    assert with_bkg.counts.shape == without.counts.shape
    assert with_bkg.counts.unit == without.counts.unit
    assert_allclose(with_bkg.counts, without.counts)
    assert_allclose(with_bkg.counts_err, without.counts_err)
    assert_allclose(
        np.broadcast_to(with_bkg.livefrac, with_bkg.counts.shape),
        np.broadcast_to(without.livefrac, without.counts.shape),
    )
    assert with_bkg.livefrac_error is None


def test_background_scaled_and_subtracted(cpd, bkg):
    without = cpd.get_data(vtype="c", detector_indices=[0, 1], pixel_indices=[0], **NO_ELUT)
    with_bkg = cpd.get_data(vtype="c", detector_indices=[0, 1], pixel_indices=[0], bkg=bkg, **NO_ELUT)

    # both livetime fractions are the same, so the subtracted background is 2 ct / 100 s for each 1 s science bin
    assert_allclose(with_bkg.counts, without.counts - 0.02 * u.ct)
    assert np.all(with_bkg.counts_err > without.counts_err)


def test_background_spectrogram(bkg_zero, bkg):
    rng = np.random.default_rng(1)
    spec = _make_product(Spectrogram, rng.integers(500, 1000, size=(3, 4)).astype(float), detector_off=(9,))

    without = spec.get_data(vtype="cr", **NO_ELUT)
    zero = spec.get_data(vtype="cr", bkg=bkg_zero, **NO_ELUT)
    assert zero.counts.shape == without.counts.shape == (3, 1, 1, 4)
    assert_allclose(zero.counts, without.counts)
    assert_allclose(zero.counts_err, without.counts_err)

    # the background of the 31 detectors and 12 pixels summed on board into the spectrogram
    with_bkg = spec.get_data(vtype="c", bkg=bkg, **NO_ELUT)
    assert_allclose(with_bkg.counts, spec.get_data(vtype="c", **NO_ELUT).counts - 31 * 12 * 0.02 * u.ct)


def test_background_negative_set_to_zero(cpd):
    big = _make_product(CompressedPixelData, np.full((1, 32, 12, 4), 1e6), timedel=100)
    data = cpd.get_data(vtype="c", detector_indices=[[0, 31]], bkg=big, **NO_ELUT)
    assert np.all(data.counts == 0)
    assert np.all(data.counts_err > 0)


def test_background_livetime_false(cpd, bkg):
    with pytest.raises(ValueError, match="needs the livetime correction"):
        cpd.get_data(livetime_correction=False, bkg=bkg, **NO_ELUT)


def test_background_not_a_product(cpd):
    with pytest.raises(TypeError, match="bkg must be a STIX science product"):
        cpd.get_data(bkg=np.zeros(3), **NO_ELUT)


def test_background_spectrogram_rejected(cpd):
    spec = _make_product(Spectrogram, np.ones((1, 4)), timedel=100)
    with pytest.raises(ValueError, match="must be pixel data"):
        cpd.get_data(bkg=spec, **NO_ELUT)


def test_background_summed_pixels_rejected(cpd, bkg):
    pmask = np.zeros((4, 12), dtype=int)
    for i in range(4):
        pmask[i, i::4] = 1
    scpd = _make_product(SummedCompressedPixelData, np.ones((2, 32, 4, 4)))
    scpd.data["pixel_masks"] = np.tile(pmask, (2, 1, 1))
    scpd.pixel_masks = type(cpd.pixel_masks)(scpd.data["pixel_masks"])
    with pytest.raises(ValueError, match="summed pixel data"):
        scpd.get_data(bkg=bkg, **NO_ELUT)


def test_background_several_time_bins(cpd):
    two = _make_product(CompressedPixelData, np.ones((2, 32, 12, 4)), timedel=100)
    with pytest.raises(ValueError, match="single time bin, but it has 2"):
        cpd.get_data(bkg=two, **NO_ELUT)


def test_background_energy_mismatch(cpd):
    energies = QTable()
    energies["e_low"] = [4, 5, 6, 8] * u.keV
    energies["e_high"] = [5, 6, 8, 10] * u.keV
    other = _make_product(CompressedPixelData, np.ones((1, 32, 12, 4)), timedel=100, energies=energies)
    with pytest.raises(ValueError, match=r"science energy bins \['6-7', '7-8'\] keV are not in the background"):
        cpd.get_data(bkg=other, **NO_ELUT)


def test_background_detector_off(cpd):
    other = _make_product(CompressedPixelData, np.ones((1, 31, 11, 4)), timedel=100, detector_off=(3,), pixel_off=(2,))
    with pytest.raises(ValueError, match=r"Detectors \[3\] and pixels \[2\] are selected but are off"):
        cpd.get_data(bkg=other, **NO_ELUT)
    # detectors and pixels that are not selected need not be in the background
    cpd.get_data(detector_indices=[0, 1], pixel_indices=[0, 1], bkg=other, **NO_ELUT)


def test_background_detector_off_spectrogram():
    spec = _make_product(Spectrogram, np.ones((2, 4)))
    other = _make_product(CompressedPixelData, np.ones((1, 31, 12, 4)), timedel=100, detector_off=(5,))
    with pytest.raises(ValueError, match=r"Detectors \[5\] and pixels \[\] were summed on board"):
        spec.get_data(bkg=other, **NO_ELUT)
