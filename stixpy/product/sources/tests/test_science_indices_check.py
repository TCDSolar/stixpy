"""
Tests for `ScienceData._indices_check`, which validates the detector, pixel and energy
selections passed to `get_data` and fills in the defaults.

Most tests use a stand-in product with only the attributes `_indices_check` reads (the
detector and pixel masks, the counts shape and the energy table), so they run without
downloading data. The tests at the end use a real CPD file and are marked ``remote_data``.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.table import QTable

from stixpy.product import Product
from stixpy.product.sources.science import ScienceData

# The 33 STIX science energy edges: 32 bins from the 0-4 keV bin to the open 150 keV - top bin.
STIX_EDGES = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 70, 76,
              84, 100, 120, 150, np.nan]  # fmt: skip

TOP24 = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

CPD_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-cpd_20240310T115906-20240310T121540_V02_2403109216-57290.fits"
T_RANGE_BKGDET = ["2024-03-10T12:05:40", "2024-03-10T12:06:40"]


def make_product(*, detectors=None, pixels=None, spectrogram=False, first_bin=0):
    """
    Stand-in product with the attributes `_indices_check` reads.

    ``detectors`` and ``pixels`` are the indices set in the masks (default all 32 / 12), ``spectrogram`` gives
    counts without detector and pixel axes, and ``first_bin`` drops bins from the bottom
    of the energy table (1 gives a product without the 0-4 keV bin, i.e. 31 energy rows).
    """
    detectors = range(32) if detectors is None else detectors
    pixels = range(12) if pixels is None else pixels

    detector_mask = np.zeros((1, 32), dtype=int)
    detector_mask[0, list(detectors)] = 1
    pixel_mask = np.zeros((1, 12), dtype=int)
    pixel_mask[0, list(pixels)] = 1

    edges = np.array(STIX_EDGES, dtype=float)[first_bin:]
    energies = QTable({"e_low": edges[:-1] * u.keV, "e_high": edges[1:] * u.keV})

    n_energy = len(energies)
    counts = np.zeros((2, n_energy)) if spectrogram else np.zeros((2, 32, 12, n_energy))

    return SimpleNamespace(
        data={"counts": counts},
        detector_masks=SimpleNamespace(masks=detector_mask),
        pixel_masks=SimpleNamespace(masks=pixel_mask),
        energies=energies,
    )


@pytest.fixture
def cpd():
    """A full compressed pixel data product: 32 detectors, 12 pixels, 32 energy bins."""
    return make_product()


# ---------------------------------------------------------------------------
# Detector indices
# ---------------------------------------------------------------------------


def test_detectors_none_selects_all_in_mask():
    product = make_product(detectors=[d for d in range(32) if d != 3])
    detector_indices, _, _ = ScienceData._indices_check(product, None, None, None)
    assert np.array_equal(detector_indices, [d for d in range(32) if d != 3])


@pytest.mark.parametrize("detector_indices", [[0, 5, 31], np.array([0, 5, 31]), [[0, 3], [10, 12]]])
def test_detectors_in_mask_returned_unchanged(cpd, detector_indices):
    # no warning expected: with filterwarnings = error any warning fails the test
    result, _, _ = ScienceData._indices_check(cpd, detector_indices, None, None)
    assert np.array_equal(result, detector_indices)


def test_detectors_not_in_mask_warn_and_are_kept():
    product = make_product(detectors=range(10))
    with pytest.warns(UserWarning, match=r"detector indices are not available in the product: \[12, 20\]"):
        result, _, _ = ScienceData._indices_check(product, [1, 12, 20], None, None)
    assert np.array_equal(result, [1, 12, 20])


def test_detector_range_not_in_mask_warns():
    product = make_product(detectors=range(10))
    with pytest.warns(UserWarning, match=r"Detector indices \[10, 11\] in range \[8, 11\]"):
        result, _, _ = ScienceData._indices_check(product, [[0, 3], [8, 11]], None, None)
    assert np.array_equal(result, [[0, 3], [8, 11]])


@pytest.mark.parametrize("label", ["top24", "TOP24", "Top24"])
def test_top24_label(cpd, label):
    result, _, _ = ScienceData._indices_check(cpd, label, None, None)
    assert np.array_equal(result, TOP24)
    # no CFL (8), BKG (9) or fine detectors (1a-2c)
    assert not set(result.tolist()) & {8, 9, 10, 11, 12, 16, 17, 18}


def test_top24_label_is_not_checked_against_mask():
    # a label is a fixed set, so a detector missing from the mask gives no warning
    product = make_product(detectors=range(1, 32))
    result, _, _ = ScienceData._indices_check(product, "top24", None, None)
    assert 0 in result


def test_top24_label_keeps_pixel_default():
    product = make_product(pixels=range(8))
    _, pixel_indices, _ = ScienceData._indices_check(product, "top24", None, None)
    assert np.array_equal(pixel_indices, range(8))


def test_unknown_detector_label(cpd):
    with pytest.raises(ValueError, match="Unknown detector label 'cfl'"):
        ScienceData._indices_check(cpd, "cfl", None, None)


def test_detector_indices_as_numpy_array(cpd):
    # used to fail with "truth value of an array ... is ambiguous" from comparing the array to "top24"
    result, _, _ = ScienceData._indices_check(cpd, np.array([5, 6, 7]), None, None)
    assert np.array_equal(result, [5, 6, 7])


def test_spectrogram_detectors_given_warn_and_are_dropped():
    product = make_product(spectrogram=True)
    with pytest.warns(UserWarning, match="spectrogram file"):
        detector_indices, _, _ = ScienceData._indices_check(product, [0, 1, 2], None, None)
    assert detector_indices is None


def test_spectrogram_detectors_none():
    product = make_product(spectrogram=True)
    detector_indices, pixel_indices, _ = ScienceData._indices_check(product, None, None, None)
    assert detector_indices is None
    assert pixel_indices is None


# ---------------------------------------------------------------------------
# detector_indices="bkg" preset: the BKG detector (index 9) with its small-aperture
# pixels [2, 5]
# ---------------------------------------------------------------------------


def test_bkg_label_defaults_to_small_aperture_pixels(cpd):
    with pytest.warns(UserWarning, match=r"small-aperture pixels \[2, 5\]"):
        detector_indices, pixel_indices, _ = ScienceData._indices_check(cpd, "bkg", None, None)
    assert np.array_equal(detector_indices, [9])
    assert np.array_equal(pixel_indices, [2, 5])


@pytest.mark.parametrize("pixel_indices", [[2, 5], [5, 2], np.array([2, 5])])
def test_bkg_label_accepts_pixels_2_and_5(cpd, pixel_indices):
    # no warning expected: with filterwarnings = error any warning fails the test
    detector_indices, pixels, _ = ScienceData._indices_check(cpd, "bkg", pixel_indices, None)
    assert np.array_equal(detector_indices, [9])
    assert sorted(np.asarray(pixels).tolist()) == [2, 5]


@pytest.mark.parametrize("pixel_indices", [[0, 7], [2], [2, 3, 5], [[2, 5]]])
def test_bkg_label_rejects_other_pixels(cpd, pixel_indices):
    # [[2, 5]] is the range form, which would also sum the covered pixels 3 and 4
    with pytest.raises(ValueError, match=r'detector_indices="bkg" uses the BKG detector\'s small-aperture pixels'):
        ScienceData._indices_check(cpd, "bkg", pixel_indices, None)


def test_bkg_label_is_case_insensitive(cpd):
    with pytest.warns(UserWarning, match="small-aperture pixels"):
        detector_indices, _, _ = ScienceData._indices_check(cpd, "BKG", None, None)
    assert np.array_equal(detector_indices, [9])


def test_detector_index_9_keeps_user_pixels(cpd):
    # the numeric index is how to use the BKG detector with other pixels, so it is not restricted
    detector_indices, pixel_indices, _ = ScienceData._indices_check(cpd, [9], [0, 7], None)
    assert np.array_equal(detector_indices, [9])
    assert np.array_equal(pixel_indices, [0, 7])


# ---------------------------------------------------------------------------
# Pixel indices
# ---------------------------------------------------------------------------


def test_pixels_none_selects_all_in_mask():
    product = make_product(pixels=range(8))
    _, pixel_indices, _ = ScienceData._indices_check(product, None, None, None)
    assert np.array_equal(pixel_indices, range(8))


@pytest.mark.parametrize("pixel_indices", [[0, 4, 11], np.array([0, 4, 11]), [[0, 3], [8, 11]]])
def test_pixels_in_mask_returned_unchanged(cpd, pixel_indices):
    _, result, _ = ScienceData._indices_check(cpd, None, pixel_indices, None)
    assert np.array_equal(result, pixel_indices)


def test_pixels_not_in_mask_warn_and_are_kept():
    product = make_product(pixels=range(8))
    with pytest.warns(UserWarning, match=r"pixel indices are not available in the product: \[9, 10\]"):
        _, result, _ = ScienceData._indices_check(product, None, [0, 9, 10], None)
    assert np.array_equal(result, [0, 9, 10])


def test_pixel_range_not_in_mask_warns():
    product = make_product(pixels=range(8))
    with pytest.warns(UserWarning, match=r"Pixel indices \[8, 9\] in range \[4, 9\]"):
        _, result, _ = ScienceData._indices_check(product, None, [[4, 9]], None)
    assert np.array_equal(result, [[4, 9]])


def test_spectrogram_pixels_given_warn_and_are_dropped():
    product = make_product(spectrogram=True)
    with pytest.warns(UserWarning, match="spectrogram file"):
        _, pixel_indices, _ = ScienceData._indices_check(product, None, [0, 1], None)
    assert pixel_indices is None


# ---------------------------------------------------------------------------
# Energy indices (rows of product.energies, i.e. the last axis of counts)
# ---------------------------------------------------------------------------


def test_energy_none_is_not_checked(cpd):
    _, _, energy_indices = ScienceData._indices_check(cpd, None, None, None)
    assert energy_indices is None


@pytest.mark.parametrize("energy_indices", [[0, 5, 31], [[1, 10], [20, 31]]])
def test_energy_in_table_returned_unchanged(cpd, energy_indices):
    _, _, result = ScienceData._indices_check(cpd, None, None, energy_indices)
    assert result == energy_indices


@pytest.mark.parametrize("energy_indices", [[32], [-1], [0, 40]])
def test_energy_outside_table_raises(cpd, energy_indices):
    with pytest.raises(ValueError, match=r"The following energy indices are not included"):
        ScienceData._indices_check(cpd, None, None, energy_indices)


def test_energy_range_outside_table_raises(cpd):
    with pytest.raises(ValueError, match=r"Energy indices \[32, 33\] in range \[30, 33\]"):
        ScienceData._indices_check(cpd, None, None, [[0, 5], [30, 33]])


def test_energy_error_names_full_table_range(cpd):
    # nanmax skips the NaN upper edge of the open top bin
    with pytest.raises(ValueError, match=r"energy indices 0-31 \(0\.0 - 150\.0 keV\)"):
        ScienceData._indices_check(cpd, None, None, [32])


def test_energy_uses_product_rows_without_0_4_bin():
    # 31 rows starting at 4 keV: the last valid index is 30, which used to raise an IndexError
    product = make_product(first_bin=1)
    _, _, result = ScienceData._indices_check(product, None, None, [[0, 30]])
    assert result == [[0, 30]]
    with pytest.raises(ValueError, match=r"energy indices 0-30 \(4\.0 - 150\.0 keV\)"):
        ScienceData._indices_check(product, None, None, [31])


def test_energy_from_kev_passes_check():
    # what get_data does: keV ranges are turned into row indices, then checked
    product = make_product(first_bin=1)
    energy_indices = ScienceData._energy_indices_format([[6, 10], [25, 100]] * u.keV, product.energies)
    assert np.array_equal(energy_indices, [[2, 5], [16, 27]])
    _, _, result = ScienceData._indices_check(product, None, None, energy_indices)
    assert np.array_equal(result, [[2, 5], [16, 27]])


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpd_2024_03_10():
    return Product(CPD_URL)


@pytest.mark.remote_data
def test_bkg_label_get_data(cpd_2024_03_10):
    # pytest.warns re-emits the other informational warnings get_data raises (ELUT,
    # livetime, ...), so ignore those outside it and only require the "bkg" one
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with pytest.warns(UserWarning, match=r"small-aperture pixels \[2, 5\]"):
            counts_label, *_ = cpd_2024_03_10.get_data(time_indices=T_RANGE_BKGDET, detector_indices="bkg")
        counts_explicit, *_ = cpd_2024_03_10.get_data(
            time_indices=T_RANGE_BKGDET, detector_indices=[9], pixel_indices=[2, 5]
        )
    # the preset must select the same data as giving detector 9 and pixels 2 and 5 explicitly
    assert counts_label.shape[1] == 1  # one detector
    assert_allclose(counts_label, counts_explicit)


@pytest.mark.remote_data
def test_bkg_label_get_data_rejects_other_pixels(cpd_2024_03_10):
    with pytest.raises(ValueError, match="small-aperture pixels"):
        cpd_2024_03_10.get_data(time_indices=T_RANGE_BKGDET, detector_indices="bkg", pixel_indices=[0, 7])
