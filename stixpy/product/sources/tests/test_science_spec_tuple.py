from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.io import fits

from stixpy.calibration.elut import get_elut_correction
from stixpy.product import Product
from stixpy.product.sources._selection import _kept_energy_bins, _on_indices
from stixpy.product.sources.science import subtract_background

SPEC_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-spec_20240310T073505-20240310T134008_V02_2403103995-60712.fits"
# A pixel-data background, as in STIX-GSW's stx_convert_spectrogram
BKG_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/08/SCI/solo_L1_stix-sci-xray-cpd_20240308T193915-20240308T203235_V02_2403087339-57240.fits"

EXPECTED_VALUES_PATH = Path(__file__).parent / "data" / "expected_values_spec_tuple.fits.gz"

# Peak rate is ~2e5 ct/s, so one float32 ULP of the IDL output is ~0.02. atol has
# to sit above that: near-zero background-subtracted bins inherit the absolute
# rounding of large operands, so their relative error is unbounded.
RTOL = 1e-6
ATOL = 5e-4

_LABELS = ("WITH_BKGSUB", "WITHOUT_BKGSUB")


def _read_expected_values(path):
    """Load the reference extensions, or None if the file isn't there yet.

    Returning None keeps the shape and sanity assertions live without reference
    data, so the tests still catch crashes, NaNs and negative errors.
    """
    if not path.exists():
        return None
    with fits.open(path) as hdul:
        # Index by full EXTNAME rather than matching on a "COUNTS_" prefix:
        # COUNTS_ERR_* also starts with COUNTS_, so prefix matching would pull
        # the error extensions in as counts.
        return {
            label: {
                "counts": hdul[f"COUNTS_{label}"].data,
                "counts_err": hdul[f"COUNTS_ERR_{label}"].data,
            }
            for label in _LABELS
        }


@pytest.fixture(scope="module")
def spec_product():
    return Product(SPEC_URL)


@pytest.fixture(scope="module")
def bkg_product():
    return Product(BKG_URL)


@pytest.fixture(scope="module")
def expected_values():
    return _read_expected_values(EXPECTED_VALUES_PATH)


def _to_value(array):
    """Strip units if present so comparisons against the stored arrays work."""
    if isinstance(array, u.Quantity):
        return array.to_value(array.unit)
    return np.asarray(array)


def _get_summed_counts(spec_prod, bkg=None):
    """Call get_data and reduce to (n_time, n_energy).

    No detector_indices or pixel_indices are passed: a spectrogram is already
    summed over detectors and pixels onboard.
    """
    # The saved IDL reference uses integer triggers.
    spec = spec_prod.get_data(vtype="cr", elut_correction=True, livetime_correction=True, ssw_compatible=True)
    if bkg is not None:
        spec = subtract_background(spec, bkg)

    counts = np.nansum(spec.counts, axis=(1, 2))
    counts_err = np.sqrt(np.nansum(spec.counts_err**2, axis=(1, 2)))
    return counts, counts_err


@pytest.fixture(scope="module")
def spec_bkgsub(spec_product, bkg_product):
    return _get_summed_counts(spec_product, bkg_product)


@pytest.fixture(scope="module")
def spec_nobkgsub(spec_product):
    return _get_summed_counts(spec_product)


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _assert_counts(result, expected, label):
    counts = _to_value(result[0])
    assert counts.ndim == 2
    assert np.all(np.isfinite(counts))
    background_subtracted = "WITH_BKGSUB" in label
    if not background_subtracted:
        assert np.all(counts >= 0)
    if expected is not None:
        # The saved IDL reference clips negative background-subtracted bins.
        counts = counts * _reference_elut_factor()
        comparison = np.maximum(counts, 0) if background_subtracted else counts
        assert_allclose(comparison, expected[label]["counts"], rtol=RTOL, atol=ATOL)


def _assert_counts_err(result, expected, label):
    counts_err = _to_value(result[1])
    assert counts_err.ndim == 2
    assert np.all(np.isfinite(counts_err))
    assert np.all(counts_err >= 0)
    if expected is not None:
        counts_err = counts_err * _reference_elut_factor()
        assert_allclose(counts_err, expected[label]["counts_err"], rtol=RTOL, atol=ATOL)


def _reference_elut_factor():
    """
    The ELUT factor the reference spectrogram counts include, averaged over the detectors and pixels summed on board.

    get_data no longer ELUT corrects spectrograms, as STIX-GSW does not; this is the factor it applied before.
    """
    spec_prod = Product(SPEC_URL)
    _, _, bins, bins_actual = get_elut_correction(np.array(spec_prod.energies["channel"]), spec_prod)
    detectors, pixels = _on_indices(spec_prod.detector_masks.masks), _on_indices(spec_prod.pixel_masks.masks)

    def mean(values):
        return values[:, :, pixels].mean(axis=2)[:, detectors].mean(axis=1)

    return (mean(bins) / mean(bins_actual))[..., _kept_energy_bins(spec_prod.energies)]


# ---------------------------------------------------------------------------
# bkg=<bkg product>
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_counts_bkgsub(spec_bkgsub, expected_values):
    _assert_counts(spec_bkgsub, expected_values, "WITH_BKGSUB")


@pytest.mark.remote_data
def test_counts_err_bkgsub(spec_bkgsub, expected_values):
    _assert_counts_err(spec_bkgsub, expected_values, "WITH_BKGSUB")


# ---------------------------------------------------------------------------
# bkg=None
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_counts_nobkgsub(spec_nobkgsub, expected_values):
    _assert_counts(spec_nobkgsub, expected_values, "WITHOUT_BKGSUB")


@pytest.mark.remote_data
def test_counts_err_nobkgsub(spec_nobkgsub, expected_values):
    _assert_counts_err(spec_nobkgsub, expected_values, "WITHOUT_BKGSUB")
