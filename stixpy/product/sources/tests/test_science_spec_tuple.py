from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.io import fits

from stixpy.product import Product

SPEC_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-spec_20240310T073505-20240310T134008_V02_2403103995-60712.fits"

EXPECTED_VALUES_PATH = Path(__file__).parent / "data" / "expected_values_spec_tuple.fits.gz"

# Peak rate is ~2e5 ct/s, so one float32 ULP of the IDL output is ~0.02. atol has
# to sit above that: near-zero background-subtracted bins inherit the absolute
# rounding of large operands, so their relative error is unbounded.
RTOL = 1e-6
ATOL = 5e-4

_LABELS = ("WITHOUT_BKGSUB",)


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
def expected_values():
    return _read_expected_values(EXPECTED_VALUES_PATH)


def _to_value(array):
    """Strip units if present so comparisons against the stored arrays work."""
    if isinstance(array, u.Quantity):
        return array.to_value(array.unit)
    return np.asarray(array)


def _get_summed_counts(spec_prod):
    """Call get_data and reduce to (n_time, n_energy).

    No detector_indices or pixel_indices are passed: a spectrogram is already
    summed over detectors and pixels onboard.
    """
    spec = spec_prod.get_data(vtype="cr", elut_correction=True, livetime_correction=True)

    counts = np.nansum(spec.counts, axis=(1, 2))
    counts_err = np.sqrt(np.nansum(spec.counts_err**2, axis=(1, 2)))
    return counts, counts_err


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
    assert np.all(counts >= 0)
    if expected is not None:
        assert_allclose(counts, expected[label]["counts"], rtol=RTOL, atol=ATOL)


def _assert_counts_err(result, expected, label):
    counts_err = _to_value(result[1])
    assert counts_err.ndim == 2
    assert np.all(np.isfinite(counts_err))
    assert np.all(counts_err >= 0)
    if expected is not None:
        assert_allclose(counts_err, expected[label]["counts_err"], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# bkg=None
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_counts_nobkgsub(spec_nobkgsub, expected_values):
    _assert_counts(spec_nobkgsub, expected_values, "WITHOUT_BKGSUB")


@pytest.mark.remote_data
def test_counts_err_nobkgsub(spec_nobkgsub, expected_values):
    _assert_counts_err(spec_nobkgsub, expected_values, "WITHOUT_BKGSUB")
