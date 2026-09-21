import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.io import fits

from stixpy.product import Product

SPEC_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-spec_20240310T073505-20240310T134008_V02_2403103995-60712.fits"
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


def _get_summed_counts(spec_prod, bkg):
    """Call get_data in tuple mode and reduce to (n_time, n_energy).

    No detector_indices or pixel_indices are passed: a spectrogram is already
    summed over detectors and pixels onboard, and _indices_check discards them
    for level-4 data with a warning.

    Both warnings are suppressed at the call site rather than in the global
    pytest config, so the suppression stays scoped to this one get_data call and
    nothing in the reductions or assertions below is covered by it:

      - A NumPy DeprecationWarning ("Conversion of an array with ndim > 0 to a
        scalar is deprecated") on the spectrogram path. An upstream bug worth
        fixing in stixpy, not a test problem.
      - A stixpy UserWarning noting that the livetime is averaged across
        detectors to match the IDL behaviour, and that detector/pixel indices
        are ignored for level-4 data.

    With `filterwarnings = error` in the pytest config either one would escalate
    and turn *fixture setup* into an ERROR for every test depending on the
    fixture. UserWarning is ignored wholesale rather than matched on message -
    note this also hides any future UserWarning from get_data, so if these tests
    start disagreeing with the reference values for no obvious reason, narrow
    this filter to a message match and re-run to see what get_data is saying.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        spec = spec_prod.get_data(
            vtype="cr",
            sunkit_spex_spectrum=False,
            elut_correction=True,
            livetime_correction=True,
            bkg=bkg,
        )

    counts = np.nansum(spec[0], axis=(1, 2))
    counts_err = np.sqrt(np.nansum(spec[1] ** 2, axis=(1, 2)))
    return counts, counts_err


@pytest.fixture(scope="module")
def spec_bkgsub(spec_product, bkg_product):
    return _get_summed_counts(spec_product, bkg_product)


@pytest.fixture(scope="module")
def spec_nobkgsub(spec_product):
    return _get_summed_counts(spec_product, None)


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
