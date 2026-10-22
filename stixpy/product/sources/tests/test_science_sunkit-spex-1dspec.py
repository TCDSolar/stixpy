import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sunkit_spex.spectrum.spectrum import Spectrum

import astropy.units as u
from astropy.io import fits

from stixpy.imaging.flare_location import estimate_flare_location
from stixpy.product import Product

# Each selection uses its own integration window. The reference files bear this
# out: the top24 file stores EXPTIME ~4.00 s (the 5 s window below) while the
# bkgdet file stores EXPTIME ~53.58 s (the 60 s window).
T_RANGE_TOP24 = ["2024-03-10T12:05:50", "2024-03-10T12:05:55"]
T_RANGE_BKGDET = ["2024-03-10T12:05:40", "2024-03-10T12:06:40"]

# TODO: point these at the CPD/background files covering the time ranges above.
CPD_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-cpd_20240310T115906-20240310T121540_V02_2403109216-57290.fits"
BKG_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/08/SCI/solo_L1_stix-sci-xray-cpd_20240308T193915-20240308T203235_V02_2403087339-57240.fits"

# Reference values live in FITS files next to this test module (data/):
#   expected_values_top24_detector_sum_new.fits    -> detector_indices="top24"
#   expected_values_bkg_detector_sum_new.fits       -> detector_indices=[9], pixel_indices=[2, 5]
#
# Each file has extensions:
#   SRM                          (3210, 30), header GEOAREA [cm^2]
#   COUNTS_WITH_BKG              (30,), header EXPTIME [s]
#   COUNTS_ERR_WITH_BKG          (30,)
#   COUNTS_WITH_BKG_NO_ELUT      (30,), header EXPTIME [s]
#   COUNTS_ERR_WITH_BKG_NO_ELUT  (30,)
#   COUNTS_NO_BKG                (30,), header EXPTIME [s]
#   COUNTS_ERR_NO_BKG            (30,)
#   COUNTS_NO_BKG_NO_ELUT        (30,), header EXPTIME [s]
#   COUNTS_ERR_NO_BKG_NO_ELUT    (30,)
#
# SRM/GEOAREA are only stored once per file (from the elut_correction=True,
# bkg=<bkg product> call) since the SRM doesn't depend on ELUT correction or
# background subtraction. Flare location isn't stored here - it's estimated
# fresh from the CPD product every run, via the `flare_location` fixture below.
DATA_DIR = Path(__file__).parent / "data"
EXPECTED_VALUES_TOP24_PATH = DATA_DIR / "expected_values_top24_detector_sum_new.fits.gz"
EXPECTED_VALUES_BKGDET_PATH = DATA_DIR / "expected_values_bkg_detector_sum_new.fits.gz"

# The GEOAREA card in the top24 reference file (18.4695) appears to have been
# written rounded to 6 significant figures: the bkgdet file stores exactly
# 2 * 0.096195 = 0.19239 cm^2 for its two pixels, which scaled to the 192
# pixels of top24 gives 18.46944, i.e. 3.2e-6 relative off the stored value.
# rtol=1e-6 (as used for exposure time) therefore fails on geo area alone, so
# use a tolerance that accommodates the stored precision. If the reference file
# is ever regenerated with full precision this can be tightened back to 1e-6.
GEO_AREA_RTOL = 1e-4

# Suffixes of the four counts/counts_err/EXPTIME extension groups in each file,
# keyed by (elut_correction, background subtracted?).
_VARIANTS = {
    ("elut", "bkgsub"): "WITH_BKG",
    ("noelut", "bkgsub"): "WITH_BKG_NO_ELUT",
    ("elut", "nobkgsub"): "NO_BKG",
    ("noelut", "nobkgsub"): "NO_BKG_NO_ELUT",
}


def _read_expected_values(path):
    """Load all reference extensions from an expected-values FITS file.

    Returns None if the file hasn't been generated/placed yet, so tests can
    still exercise shape/sanity checks without failing on missing data.
    """
    if not path.exists():
        return None
    with fits.open(path) as hdul:
        values = {
            "geo_area": hdul["SRM"].header["GEOAREA"],
            "srm": hdul["SRM"].data,
        }
        for suffix in _VARIANTS.values():
            counts_hdu = hdul[f"COUNTS_{suffix}"]
            values[suffix] = {
                "counts": counts_hdu.data,
                "counts_err": hdul[f"COUNTS_ERR_{suffix}"].data,
                "exposure_time_s": counts_hdu.header["EXPTIME"],
            }
        return values


@pytest.fixture(scope="module")
def cpd_2024_03_10():
    return Product(CPD_URL)


@pytest.fixture(scope="module")
def bkg_2024_03_10():
    return Product(BKG_URL)


@pytest.fixture(scope="module")
def flare_location():
    t_range = ["2024-03-10T12:05:50", "2024-03-10T12:06:00"]
    return estimate_flare_location(CPD_URL, t_range, plot=False)


@pytest.fixture(scope="module")
def expected_top24():
    return _read_expected_values(EXPECTED_VALUES_TOP24_PATH)


@pytest.fixture(scope="module")
def expected_bkgdet():
    return _read_expected_values(EXPECTED_VALUES_BKGDET_PATH)


def _get_spectrum(cpd, flare_location, *, time_indices, detector_indices, elut_correction, bkg, pixel_indices=None):
    kwargs = {
        "time_indices": time_indices,
        "sunkit_spex_spectrum": True,
        "flare_location": flare_location,
        "elut_correction": elut_correction,
        "detector_indices": detector_indices,
        "bkg": bkg,
        "sunkit_spex_detector_sum": True,
        "sunkit_spex_systematic_error": True,
        # No photon-axis trim: the reference SRM was generated before get_data
        # gained srm_e_min, so it still spans the full 3210 photon bins. Passing
        # False keeps the SRM untrimmed and the stored shape valid.
        "srm_e_min": False,
    }
    if pixel_indices is not None:
        kwargs["pixel_indices"] = pixel_indices
    # stixpy emits a NumPy DeprecationWarning ("Conversion of an array with
    # ndim > 0 to a scalar is deprecated") on the single-detector/pixel-subset
    # path used by the bkgdet selection. pytest >= 8 re-emits any warning that
    # `pytest.warns` didn't match, so with `filterwarnings = error` in the
    # pytest config that warning escalates and turns *fixture setup* into an
    # ERROR for every test depending on the fixture. Suppress it here so the
    # suppression stays scoped to this one call rather than going in the global
    # config - it's an upstream bug worth fixing in stixpy, not a test problem.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with pytest.warns(UserWarning, match="sunkit_spex_spectrum = True"):
            return cpd.get_data(**kwargs)


# ---------------------------------------------------------------------------
# top24 selection: detector_indices="top24", detectors summed, single time
# bin -> get_data returns one Spectrum rather than a sequence/collection.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec_top24_bkgsub_elut(cpd_2024_03_10, bkg_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_TOP24,
        detector_indices="top24",
        elut_correction=True,
        bkg=bkg_2024_03_10,
    )


@pytest.fixture(scope="module")
def spec_top24_bkgsub_noelut(cpd_2024_03_10, bkg_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_TOP24,
        detector_indices="top24",
        elut_correction=False,
        bkg=bkg_2024_03_10,
    )


@pytest.fixture(scope="module")
def spec_top24_nobkgsub_elut(cpd_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_TOP24,
        detector_indices="top24",
        elut_correction=True,
        bkg=None,
    )


@pytest.fixture(scope="module")
def spec_top24_nobkgsub_noelut(cpd_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_TOP24,
        detector_indices="top24",
        elut_correction=False,
        bkg=None,
    )


# ---------------------------------------------------------------------------
# bkgdet selection: detector_indices=[9], pixel_indices=[2, 5], detectors
# summed, single time bin -> get_data returns one Spectrum.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec_bkgdet_bkgsub_elut(cpd_2024_03_10, bkg_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_BKGDET,
        detector_indices="bkg",
        elut_correction=True,
        bkg=bkg_2024_03_10,
    )


@pytest.fixture(scope="module")
def spec_bkgdet_bkgsub_noelut(cpd_2024_03_10, bkg_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_BKGDET,
        detector_indices="bkg",
        elut_correction=False,
        bkg=bkg_2024_03_10,
    )


@pytest.fixture(scope="module")
def spec_bkgdet_nobkgsub_elut(cpd_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_BKGDET,
        detector_indices="bkg",
        elut_correction=True,
        bkg=None,
    )


@pytest.fixture(scope="module")
def spec_bkgdet_nobkgsub_noelut(cpd_2024_03_10, flare_location):
    return _get_spectrum(
        cpd_2024_03_10,
        flare_location,
        time_indices=T_RANGE_BKGDET,
        detector_indices="bkg",
        elut_correction=False,
        bkg=None,
    )


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _assert_is_spectrum(spec):
    assert isinstance(spec, Spectrum)
    assert hasattr(spec, "data")
    assert hasattr(spec, "uncertainty")
    assert hasattr(spec, "spectral_axis")
    assert hasattr(spec.spectral_axis, "bin_edges")
    assert hasattr(spec, "meta")


def _assert_srm(spec, expected):
    srm = np.asarray(spec.meta["srm"])
    assert srm.shape == (3210, 30)
    assert np.all(np.isfinite(srm))
    assert np.all(srm >= 0)
    if expected is not None:
        assert_allclose(srm, expected["srm"], rtol=5e-3)


def _assert_geo_area(spec, expected):
    # geo_area may come back as a plain float or as a Quantity; normalise to a
    # bare value in cm^2 so the comparison against the GEOAREA card works either
    # way (assert_allclose on a Quantity vs a float is not reliable).
    geo_area = u.Quantity(spec.meta["geo_area"], u.cm**2)
    geo_area_cm2 = geo_area.to_value(u.cm**2)
    assert geo_area_cm2 > 0
    if expected is not None:
        assert_allclose(geo_area_cm2, expected["geo_area"], rtol=GEO_AREA_RTOL)


def _assert_exposure_time(spec, expected, variant):
    exposure_time = spec.meta["exposure_time"].to(u.s)
    assert exposure_time.value > 0
    if expected is not None:
        assert_allclose(exposure_time.value, expected[variant]["exposure_time_s"], rtol=1e-6)


def _assert_counts(spec, expected, variant):
    counts = np.asarray(spec.data)
    counts_err = np.asarray(spec.uncertainty.array)
    assert counts.shape == (30,)
    assert counts_err.shape == (30,)
    assert np.all(np.isfinite(counts))
    assert np.all(counts_err >= 0)
    if expected is not None:
        assert_allclose(counts, expected[variant]["counts"], rtol=1e-7)
        assert_allclose(counts_err, expected[variant]["counts_err"], rtol=3e-7)


# ---------------------------------------------------------------------------
# top24: is-a-Spectrum checks
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_spec_top24_bkgsub_elut_is_spectrum(spec_top24_bkgsub_elut):
    _assert_is_spectrum(spec_top24_bkgsub_elut)


@pytest.mark.remote_data
def test_spec_top24_bkgsub_noelut_is_spectrum(spec_top24_bkgsub_noelut):
    _assert_is_spectrum(spec_top24_bkgsub_noelut)


@pytest.mark.remote_data
def test_spec_top24_nobkgsub_elut_is_spectrum(spec_top24_nobkgsub_elut):
    _assert_is_spectrum(spec_top24_nobkgsub_elut)


@pytest.mark.remote_data
def test_spec_top24_nobkgsub_noelut_is_spectrum(spec_top24_nobkgsub_noelut):
    _assert_is_spectrum(spec_top24_nobkgsub_noelut)


# ---------------------------------------------------------------------------
# top24: SRM / GEOAREA (only stored/tested once, from the elut+bkgsub call)
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_srm_top24_detector_sum(spec_top24_bkgsub_elut, expected_top24):
    _assert_srm(spec_top24_bkgsub_elut, expected_top24)


@pytest.mark.remote_data
def test_geo_area_top24_detector_sum(spec_top24_bkgsub_elut, expected_top24):
    _assert_geo_area(spec_top24_bkgsub_elut, expected_top24)


# ---------------------------------------------------------------------------
# top24: exposure time + counts, per elut/bkgsub combo
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_exposure_time_top24_bkgsub_elut(spec_top24_bkgsub_elut, expected_top24):
    _assert_exposure_time(spec_top24_bkgsub_elut, expected_top24, _VARIANTS[("elut", "bkgsub")])


@pytest.mark.remote_data
def test_counts_top24_bkgsub_elut(spec_top24_bkgsub_elut, expected_top24):
    _assert_counts(spec_top24_bkgsub_elut, expected_top24, _VARIANTS[("elut", "bkgsub")])


@pytest.mark.remote_data
def test_exposure_time_top24_bkgsub_noelut(spec_top24_bkgsub_noelut, expected_top24):
    _assert_exposure_time(spec_top24_bkgsub_noelut, expected_top24, _VARIANTS[("noelut", "bkgsub")])


@pytest.mark.remote_data
def test_counts_top24_bkgsub_noelut(spec_top24_bkgsub_noelut, expected_top24):
    _assert_counts(spec_top24_bkgsub_noelut, expected_top24, _VARIANTS[("noelut", "bkgsub")])


@pytest.mark.remote_data
def test_exposure_time_top24_nobkgsub_elut(spec_top24_nobkgsub_elut, expected_top24):
    _assert_exposure_time(spec_top24_nobkgsub_elut, expected_top24, _VARIANTS[("elut", "nobkgsub")])


@pytest.mark.remote_data
def test_counts_top24_nobkgsub_elut(spec_top24_nobkgsub_elut, expected_top24):
    _assert_counts(spec_top24_nobkgsub_elut, expected_top24, _VARIANTS[("elut", "nobkgsub")])


@pytest.mark.remote_data
def test_exposure_time_top24_nobkgsub_noelut(spec_top24_nobkgsub_noelut, expected_top24):
    _assert_exposure_time(spec_top24_nobkgsub_noelut, expected_top24, _VARIANTS[("noelut", "nobkgsub")])


@pytest.mark.remote_data
def test_counts_top24_nobkgsub_noelut(spec_top24_nobkgsub_noelut, expected_top24):
    _assert_counts(spec_top24_nobkgsub_noelut, expected_top24, _VARIANTS[("noelut", "nobkgsub")])


# ---------------------------------------------------------------------------
# bkgdet: is-a-Spectrum checks
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_spec_bkgdet_bkgsub_elut_is_spectrum(spec_bkgdet_bkgsub_elut):
    _assert_is_spectrum(spec_bkgdet_bkgsub_elut)


@pytest.mark.remote_data
def test_spec_bkgdet_bkgsub_noelut_is_spectrum(spec_bkgdet_bkgsub_noelut):
    _assert_is_spectrum(spec_bkgdet_bkgsub_noelut)


@pytest.mark.remote_data
def test_spec_bkgdet_nobkgsub_elut_is_spectrum(spec_bkgdet_nobkgsub_elut):
    _assert_is_spectrum(spec_bkgdet_nobkgsub_elut)


@pytest.mark.remote_data
def test_spec_bkgdet_nobkgsub_noelut_is_spectrum(spec_bkgdet_nobkgsub_noelut):
    _assert_is_spectrum(spec_bkgdet_nobkgsub_noelut)


# ---------------------------------------------------------------------------
# bkgdet: SRM / GEOAREA (only stored/tested once, from the elut+bkgsub call)
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_srm_bkgdet_detector_sum(spec_bkgdet_bkgsub_elut, expected_bkgdet):
    _assert_srm(spec_bkgdet_bkgsub_elut, expected_bkgdet)


@pytest.mark.remote_data
def test_geo_area_bkgdet_detector_sum(spec_bkgdet_bkgsub_elut, expected_bkgdet):
    _assert_geo_area(spec_bkgdet_bkgsub_elut, expected_bkgdet)


# ---------------------------------------------------------------------------
# bkgdet: exposure time + counts, per elut/bkgsub combo
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_exposure_time_bkgdet_bkgsub_elut(spec_bkgdet_bkgsub_elut, expected_bkgdet):
    _assert_exposure_time(spec_bkgdet_bkgsub_elut, expected_bkgdet, _VARIANTS[("elut", "bkgsub")])


@pytest.mark.remote_data
def test_counts_bkgdet_bkgsub_elut(spec_bkgdet_bkgsub_elut, expected_bkgdet):
    _assert_counts(spec_bkgdet_bkgsub_elut, expected_bkgdet, _VARIANTS[("elut", "bkgsub")])


@pytest.mark.remote_data
def test_exposure_time_bkgdet_bkgsub_noelut(spec_bkgdet_bkgsub_noelut, expected_bkgdet):
    _assert_exposure_time(spec_bkgdet_bkgsub_noelut, expected_bkgdet, _VARIANTS[("noelut", "bkgsub")])


@pytest.mark.remote_data
def test_counts_bkgdet_bkgsub_noelut(spec_bkgdet_bkgsub_noelut, expected_bkgdet):
    _assert_counts(spec_bkgdet_bkgsub_noelut, expected_bkgdet, _VARIANTS[("noelut", "bkgsub")])


@pytest.mark.remote_data
def test_exposure_time_bkgdet_nobkgsub_elut(spec_bkgdet_nobkgsub_elut, expected_bkgdet):
    _assert_exposure_time(spec_bkgdet_nobkgsub_elut, expected_bkgdet, _VARIANTS[("elut", "nobkgsub")])


@pytest.mark.remote_data
def test_counts_bkgdet_nobkgsub_elut(spec_bkgdet_nobkgsub_elut, expected_bkgdet):
    _assert_counts(spec_bkgdet_nobkgsub_elut, expected_bkgdet, _VARIANTS[("elut", "nobkgsub")])


@pytest.mark.remote_data
def test_exposure_time_bkgdet_nobkgsub_noelut(spec_bkgdet_nobkgsub_noelut, expected_bkgdet):
    _assert_exposure_time(spec_bkgdet_nobkgsub_noelut, expected_bkgdet, _VARIANTS[("noelut", "nobkgsub")])


@pytest.mark.remote_data
def test_counts_bkgdet_nobkgsub_noelut(spec_bkgdet_nobkgsub_noelut, expected_bkgdet):
    _assert_counts(spec_bkgdet_nobkgsub_noelut, expected_bkgdet, _VARIANTS[("noelut", "nobkgsub")])
