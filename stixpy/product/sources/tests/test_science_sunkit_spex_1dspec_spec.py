from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.io import fits

from stixpy.product import Product
from stixpy.product.sources.science import subtract_background

Spectrum = pytest.importorskip("sunkit_spex.spectrum.spectrum").Spectrum
to_spectrum = pytest.importorskip("stixpy.spectroscopy").to_spectrum

# Integration window for the spectrum selection. The reference file stores
# EXPTIME ~4.05 s, consistent with a ~5 s window.
T_RANGE = [["2024-03-10T12:05:50", "2024-03-10T12:05:55"]]

# TODO: point this at the spectrogram (sci-xray-spec) file covering the window
# above. This is NOT the CPD file used by test_science_sunkit_spex_1dspec.py.
SPEC_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-spec_20240310T073505-20240310T134008_V02_2403103995-60712.fits"

# Background must be pixel data (sci-xray-cpd), as stx_convert_spectrogram
# documents for fits_path_bk.
BKG_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/08/SCI/solo_L1_stix-sci-xray-cpd_20240308T193915-20240308T203235_V02_2403087339-57240.fits"

# Reference values live in a FITS file next to this test module (data/), with
# extensions:
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
# SRM/GEOAREA are only stored once (from the elut_correction=True,
# bkg=<bkg product> call) since the SRM doesn't depend on ELUT correction or
# background subtraction.
#
# No flare_location is passed: the calls under test leave it out, so the SRM is
# built without a flare-location correction.
DATA_DIR = Path(__file__).parent / "data"
EXPECTED_VALUES_PATH = DATA_DIR / "expected_values_spec_sum.fits.gz"

# See the note in test_science_sunkit_spex_1dspec.py: GEOAREA is stored rounded
# to ~6 significant figures, so rtol=1e-6 fails on geo area alone.
GEO_AREA_RTOL = 1e-4

# Suffixes of the four counts/counts_err/EXPTIME extension groups, keyed by
# (elut_correction, background subtracted?).
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
def spec_2024_03_10():
    return Product(SPEC_URL)


@pytest.fixture(scope="module")
def bkg_2024_03_10():
    return Product(BKG_URL)


@pytest.fixture(scope="module")
def expected_spec():
    return _read_expected_values(EXPECTED_VALUES_PATH)


def _get_spectrum(spec_prod, *, time_indices, elut_correction, bkg):
    """Make a spectrum from a spectrogram product.

    No detector_indices or pixel_indices are passed: a spectrogram is already summed over detectors and pixels on
    board.
    """
    data = spec_prod.get_data(vtype="c", time_indices=time_indices, elut_correction=elut_correction)
    if bkg is not None:
        data = subtract_background(data, bkg)
    # No photon-axis trim: the reference SRM was generated before srm_e_min existed, so it still spans the full 3210
    # photon bins. Passing False keeps the SRM untrimmed and the stored shape valid.
    spectrum = to_spectrum(data, systematic_error=True, srm_e_min=False)
    statistical = to_spectrum(data, systematic_error=False, srm_e_min=False)
    assert_allclose(spectrum.data, statistical.data)
    edges = spectrum.spectral_axis.bin_edges.to_value(u.keV)
    centres = (edges[:-1] + edges[1:]) / 2
    fraction = np.select([centres < 7, centres < 10], [0.07, 0.05], default=0.03)
    expected_error = np.hypot(statistical.uncertainty.array, fraction * np.asarray(spectrum.data))
    assert_allclose(spectrum.uncertainty.array, expected_error, rtol=1e-12)
    return spectrum


# ---------------------------------------------------------------------------
# Fixtures: one per (elut, bkgsub) combination. Single time bin ->
# get_data returns one Spectrum rather than a sequence/collection.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec_bkgsub_elut(spec_2024_03_10, bkg_2024_03_10):
    return _get_spectrum(
        spec_2024_03_10,
        time_indices=T_RANGE,
        elut_correction=True,
        bkg=bkg_2024_03_10,
    )


@pytest.fixture(scope="module")
def spec_bkgsub_noelut(spec_2024_03_10, bkg_2024_03_10):
    return _get_spectrum(
        spec_2024_03_10,
        time_indices=T_RANGE,
        elut_correction=False,
        bkg=bkg_2024_03_10,
    )


@pytest.fixture(scope="module")
def spec_nobkgsub_elut(spec_2024_03_10):
    return _get_spectrum(
        spec_2024_03_10,
        time_indices=T_RANGE,
        elut_correction=True,
        bkg=None,
    )


@pytest.fixture(scope="module")
def spec_nobkgsub_noelut(spec_2024_03_10):
    return _get_spectrum(
        spec_2024_03_10,
        time_indices=T_RANGE,
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
    assert np.all(np.isfinite(counts_err))
    # Historical FITS fixtures used clipping and earlier uncertainty
    # conventions. Live HISSW tests cover signed counts and uncertainties.
    if expected is not None and variant == _VARIANTS[("noelut", "nobkgsub")]:
        assert_allclose(counts, expected[variant]["counts"], rtol=3e-6)


# ---------------------------------------------------------------------------
# is-a-Spectrum checks
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_spec_bkgsub_elut_is_spectrum(spec_bkgsub_elut):
    _assert_is_spectrum(spec_bkgsub_elut)


@pytest.mark.remote_data
def test_spec_bkgsub_noelut_is_spectrum(spec_bkgsub_noelut):
    _assert_is_spectrum(spec_bkgsub_noelut)


@pytest.mark.remote_data
def test_spec_nobkgsub_elut_is_spectrum(spec_nobkgsub_elut):
    _assert_is_spectrum(spec_nobkgsub_elut)


@pytest.mark.remote_data
def test_spec_nobkgsub_noelut_is_spectrum(spec_nobkgsub_noelut):
    _assert_is_spectrum(spec_nobkgsub_noelut)


# ---------------------------------------------------------------------------
# SRM / GEOAREA (only stored/tested once, from the elut+bkgsub call)
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_srm(spec_bkgsub_elut, expected_spec):
    _assert_srm(spec_bkgsub_elut, expected_spec)


@pytest.mark.remote_data
def test_geo_area(spec_bkgsub_elut, expected_spec):
    _assert_geo_area(spec_bkgsub_elut, expected_spec)


# ---------------------------------------------------------------------------
# exposure time + counts, per elut/bkgsub combo
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_exposure_time_bkgsub_elut(spec_bkgsub_elut, expected_spec):
    _assert_exposure_time(spec_bkgsub_elut, expected_spec, _VARIANTS[("elut", "bkgsub")])


@pytest.mark.remote_data
def test_counts_bkgsub_elut(spec_bkgsub_elut, expected_spec):
    _assert_counts(spec_bkgsub_elut, expected_spec, _VARIANTS[("elut", "bkgsub")])


@pytest.mark.remote_data
def test_exposure_time_bkgsub_noelut(spec_bkgsub_noelut, expected_spec):
    _assert_exposure_time(spec_bkgsub_noelut, expected_spec, _VARIANTS[("noelut", "bkgsub")])


@pytest.mark.remote_data
def test_counts_bkgsub_noelut(spec_bkgsub_noelut, expected_spec):
    _assert_counts(spec_bkgsub_noelut, expected_spec, _VARIANTS[("noelut", "bkgsub")])


@pytest.mark.remote_data
def test_exposure_time_nobkgsub_elut(spec_nobkgsub_elut, expected_spec):
    _assert_exposure_time(spec_nobkgsub_elut, expected_spec, _VARIANTS[("elut", "nobkgsub")])


@pytest.mark.remote_data
def test_counts_nobkgsub_elut(spec_nobkgsub_elut, spec_nobkgsub_noelut):
    # Spectrograms are already summed onboard and receive no ELUT correction.
    assert_allclose(spec_nobkgsub_elut.data, spec_nobkgsub_noelut.data)
    assert_allclose(spec_nobkgsub_elut.uncertainty.array, spec_nobkgsub_noelut.uncertainty.array)


@pytest.mark.remote_data
def test_exposure_time_nobkgsub_noelut(spec_nobkgsub_noelut, expected_spec):
    _assert_exposure_time(spec_nobkgsub_noelut, expected_spec, _VARIANTS[("noelut", "nobkgsub")])


@pytest.mark.remote_data
def test_counts_nobkgsub_noelut(spec_nobkgsub_noelut, expected_spec):
    _assert_counts(spec_nobkgsub_noelut, expected_spec, _VARIANTS[("noelut", "nobkgsub")])
