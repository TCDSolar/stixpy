from pathlib import Path
 
import numpy as np
import pytest
from numpy.testing import assert_allclose
 
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import Helioprojective
 
from stixpy.product import Product
from stixpy.coordinates.flare_location import stx_estimate_flare_location

from sunkit_spex.spectrum.spectrum import SpectralAxis, Spectrum
 
T_RANGE = ["2024-03-10T12:05:50", "2024-03-10T12:05:55"]
 
# TODO: point these at the CPD/background files covering T_RANGE.
CPD_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-cpd_20240310T115906-20240310T121540_V02_2403109216-57290.fits"
BKG_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/08/SCI/solo_L1_stix-sci-xray-cpd_20240308T193915-20240308T203235_V02_2403087339-57240.fits"
 
# Reference values live in a FITS file next to this test module:
#   primary header: GEOAREA (cm^2), EXPTIME (s)
#   extensions:      ENERGY_BIN_SUMS (30,), PHOTON_BLOCK_SUMS (50,),
#                     COUNTS (30,), COUNTS_ERR (30,)
# See generate_expected_values.py for how to (re)generate it. Flare location
# isn't stored here any more - it's estimated fresh from the CPD product
# every run, via the `flare_location` fixture below.
EXPECTED_VALUES_PATH = Path(__file__).parent / "data" / "expected_values_top24_detector_sum.fits"
# The no-background reference lives in the SAME file as above, under distinct
# extensions (COUNTS_NO_BKG / COUNTS_ERR_NO_BKG), since raw counts differ from
# the background-subtracted COUNTS / COUNTS_ERR.
 
 
@pytest.fixture(scope="module")
def cpd_2024_03_10():
    return Product(CPD_URL)
 
 
@pytest.fixture(scope="module")
def bkg_2024_03_10():
    return Product(BKG_URL)
 
 
@pytest.fixture
def expected_values():
    if not EXPECTED_VALUES_PATH.exists():
        return None
    with fits.open(EXPECTED_VALUES_PATH) as hdul:
        header = hdul[0].header
        return {
            "geo_area": header["GEOAREA"],
            "exposure_time_s": header["EXPTIME"],
            "srm": hdul["SRM"].data,
            "counts": hdul["COUNTS"].data,
            "counts_err": hdul["COUNTS_ERR"].data,
        }
 
 
@pytest.fixture
def expected_values_no_bkg():
    if not EXPECTED_VALUES_PATH.exists():
        return None
    with fits.open(EXPECTED_VALUES_PATH) as hdul:
        # No-bkg values are stored in the same file under their own extensions.
        # If they haven't been generated yet, skip the value comparison.
        if "COUNTS_NO_BKG" not in hdul or "COUNTS_ERR_NO_BKG" not in hdul:
            return None
        counts_hdu = hdul["COUNTS_NO_BKG"]
        result = {
            "counts": counts_hdu.data,
            "counts_err": hdul["COUNTS_ERR_NO_BKG"].data,
        }
        # No-bkg exposure time is stored as an EXPTIME card on the COUNTS_NO_BKG
        # extension header (8-char FITS keyword limit rules out EXPTIME_NO_BKG).
        if "EXPTIME" in counts_hdu.header:
            result["exposure_time_s"] = counts_hdu.header["EXPTIME"]
        return result
 
 
@pytest.fixture(scope="module")
def flare_location():
    t_range = ["2024-03-10T12:05:50", "2024-03-10T12:06:00"]
    return stx_estimate_flare_location(CPD_URL, t_range, plot=False)
 
 
@pytest.fixture(scope="module")
def spec_1d(cpd_2024_03_10, bkg_2024_03_10, flare_location):
    """
    detector_indices="top24", detectors summed (the default, passed
    explicitly here), single time bin -> get_data returns one Spectrum
    rather than a sequence/collection.
    """
    with pytest.warns(UserWarning):
        return cpd_2024_03_10.get_data(
            time_indices=T_RANGE,
            sunkit_spex_spectrum=True,
            flare_location=flare_location,
            detector_indices="top24",
            sunkit_spex_detector_sum=True,
            bkg=bkg_2024_03_10,
            sunkit_spex_systematic_error=True,
        )

 
@pytest.fixture(scope="module")
def spec_1d_no_bkg(cpd_2024_03_10, flare_location):
    """
    Same selection as `spec_1d` but with NO background subtraction (bkg=None):
    detector_indices="top24", detectors summed, single time bin -> get_data
    returns one Spectrum. Counts here are raw (not background-subtracted).
    """
    with pytest.warns(UserWarning):
        return cpd_2024_03_10.get_data(
            time_indices=T_RANGE,
            sunkit_spex_spectrum=True,
            flare_location=flare_location,
            detector_indices="top24",
            sunkit_spex_detector_sum=True,
            bkg=None,
            sunkit_spex_systematic_error=True,
        )

 
@pytest.mark.remote_data
def test_spec_1d_is_spectrum(spec_1d):
    assert isinstance(spec_1d, Spectrum)
    assert hasattr(spec_1d, "data")
    assert hasattr(spec_1d, "uncertainty")
    assert hasattr(spec_1d, "spectral_axis")
    assert hasattr(spec_1d.spectral_axis, "bin_edges")
    assert hasattr(spec_1d, "meta")

 
@pytest.mark.remote_data
def test_srm_top24_detector_sum(spec_1d, expected_values):
    srm = np.asarray(spec_1d.meta["srm"])
    assert srm.shape == (3210, 30)
    assert np.all(np.isfinite(srm))
    assert np.all(srm >= 0)
 
    if expected_values is not None:
        assert_allclose(srm, expected_values["srm"], rtol=5e-3)

 
@pytest.mark.remote_data
def test_geo_area_top24_detector_sum(spec_1d, expected_values):
    geo_area = spec_1d.meta["geo_area"]
    assert geo_area > 0
    if expected_values is not None:
        assert_allclose(geo_area, expected_values["geo_area"], rtol=1e-6)
 
 
@pytest.mark.remote_data
def test_exposure_time_top24_detector_sum(spec_1d, expected_values):
    exposure_time = spec_1d.meta["exposure_time"].to(u.s)
    if expected_values is not None:
        assert_allclose(exposure_time.value, expected_values["exposure_time_s"], rtol=1e-6)
 
 
@pytest.mark.remote_data
def test_counts_top24_detector_sum(spec_1d, expected_values):
    counts = np.asarray(spec_1d.data)
    counts_err = np.asarray(spec_1d.uncertainty.array)
    assert counts.shape == (30,)
    assert counts_err.shape == (30,)
    assert np.all(np.isfinite(counts))
    assert np.all(counts_err >= 0)
    if expected_values is not None:
        assert_allclose(counts, expected_values["counts"], rtol=1e-7)
        assert_allclose(counts_err, expected_values["counts_err"], rtol=4e-4)
 
 
@pytest.mark.remote_data
def test_spec_1d_no_bkg_is_spectrum(spec_1d_no_bkg):
    assert isinstance(spec_1d_no_bkg, Spectrum)
    assert hasattr(spec_1d_no_bkg, "data")
    assert hasattr(spec_1d_no_bkg, "uncertainty")
    assert hasattr(spec_1d_no_bkg, "spectral_axis")
    assert hasattr(spec_1d_no_bkg.spectral_axis, "bin_edges")
    assert hasattr(spec_1d_no_bkg, "meta")
 
 
@pytest.mark.remote_data
def test_counts_top24_detector_sum_no_bkg(spec_1d_no_bkg, expected_values_no_bkg):
    counts = np.asarray(spec_1d_no_bkg.data)
    counts_err = np.asarray(spec_1d_no_bkg.uncertainty.array)
    assert counts.shape == (30,)
    assert counts_err.shape == (30,)
    assert np.all(np.isfinite(counts))
    assert np.all(counts_err >= 0)
    if expected_values_no_bkg is not None:
        print(counts/expected_values_no_bkg["counts"])
        assert_allclose(counts, expected_values_no_bkg["counts"], rtol=1e-7)
        assert_allclose(counts_err, expected_values_no_bkg["counts_err"], rtol=4e-4)
 
 
@pytest.mark.remote_data
def test_exposure_time_top24_detector_sum_no_bkg(spec_1d_no_bkg, expected_values_no_bkg):
    exposure_time = spec_1d_no_bkg.meta["exposure_time"].to(u.s)
    assert exposure_time.value > 0
    if expected_values_no_bkg is not None and "exposure_time_s" in expected_values_no_bkg:
        assert_allclose(exposure_time.value, expected_values_no_bkg["exposure_time_s"], rtol=1e-6)