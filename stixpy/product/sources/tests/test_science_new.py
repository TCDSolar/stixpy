# from pathlib import Path
 
# import numpy as np
# import pytest
# from numpy.testing import assert_allclose
 
# import astropy.units as u
# from astropy.coordinates import SkyCoord
# from astropy.io import fits
# from astropy.time import Time
# from sunpy.coordinates import Helioprojective
 
# from stixpy.product import Product
# from stixpy.coordinates.flare_location import stx_estimate_flare_location

# T_RANGE = ["2024-03-10T12:05:50", "2024-03-10T12:05:55"]
 

#  # TODO: point these at the CPD/background files covering T_RANGE.
# CPD_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/10/SCI/solo_L1_stix-sci-xray-cpd_20240310T115906-20240310T121540_V02_2403109216-57290.fits"
# BKG_URL = "https://pub099.cs.technik.fhnw.ch/fits/L1/2024/03/08/SCI/solo_L1_stix-sci-xray-cpd_20240308T193915-20240308T203235_V02_2403087339-57240.fits"

 
# # Reference values live in a FITS file next to this test module:
# #   primary header: GEOAREA (cm^2), EXPTIME (s)
# #   extensions:      ENERGY_BIN_SUMS (30,), PHOTON_BLOCK_SUMS (50,),
# #                     COUNTS (30,), COUNTS_ERR (30,)
# # See generate_expected_values.py for how to (re)generate it. Flare location
# # isn't stored here any more - it's estimated fresh from the CPD product
# # every run (see _get_flare_location), so there's nothing to keep in sync.
# EXPECTED_VALUES_PATH = Path(__file__).parent / "data" / "expected_values_top24_detector_sum.fits"
 
 
# @pytest.fixture
# def cpd_2024_03_10():
#     return Product(CPD_URL)
 
 
# @pytest.fixture
# def bkg_2024_03_10():
#     return Product(BKG_URL)
 
 
# @pytest.fixture
# def expected_values():
#     if not EXPECTED_VALUES_PATH.exists():
#         return None
#     with fits.open(EXPECTED_VALUES_PATH) as hdul:
#         header = hdul[0].header
#         return {
#             "geo_area": header["GEOAREA"],
#             "exposure_time_s": header["EXPTIME"],
#             "energy_bin_sums": hdul["ENERGY_BIN_SUMS"].data,
#             "photon_block_sums": hdul["PHOTON_BLOCK_SUMS"].data,
#             "counts": hdul["COUNTS"].data,
#             "counts_err": hdul["COUNTS_ERR"].data,
#         }
 
 
# def _srm_fingerprint(srm, n_blocks=1000):
#     """
#     Cheap stand-in for storing the full (3250, 30) SRM as a reference: sum
#     over energy bins, and sum over photon bins in coarse blocks. Enough to
#     catch shape/scale/localised regressions without the array itself, which
#     matters once this gets repeated per-detector for the non-summed case.
#     """
#     energy_bin_sums = srm.mean(axis=0)
#     photon_bin_sums = srm.mean(axis=1)
#     # edges = np.linspace(0, len(photon_bin_sums), n_blocks + 1).astype(int)
#     # photon_block_sums = np.array([photon_bin_sums[edges[i]:edges[i + 1]].sum() for i in range(n_blocks)])
#     return energy_bin_sums, photon_bin_sums
 
 
# # def _get_flare_location(cpd):
# #     """
# #     Estimate the flare location straight from CPD_URL. stx_estimate_flare_location
# #     requires a path/URL and loads its own copy internally - confirmed by
# #     passing the already-loaded `cpd` product directly, which raised
# #     TypeError - so this is a second, separate load of the same file, not
# #     something we can avoid at this level. Kept as a function (rather than
# #     inlined) so the test and generate_expected_values.py share one estimate.
# #     """
# #     # t_range = [Time(cpd.meta["DATE-BEG"]), Time(cpd.meta["DATE-END"]) - 10 * u.min]
# #     t_range = ["2024-03-10T12:05:50", "2024-03-10T12:05:55"]

 
 
# @pytest.mark.remote_data
# def test_get_data_srm_top24_detector_sum(cpd_2024_03_10, bkg_2024_03_10, expected_values):
#     # Scenario: detector_indices="top24", detectors summed (the default,
#     # passed explicitly here), single time bin -> get_data returns one
#     # Spectrum rather than a sequence/collection.
#     t_range = ["2024-03-10T12:05:50", "2024-03-10T12:06:00"]
#     flare_location = stx_estimate_flare_location(CPD_URL, t_range, plot=False)
 

#     with pytest.warns(UserWarning):
#         spec_1d = cpd_2024_03_10.get_data(
#             time_indices=T_RANGE,
#             sunkit_spex_spectrum=True,
#             flare_location=flare_location,
#             detector_indices="top24",
#             sunkit_spex_detector_sum=True,
#             bkg=bkg_2024_03_10,
#             sunkit_spex_systematic_error=True,
#         )
 
#     srm = np.asarray(spec_1d.meta["srm"])
#     assert srm.shape == (3060, 30)
#     assert np.all(np.isfinite(srm))
#     assert np.all(srm >= 0)
 
#     if expected_values is not None:
#         energy_bin_sums, photon_block_sums = _srm_fingerprint(srm)
#         assert_allclose(energy_bin_sums, expected_values["energy_bin_sums"], rtol=1e-3)
#         assert_allclose(photon_block_sums, expected_values["photon_block_sums"], rtol=3e-3)
 
#     geo_area = spec_1d.meta["geo_area"]
#     assert geo_area > 0
#     if expected_values is not None:
#         assert_allclose(geo_area, expected_values["geo_area"], rtol=1e-6)
 
#     exposure_time = spec_1d.meta["exposure_time"].to(u.s)
#     if expected_values is not None:
#         assert_allclose(exposure_time.value, expected_values["exposure_time_s"], rtol=1e-6)
 
#     counts = np.asarray(spec_1d.data)
#     counts_err = np.asarray(spec_1d.uncertainty.array)
#     assert counts.shape == (30,)
#     assert counts_err.shape == (30,)
#     assert np.all(np.isfinite(counts))
#     assert np.all(counts_err >= 0)
#     if expected_values is not None:
#         assert_allclose(counts, expected_values["counts"], rtol=1e-6)
#         assert_allclose(counts_err, expected_values["counts_err"], rtol=4e-4)


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
            "energy_bin_sums": hdul["ENERGY_BIN_SUMS"].data,
            "photon_block_sums": hdul["PHOTON_BLOCK_SUMS"].data,
            "counts": hdul["COUNTS"].data,
            "counts_err": hdul["COUNTS_ERR"].data,
        }
 
 
def _srm_fingerprint(srm):
    """
    Cheap stand-in for storing the full (3250, 30) SRM as a reference: mean
    over energy bins, and mean over photon bins. Enough to catch
    shape/scale/localised regressions without storing the array itself.
    """
    energy_bin_sums = srm.mean(axis=0)
    photon_bin_sums = srm.mean(axis=1)
    return energy_bin_sums, photon_bin_sums
 
 
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
    assert srm.shape == (3060, 30)
    assert np.all(np.isfinite(srm))
    assert np.all(srm >= 0)
 
    if expected_values is not None:
        energy_bin_sums, photon_block_sums = _srm_fingerprint(srm)
        assert_allclose(energy_bin_sums, expected_values["energy_bin_sums"], rtol=1e-3)
        assert_allclose(photon_block_sums, expected_values["photon_block_sums"], rtol=3e-3)
 
 
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
        assert_allclose(counts, expected_values["counts"], rtol=1e-6)
        assert_allclose(counts_err, expected_values["counts_err"], rtol=4e-4)

