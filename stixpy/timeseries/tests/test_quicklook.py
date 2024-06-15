import astropy.units as u
import pytest
from sunpy.timeseries import TimeSeries

from stixpy.timeseries.quicklook import HKMaxi, QLBackground, QLLightCurve, QLVariance


@pytest.mark.remote_data
@pytest.fixture(scope="session")
def ql_lightcurve():
    ql_lc = TimeSeries(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-lightcurve_20200506_V02.fits"
    )
    return ql_lc


@pytest.mark.remote_data
@pytest.fixture(scope="session")
def ql_background():
    ql_bkg = TimeSeries(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-background_20200506_V02.fits"
    )
    return ql_bkg


@pytest.mark.remote_data
@pytest.fixture(scope="session")
def ql_variance():
    ql_var = TimeSeries(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-variance_20200506_V02.fits"
    )
    return ql_var


@pytest.mark.remote_data
@pytest.fixture(scope="session")
def hk_maxi():
    hk_maxi = TimeSeries(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/HK/solo_L1_stix-hk-maxi_20200506_V02.fits"
    )
    return hk_maxi


@pytest.mark.remote_data
def test_ql_lightcurve(ql_lightcurve):
    assert isinstance(ql_lightcurve, QLLightCurve)
    assert ql_lightcurve.quantity("4-10 keV").unit == u.ct / (u.s * u.keV)


@pytest.mark.remote_data
def test_ql_lightcurve_plot(ql_lightcurve):
    ql_lightcurve.plot()
    ql_lightcurve.plot(columns=["4-10 keV"])


@pytest.mark.remote_data
def test_ql_background(ql_background):
    assert isinstance(ql_background, QLBackground)
    assert ql_background.quantity("4-10 keV").unit == u.ct / (u.s * u.keV)


@pytest.mark.remote_data
def test_ql_background_plot(ql_background):
    ql_background.plot()
    ql_background.plot(columns=["4-10 keV"])


@pytest.mark.remote_data
def test_ql_variance(ql_variance):
    assert isinstance(ql_variance, QLVariance)
    assert ql_variance.quantity("4-20 keV").unit == u.ct / (u.s * u.keV)


@pytest.mark.remote_data
def test_ql_variance_plot(ql_variance):
    ql_variance.plot()
    ql_variance.plot(columns=["4-20 keV"])


@pytest.mark.remote_data
def test_hk_maxi(hk_maxi):
    assert isinstance(hk_maxi, HKMaxi)
    assert hk_maxi.quantity("hk_att_c").unit == u.mA


@pytest.mark.remote_data
def test_hk_maxi_plot(hk_maxi):
    hk_maxi.plot()
    hk_maxi.plot(columns=["hk_asp_photoa0_v"])
