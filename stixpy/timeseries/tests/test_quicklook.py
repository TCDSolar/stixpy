import astropy.units as u
import pytest
from sunpy.timeseries import TimeSeries

from stixpy.timeseries.quicklook import HKMaxi, QLBackground, QLLightCurve, QLVariance


@pytest.mark.remote_data
def test_ql_lightcurve():
    ql_lc = TimeSeries(
        r"https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-lightcurve_20200506_V02.fits"
    )
    assert isinstance(ql_lc, QLLightCurve)
    assert ql_lc.quantity('4-10 keV').unit == u.ct/(u.s * u.keV)


@pytest.mark.remote_data
def test_qlbackground():
    ql_bkg = TimeSeries(
        r"https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-background_20200506_V02.fits"
    )
    assert isinstance(ql_bkg, QLBackground)
    assert ql_bkg.quantity('4-10 keV').unit == u.ct / (u.s * u.keV)


@pytest.mark.remote_data
def test_qlvariance():
    ql_var = TimeSeries(
        r"https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-variance_20200506_V02.fits"
    )
    assert isinstance(ql_var, QLVariance)
    assert ql_var.quantity('4-20 keV').unit == u.ct / (u.s * u.keV)


@pytest.mark.remote_data
def test_hk_maxi():
    hk_maxi = TimeSeries(
        "https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/HK/solo_L1_stix-hk-maxi_20200506_V02U.fits"
    )
    assert isinstance(hk_maxi, HKMaxi)
