from sunpy.timeseries import TimeSeries
from stixpy.timeseries.quicklook import QLLightCurve, QLBackground, QLVariance


def test_qllightcurve():
    ql_lc = TimeSeries('../../data/solo_L1_stix-ql-lightcurve_20200607_V01.fits')
    assert isinstance(ql_lc, QLLightCurve)
    assert len(ql_lc.data) == 21600


def test_qlbackground():
    return False


def test_qlvariance():
    return False
