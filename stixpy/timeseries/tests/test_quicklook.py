from pathlib import Path

import pytest
from sunpy.timeseries import TimeSeries

from stixpy.data import test
from stixpy.timeseries.quicklook import *


@pytest.mark.remote_data
def test_ql_lightcurve():
    ql_lc = TimeSeries(
        r'https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-lightcurve_20200506_V02.fits')
    assert isinstance(ql_lc, QLLightCurve)


@pytest.mark.remote_data
def test_qlbackground():
    ql_lc = TimeSeries(
        r'https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-background_20200506_V02.fits')
    assert isinstance(ql_lc, QLBackground)


@pytest.mark.remote_data
def test_qlvariance():
    ql_lc = TimeSeries(
        r'https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-variance_20200506_V02.fits')
    assert isinstance(ql_lc, QLVariance)


@pytest.mark.remote_data
def test_hk_maxi():
    hk_maxi = TimeSeries(
        'https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/HK/solo_L1_stix-hk-maxi_20200506_V02U.fits')
    assert isinstance(hk_maxi, HKMaxi)
