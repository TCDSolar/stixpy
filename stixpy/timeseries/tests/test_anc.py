import astropy.units as u
from sunpy.timeseries import TimeSeries

from stixpy.timeseries import ANCAspect


def test_anc():
    anc = TimeSeries(
        "https://pub099.cs.technik.fhnw.ch/fits/ANC/2025/07/29/ASP/solo_ANC_stix-asp-ephemeris_20250729_V02U.fits"
    )
    assert isinstance(anc, ANCAspect)
    assert anc.quantity("y_srf").unit == u.arcsec
