
from sunpy.net import Fido, attrs as a
from sunpy.time import TimeRange

from stixpy.net.client import STIXClient


def test_search_date():
    res = Fido.search(a.Time('2020-05-01T00:00', '2020-05-02T00:00'), a.Instrument.stix)
    assert len(res) == 1
    stix_res = res.get_response(0)
    assert len(stix_res) == 12


def test_search_date_product():
    res = Fido.search(a.Time('2020-05-01T00:00', '2020-05-01T023:59'),
                      a.stix.DataProduct.ql_lightcurve, a.Instrument.stix)
    assert len(res) == 1
    stix_res = res.get_response(0)
    assert len(stix_res) == 1


def test_search_date_product_xray_level1():
    res = Fido.search(a.Time('2020-05-23T00:00', '2020-05-25T00:00'), a.Instrument.stix,
                      a.stix.DataProduct.sci_xray_l1, a.stix.DataType.sci)
    assert len(res) == 1
    stix_res = res.get_response(0)
    assert len(stix_res) == 12
