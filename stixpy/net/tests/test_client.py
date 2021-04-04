import pytest
from sunpy.net import Fido, attrs as a


from stixpy.net.client import STIXClient


@pytest.fixture
def client():
    return STIXClient()


@pytest.mark.remote_data
def test_search_date(client):
    res = client.search(a.Time('2020-05-01T00:00', '2020-05-02T00:00'), a.Instrument.stix)
    assert len(res) == 41

@pytest.mark.remote_data
def test_search_date_product():
    res = Fido.search(a.Time('2020-05-01T00:00', '2020-05-01T23:59'),
                      a.stix.DataProduct.ql_lightcurve, a.Instrument.stix)
    assert len(res) == 1

@pytest.mark.remote_data
def test_search_date_product_xray_level1():
    res = Fido.search(a.Time('2020-05-23T00:00', '2020-05-25T00:00'), a.Instrument.stix,
                      a.stix.DataProduct.sci_xray_l1, a.stix.DataType.sci)
    assert len(res) == 1
