import pytest
from sunpy.net import Fido, attrs as a


from stixpy.net.client import STIXClient


@pytest.fixture
def client():
    return STIXClient()


@pytest.mark.remote_data
def test_search_date(client):
    res = client.search(a.Time('2020-05-01T00:00', '2020-05-01T23:59'), a.Instrument.stix)
    assert len(res) == 37


@pytest.mark.remote_data
def test_search_date_product():
    res1 = Fido.search(a.Time('2020-05-01T00:00', '2020-05-01T23:59'),
                      a.stix.DataProduct.ql_lightcurve, a.Instrument.stix)
    assert len(res1[0]) == 1

    res2 = Fido.search(a.Time('2020-05-01T00:00', '2020-05-02T00:00'),
                      a.stix.DataProduct.ql_lightcurve, a.Instrument.stix)
    assert len(res2[0]) == 2

    res3 = Fido.search(a.Time("2022-01-20 05:40", "2022-01-20 06:20"), a.Instrument("STIX"),
                      a.stix.DataType("sci"))
    assert len(res3[0]) == 9


@pytest.mark.remote_data
def test_search_date_product_xray_level1():
    res = Fido.search(a.Time('2020-05-23T00:00', '2020-05-25T00:00'), a.Instrument.stix,
                      a.stix.DataProduct.sci_xray_l1, a.stix.DataType.sci)
    assert len(res) == 1


@pytest.mark.remote_data
def test_search_date_product_xray_level1():
    res = Fido.search(a.Time('2021-05-07T00:00', '2021-05-08T00:00'), a.Instrument.stix,
                      a.stix.DataType.sci)
    assert len(res) == 1


@pytest.mark.remote_data
def test_fido():
    res = Fido.search(a.Time('2020-11-17T00:00', '2020-11-17T23:59'), a.Instrument.stix)
    assert len(res['stix']) == 50

    res_ql = Fido.search(a.Time('2020-11-17T00:00', '2020-11-17T23:59'), a.Instrument.stix,
                         a.stix.DataType.ql)
    assert len(res_ql['stix']) == 5

    res_sci = Fido.search(a.Time('2020-11-17T00:00', '2020-11-17T23:59'), a.Instrument.stix,
                          a.stix.DataType.sci)
    assert len(res_sci['stix']) == 42
