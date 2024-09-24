import os
from pathlib import Path
from unittest import mock

import pytest
from sunpy.net import Fido
from sunpy.net import attrs as a

from stixpy.net.client import STIXClient

MOCK_PATH = "sunpy.net.scraper.urlopen"


@pytest.fixture
def client():
    return STIXClient()


@pytest.fixture
def clientlocal():
    return STIXClient(source=f'file://{Path(__file__).parent / "data"}{os.sep}')


@pytest.fixture
def http_response():
    path = Path(__file__).parent / "data" / "test.html"
    with path.open("r") as f:
        response_html = f.read()

    return response_html


@pytest.mark.parametrize(
    "time_range, nfiles",
    [
        (("2022-01-01T00:00:00", "2022-01-01T03:00:00"), 4),
        (("2022-01-01T00:00:00", "2022-01-01T01:00:00"), 2),
        (("2022-01-01T00:30:00", "2022-01-01T01:00:00"), 2),
        (("2022-01-01T00:35:00", "2022-01-01T00:45:00"), 1),
        (("2022-01-01T00:00:00", "2022-01-01T00:35:00"), 1),
        (("2022-01-01T02:45:00", "2022-01-01T05:00:00"), 1),
        (("2023-01-01T02:45:00", "2023-01-01T05:00:00"), 0),
    ],
)
@mock.patch(MOCK_PATH)
def test_client(urlopen, client, http_response, time_range, nfiles):
    urlopen.return_value.read = mock.MagicMock()
    urlopen.return_value.read.side_effect = [http_response] * 5
    urlopen.close = mock.MagicMock(return_value=None)
    query = client.search(a.Time(*time_range), a.Instrument.stix, a.Level("L1"), a.stix.DataType.sci)
    assert len(query) == nfiles


@pytest.mark.parametrize(
    "time_range, level, dtype, nfiles",
    [
        (("2022-01-01T00:00:00", "2022-01-01T03:00:00"), "L1", a.stix.DataType.sci, 4),
        (("2022-01-01T00:00:00", "2022-01-01T01:00:00"), "L1", a.stix.DataType.sci, 2),
        (("2022-01-01T00:30:00", "2022-01-01T01:00:00"), "L1", a.stix.DataType.sci, 2),
        (("2022-01-01T00:35:00", "2022-01-01T00:45:00"), "L1", a.stix.DataType.sci, 1),
        (("2022-01-01T00:00:00", "2022-01-01T00:35:00"), "L1", a.stix.DataType.sci, 1),
        (("2022-01-01T02:45:00", "2022-01-01T05:00:00"), "L1", a.stix.DataType.sci, 1),
        (("2023-01-01T02:45:00", "2023-01-01T05:00:00"), "L1", a.stix.DataType.sci, 0),
        (("2022-01-01T00:00:00", "2022-01-01T03:00:00"), "L1", a.stix.DataType.ql, 2),
        (("2022-01-01T00:00:00", "2022-01-01T03:00:00"), "ANC", a.stix.DataType.asp, 1),
        (("2022-01-01T00:00:00", "2022-01-02T03:00:00"), "ANC", a.stix.DataType.asp, 2),
        (("2022-01-01T00:00:00", "2022-01-03T03:00:00"), "ANC", a.stix.DataType.asp, 3),
        (("2022-01-01T00:00:00", "2022-01-05T03:00:00"), "ANC", a.stix.DataType.asp, 3),
    ],
)
def test_local_client(clientlocal, time_range, level, dtype, nfiles):
    query = clientlocal.search(
        a.Time(*time_range),
        a.Instrument.stix,
        a.Level(level),
        dtype,
    )
    assert len(query) == nfiles


@pytest.mark.remote_data
def test_search_date(client):
    res = client.search(a.Time("2020-05-01T00:00", "2020-05-01T23:59"), a.Instrument.stix)
    assert len(res) == 37
    # this might need fixed when we change to ANC to become an level of its own


@pytest.mark.remote_data
def test_search_date_product():
    res1 = Fido.search(
        a.Time("2020-05-01T00:00", "2020-05-01T23:59"), a.stix.DataProduct.ql_lightcurve, a.Instrument.stix
    )
    assert len(res1[0]) == 1

    res2 = Fido.search(
        a.Time("2020-05-01T00:00", "2020-05-02T00:00"), a.stix.DataProduct.ql_lightcurve, a.Instrument.stix
    )
    assert len(res2[0]) == 2

    res3 = Fido.search(a.Time("2022-01-20 05:40", "2022-01-20 06:20"), a.Instrument("STIX"), a.stix.DataType("sci"))
    assert len(res3[0]) == 18


@pytest.mark.remote_data
def test_search_date_product_xray_level1():
    res = Fido.search(
        a.Time("2020-05-23T00:00", "2020-05-25T00:00"),
        a.Instrument.stix,
        a.stix.DataProduct.sci_xray_cpd,
        a.stix.DataType.sci,
    )
    assert len(res) == 1


@pytest.mark.remote_data
def test_search_date_product_sci():
    res = Fido.search(a.Time("2021-05-07T00:00", "2021-05-08T00:00"), a.Instrument.stix, a.stix.DataType.sci)
    assert len(res) == 1


@pytest.mark.remote_data
def test_fido():
    res = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix)
    assert len(res["stix"]) == 50
    # this might need fixed when we change to ANC to become an level of its own

    res_ql = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.ql)
    assert len(res_ql["stix"]) == 6

    res_sci = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.sci)
    assert len(res_sci["stix"]) == 42
