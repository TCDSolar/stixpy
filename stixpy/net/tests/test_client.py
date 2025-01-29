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
    """
    TestData dir contains the following files and directories.
    The files are empty but the structure and names are used to test the client.
    .
    ├── ANC
    │   └── 2022
    │       └── 01
    │           ├── 01
    │           │   └── ASP
    │           │       └── solo_ANC_stix-asp-ephemeris_20220101_V02.fits
    │           ├── 02
    │           │   └── ASP
    │           │       └── solo_ANC_stix-asp-ephemeris_20220102_V02.fits
    │           └── 03
    │               └── ASP
    │                   └── solo_ANC_stix-asp-ephemeris_20220103_V02.fits
    ├── L1
    │   └── 2022
    │       └── 01
    │           └── 01
    │               ├── QL
    │               │   ├── solo_L1_stix-ql-lightcurve_20220101_V02.fits
    │               │   ├── solo_L1_stix-ql-lightcurve_20220101_V05.fits
    │               │   ├── solo_L1_stix-ql-lightcurve_20220101_V06U.fits
    │               │   └── solo_L1_stix-ql-spectra_20220101_V02.fits
    │               └── SCI
    │                   ├── solo_L1_stix-sci-xray-cpd_20220101T003000-20220101T013000_V01U_2201010007-54548.fits
    │                   ├── solo_L1_stix-sci-xray-cpd_20220101T003000-20220101T013000_V02_2201010007-54548.fits
    │                   ├── solo_L1_stix-sci-xray-cpd_20220101T003000-20220101T013000_V03U_2201010007-54548.fits
    │                   ├── solo_L1_stix-sci-xray-cpd_20220101T010000-20220101T020000_V01_2201010007-54549.fits
    │                   ├── solo_L1_stix-sci-xray-cpd_20220101T013000-20220101T023000_V04_2201010007-54550.fits
    │                   └── solo_L1_stix-sci-xray-cpd_20220101T020000-20220101T030000_V01_2201010007-54551.fits
    └── test.html
    """
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
        (("2022-01-01T00:00:00", "2022-01-01T03:00:00"), "L1", a.stix.DataType.sci, 6),
        (("2022-01-01T00:00:00", "2022-01-01T01:00:00"), "L1", a.stix.DataType.sci, 4),
        (("2022-01-01T00:30:00", "2022-01-01T01:00:00"), "L1", a.stix.DataType.sci, 4),
        (("2022-01-01T00:35:00", "2022-01-01T00:45:00"), "L1", a.stix.DataType.sci, 3),
        (("2022-01-01T00:00:00", "2022-01-01T00:35:00"), "L1", a.stix.DataType.sci, 3),
        (("2022-01-01T02:45:00", "2022-01-01T05:00:00"), "L1", a.stix.DataType.sci, 1),
        (("2023-01-01T02:45:00", "2023-01-01T05:00:00"), "L1", a.stix.DataType.sci, 0),
        (("2022-01-01T00:00:00", "2022-01-01T03:00:00"), "L1", a.stix.DataType.ql, 4),
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
    assert len(res) == 39
    # this might need fixed when we change to ANC to become an level of its own


def test_search_max_version(clientlocal):
    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, Version=a.stix.MaxVersion(3)
    )
    assert len(res) == 7

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        Version=a.stix.MaxVersion(3, allow_uncompleted=False),
    )
    assert len(res) == 6


def test_search_min_version(clientlocal):
    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, Version=a.stix.MinVersion(2)
    )
    assert len(res) == 8

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        Version=a.stix.MinVersion(2, allow_uncompleted=False),
    )
    assert len(res) == 6


def test_search_latest_version(clientlocal):
    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, Version=a.stix.LatestVersion()
    )
    assert len(res) == 7

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        Version=a.stix.LatestVersion(allow_uncompleted=False),
    )
    assert len(res) == 7

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        a.stix.DataType.ql,
        Version=a.stix.LatestVersion(allow_uncompleted=False),
    )
    res.sort("DataProduct")
    assert len(res) == 2
    assert res["DataProduct"][0] == "ql-lightcurve"
    assert res["Ver"][0] == "V05"
    assert res["DataProduct"][1] == "ql-spectra"
    assert res["Ver"][1] == "V02"

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        a.stix.DataType.ql,
        Version=a.stix.LatestVersion(allow_uncompleted=True),
    )
    res.sort("DataProduct")
    assert len(res) == 2
    assert res["DataProduct"][0] == "ql-lightcurve"
    assert res["Ver"][0] == "V06U"
    assert res["DataProduct"][1] == "ql-spectra"
    assert res["Ver"][1] == "V02"


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
    assert len(res["stix"]) == 52
    # this might need fixed when we change to ANC to become an level of its own

    res_ql = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.ql)
    assert len(res_ql["stix"]) == 6

    res_sci = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.sci)
    assert len(res_sci["stix"]) == 42
