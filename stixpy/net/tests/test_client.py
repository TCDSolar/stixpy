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
    return STIXClient(source=f"file://{Path(__file__).parent / 'data'}")


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
    urlopen.return_value.read.side_effect = ["", http_response] * 5
    urlopen.close = mock.MagicMock(return_value=None)
    query = client.search(a.Time(*time_range), a.Instrument.stix, a.Level("L1"), a.stix.DataType.sci)
    assert len(query) == nfiles


@pytest.mark.skipif(os.name == "nt", reason="Upstream sunpy bug on windows")
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
    assert len(res) == 65
    # this might need fixed when we change to ANC to become an level of its own


@pytest.mark.skipif(os.name == "nt", reason="Upstream sunpy bug on windows")
def test_search_max_version(clientlocal):
    res = clientlocal.search(a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, a.stix.MaxVersion(3))
    assert len(res) == 6

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        a.stix.MaxVersionU(3),
    )
    assert len(res) == 8


@pytest.mark.skipif(os.name == "nt", reason="Upstream sunpy bug on windows")
def test_search_min_version(clientlocal):
    res = clientlocal.search(a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, a.stix.MinVersion(2))
    assert len(res) == 6

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        a.stix.MinVersionU(2),
    )
    assert len(res) == 8


@pytest.mark.skipif(os.name == "nt", reason="Upstream sunpy bug on windows")
def test_search_version(clientlocal):
    res = clientlocal.search(a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, a.stix.Version(2))
    assert len(res) == 4

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
        a.stix.VersionU(1),
    )
    assert len(res) == 3


@pytest.mark.skipif(os.name == "nt", reason="Upstream sunpy bug on windows")
def test_search_version_and(clientlocal):
    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, a.stix.MinVersion(2), a.stix.MaxVersionU(5)
    )
    assert len(res) == 6


@pytest.mark.skipif(os.name == "nt", reason="Upstream sunpy bug on windows")
def test_search_latest_version_empty(clientlocal):
    res = clientlocal.search(a.Time("2023-01-01T00:00", "2023-01-01T23:59"), a.Instrument.stix)
    res.filter_for_latest_version()
    assert len(res) == 0


@pytest.mark.skipif(os.name == "nt", reason="Upstream but on windows")
def test_search_latest_version(clientlocal):
    res = clientlocal.search(a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix)
    res.filter_for_latest_version(allow_uncompleted=True)
    assert len(res) == 7

    res = clientlocal.search(
        a.Time("2022-01-01T00:00", "2022-01-01T23:59"),
        a.Instrument.stix,
    )
    res.filter_for_latest_version(allow_uncompleted=False)
    assert len(res) == 7

    res = clientlocal.search(a.Time("2022-01-01T00:00", "2022-01-01T23:59"), a.Instrument.stix, a.stix.DataType.ql)
    res.filter_for_latest_version(allow_uncompleted=False)
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
    )
    res.filter_for_latest_version(allow_uncompleted=True)
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
    assert len(res3[0]) == 24


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
    len_total = len(res["stix"])
    assert len_total == 68

    res_ql = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.ql)
    len_ql = len(res_ql["stix"])
    assert len_ql == 6

    res_sci = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.sci)
    len_sci = len(res_sci["stix"])
    assert len_sci == 58

    res_hk = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.hk)
    len_kh = len(res_hk["stix"])
    assert len_kh == 1

    res_asp = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.asp)
    len_asp = len(res_asp["stix"])
    assert len_asp == 1

    res_cal = Fido.search(a.Time("2020-11-17T00:00", "2020-11-17T23:59"), a.Instrument.stix, a.stix.DataType.cal)
    len_cal = len(res_cal["stix"])
    assert len_cal == 2

    assert len_ql + len_sci + len_kh + len_asp + len_cal == len_total


@pytest.mark.remote_data
def test_fido_file_crosses_date_boundary():
    q = Fido.search(a.Time("2023-06-01T00:00", "2023-06-01T00:30"), a.Instrument.stix, a.stix.DataProduct.sci_xray_spec)
    assert len(q["stix"]) == 2


def test_version_attributes_import():
    from stixpy.net.attrs import MaxVersion, MaxVersionU, MinVersion, MinVersionU, Version, VersionU

    assert Version is not None
    assert VersionU is not None
    assert MinVersion is not None
    assert MinVersionU is not None
    assert MaxVersion is not None
    assert MaxVersionU is not None


def test_version_attributes_functionality():
    from stixpy.net.attrs import MaxVersion, MaxVersionU, MinVersion, MinVersionU, Version, VersionU

    version_attr = Version(2)
    assert version_attr.matches("V02")
    assert not version_attr.matches("V03")
    assert not version_attr.matches("V02U")

    version_u_attr = VersionU(2)
    assert version_u_attr.matches("V02")
    assert version_u_attr.matches("V02U")
    assert not version_u_attr.matches("V03")

    min_version_attr = MinVersion(2)
    assert min_version_attr.matches("V02")
    assert min_version_attr.matches("V03")
    assert not min_version_attr.matches("V01")

    max_version_attr = MaxVersion(3)
    assert max_version_attr.matches("V02")
    assert max_version_attr.matches("V03")
    assert not max_version_attr.matches("V04")

    min_version_u_attr = MinVersionU(2)
    assert min_version_u_attr.matches("V02U")
    assert min_version_u_attr.matches("V03U")

    max_version_u_attr = MaxVersionU(3)
    assert max_version_u_attr.matches("V02U")
    assert max_version_u_attr.matches("V03U")
    assert not max_version_u_attr.matches("V04U")
