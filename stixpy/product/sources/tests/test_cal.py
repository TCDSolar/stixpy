"""
Tests for :mod:`stixpy.product.sources.cal`.

Unit tests use the local FITS file under ``stixpy/data/`` for the product class.
The file-selection helper ``find_energy_calibration_file_for_time`` is exercised
entirely offline: the ``cal_env`` fixture mocks its four collaborators
(``Fido.search`` / ``Fido.fetch``, ``get_elut``, ``_read_livetime`` and
``Product``) with simple defaults, and each test overrides only the parts it
needs. Only the ``@pytest.mark.remote_data`` smoke tests hit the network.
"""

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import astropy.units as u
from astropy.time import Time

from sunpy.time import TimeRange

from stixpy.data.test import STIX_CAL_ENERGY
from stixpy.product import Product
from stixpy.product.sources import cal
from stixpy.product.sources.cal import (
    CalibrationProduct,
    EnergyCalibration,
    _parse_cal_filename,
    find_energy_calibration_file_for_time,
)

LOCAL_CAL_FITS = STIX_CAL_ENERGY


# ---------------------------------------------------------------------------
# Product class
# ---------------------------------------------------------------------------


def test_energy_calibration_loads_from_local_fits():
    p = Product(LOCAL_CAL_FITS)
    assert isinstance(p, EnergyCalibration)
    assert isinstance(p, CalibrationProduct)


def test_calibration_product_ob_elut_name():
    p = Product(LOCAL_CAL_FITS)
    assert p.ob_elut_name == "elut_table_20241114"


def test_calibration_product_time_range():
    p = Product(LOCAL_CAL_FITS)
    tr = p.time_range
    assert isinstance(tr, TimeRange)
    # File header gives DATE-BEG / DATE-END spanning ~24 h.
    assert (tr.end - tr.start).to_value(u.h) == pytest.approx(24, abs=0.5)


def test_is_datasource_for_dispatch_picks_energycalibration():
    # Synthetic minimal meta: classmethod call doesn't need a real Product.
    meta = {"STYPE": 21, "SSTYPE": 6, "SSID": 41, "level": "CAL"}
    assert EnergyCalibration.is_datasource_for(meta=meta) is True

    # Wrong level
    meta_l1 = {**meta, "level": "L1"}
    assert EnergyCalibration.is_datasource_for(meta=meta_l1) is False

    # Wrong SSID
    meta_other = {**meta, "SSID": 42}
    assert EnergyCalibration.is_datasource_for(meta=meta_other) is False


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------


def test_parse_cal_filename_extracts_start_end():
    name = "solo_CAL_stix-cal-energy_20250302T230511-20250303T230512_V02.fits"
    start, end = _parse_cal_filename(name)
    assert start == Time("2025-03-02T23:05:11")
    assert end == Time("2025-03-03T23:05:12")


def test_parse_cal_filename_raises_on_unmatched_name():
    with pytest.raises(ValueError, match="Not a recognised"):
        _parse_cal_filename("not-a-cal-file.fits")


# ---------------------------------------------------------------------------
# Helper — find_energy_calibration_file_for_time (collaborators mocked)
# ---------------------------------------------------------------------------


def _make_filename(start: str, end: str, ver: int = 1) -> str:
    """Build a CAL filename from ISO-stripped timestamps."""
    return f"solo_CAL_stix-cal-energy_{start}-{end}_V{ver:02d}.fits"


class _Rows(list):
    """``Fido.search(...)["stix"]`` stand-in: row dicts plus the
    ``filter_for_latest_version`` no-op the helper calls."""

    def filter_for_latest_version(self, *args, **kwargs):
        pass


class _Fetched(list):
    """``Fido.fetch(...)`` stand-in: the downloaded paths, with no errors."""

    errors = ()


def _row(path):
    """A search-result row mirroring the columns ``STIXClient`` populates."""
    start, end = _parse_cal_filename(Path(path).name)
    return {"url": str(path), "Start Time": start, "End Time": end}


@pytest.fixture
def cal_env(monkeypatch):
    """
    Mock the four collaborators of ``find_energy_calibration_file_for_time`` and
    return a handle each test tweaks for only what it needs:

    * ``files`` — paths the Fido search "finds".
    * ``elut_name`` — ELUT entry name returned for every time (default constant).
    * ``elut_overrides`` — ``{path: name}`` to return a *different* ELUT entry for
      that file's start time (to simulate an ELUT mismatch).
    * ``livetime_of`` — ``callable(Path) -> seconds`` (default 50 ks, above the
      30 ks threshold).

    Recorders ``searches`` / ``fetches`` / ``livetime_reads`` capture calls for
    count and argument assertions.
    """
    env = SimpleNamespace(
        files=[],
        elut_name="elut_table_20230101",
        elut_overrides={},
        livetime_of=lambda path: 50_000.0,
        searches=[],
        fetches=[],
        livetime_reads=[],
    )

    def fake_search(*args, **kwargs):
        env.searches.append((args, kwargs))
        return {"stix": _Rows(_row(p) for p in env.files)}

    def fake_fetch(query, **kwargs):
        env.fetches.append(query)
        return _Fetched(row["url"] for row in query)

    def fake_get_elut(t):
        t = Time(t)
        for path, name in env.elut_overrides.items():
            file_start = _parse_cal_filename(Path(path).name)[0]
            if abs((t - file_start).sec) < 1e-3:
                return SimpleNamespace(file=f"{name}.csv")
        return SimpleNamespace(file=f"{env.elut_name}.csv")

    def fake_read_livetime(url):
        env.livetime_reads.append(url)
        return env.livetime_of(Path(url)) * u.s

    monkeypatch.setattr(cal, "Fido", SimpleNamespace(search=fake_search, fetch=fake_fetch))
    monkeypatch.setattr(cal, "_read_livetime", fake_read_livetime)
    monkeypatch.setattr("stixpy.calibration.energy.get_elut", fake_get_elut)
    # The post-download sanity check loads the file; return a non-EnergyCalibration
    # object so that block (and its warning, not under test here) is skipped.
    monkeypatch.setattr("stixpy.product.product_factory.Product", lambda path: object())
    return env


def test_find_energy_calibration_no_files_raises(monkeypatch):
    # Empty search result -> the helper raises before touching any other
    # collaborator, so just mock Fido.search to return no rows.
    monkeypatch.setattr(cal, "Fido", SimpleNamespace(search=lambda *args, **kwargs: {"stix": []}))
    with pytest.raises(ValueError, match="No calibration files found"):
        find_energy_calibration_file_for_time(Time("2023-01-02T12:00:00"))


def test_find_energy_calibration_no_elut_match_raises(cal_env):
    # get_elut returns a different entry for the file's start time vs the flare
    # time, so the strategy drops the candidate.
    path = Path(_make_filename("20230101T000000", "20230102T000000"))
    cal_env.files = [path]
    cal_env.elut_overrides = {path: "elut_table_20990101"}
    with pytest.raises(ValueError, match="ELUT mismatch"):
        find_energy_calibration_file_for_time(Time("2023-01-01T12:00:00"))


def test_find_energy_calibration_no_long_enough_files_raises(cal_env):
    # Only a 1-hour file — below the minimum duration.
    cal_env.files = [Path(_make_filename("20230101T000000", "20230101T010000"))]
    with pytest.raises(ValueError, match="duration"):
        find_energy_calibration_file_for_time(Time("2023-01-01T12:00:00"))


def test_find_energy_calibration_picks_closest_mid_time(cal_env):
    cal_env.files = [
        Path(_make_filename("20230101T000000", "20230102T000000")),  # mid 12:00 day 1
        Path(_make_filename("20230102T000000", "20230103T000000")),  # mid 12:00 day 2
        Path(_make_filename("20230103T000000", "20230104T000000")),  # mid 12:00 day 3
    ]
    result = find_energy_calibration_file_for_time(Time("2023-01-02T13:00:00"))
    assert result == cal_env.files[1]


def test_find_energy_calibration_accepts_scalar_time(cal_env):
    cal_env.files = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    result = find_energy_calibration_file_for_time(Time("2023-01-01T12:00:00"))
    assert result == cal_env.files[0]


def test_find_energy_calibration_accepts_time_range(cal_env):
    # Two equally eligible files; the range mid-time breaks the tie towards the
    # file whose mid is closest to the flare midpoint.
    cal_env.files = [
        Path(_make_filename("20230101T000000", "20230102T000000")),  # cal mid 12:00 day 1
        Path(_make_filename("20230102T000000", "20230103T000000")),  # cal mid 12:00 day 2
    ]
    # Flare range mid = 2023-01-02T01:00:00 → closer to day-2 cal.
    flare = [Time("2023-01-02T00:30:00"), Time("2023-01-02T01:30:00")]
    result = find_energy_calibration_file_for_time(flare)
    assert result == cal_env.files[1]


def test_find_energy_calibration_range_uses_start_for_lookup(cal_env):
    # A range crossing midnight: the lookup uses the range start's day.
    cal_env.files = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    flare = [Time("2023-01-01T23:00:00"), Time("2023-01-02T01:00:00")]
    result = find_energy_calibration_file_for_time(flare)
    assert result == cal_env.files[0]


# ---------------------------------------------------------------------------
# LIVETIME walk + extended-window tests
# ---------------------------------------------------------------------------


def test_find_energy_calibration_livetime_pass_top_candidate(cal_env):
    """Top-ranked candidate has plenty of LIVETIME -> returned immediately, with
    only one Fido.fetch and one LIVETIME read."""
    cal_env.files = [
        Path(_make_filename("20230101T000000", "20230102T000000")),
        Path(_make_filename("20230102T000000", "20230103T000000")),
    ]
    # Flare mid = noon Jan 2 → closest to files[1] (mid Jan 2 noon).
    result = find_energy_calibration_file_for_time(Time("2023-01-02T12:00:00"))
    assert result == cal_env.files[1]
    assert len(cal_env.fetches) == 1  # only the winner downloaded
    assert len(cal_env.livetime_reads) == 1  # only the winner's header peeked


def test_find_energy_calibration_livetime_walks_back_on_fail(cal_env):
    """Top-ranked candidate fails LIVETIME, next-ranked passes."""
    cal_env.files = [
        Path(_make_filename("20230101T000000", "20230102T000000")),  # mid Jan 1 noon
        Path(_make_filename("20230102T000000", "20230103T000000")),  # mid Jan 2 noon
    ]
    # Closest to noon Jan 2 = files[1]. Make it fail; files[0] passes.
    lifetimes = {cal_env.files[1]: 1_000.0, cal_env.files[0]: 50_000.0}
    cal_env.livetime_of = lambda path: lifetimes[path]
    result = find_energy_calibration_file_for_time(Time("2023-01-02T13:00:00"))
    assert result == cal_env.files[0]


def test_find_energy_calibration_livetime_all_fail_raises(cal_env):
    cal_env.files = [
        Path(_make_filename("20230101T000000", "20230102T000000")),
        Path(_make_filename("20230102T000000", "20230103T000000")),
    ]
    cal_env.livetime_of = lambda path: 100.0  # 100 s << 30 ks default
    with pytest.raises(ValueError, match="LIVETIME threshold"):
        find_energy_calibration_file_for_time(Time("2023-01-02T12:00:00"))


def test_find_energy_calibration_livetime_custom_threshold(cal_env):
    """min_livetime override rejects an otherwise-default-passing file."""
    cal_env.files = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    cal_env.livetime_of = lambda path: 40_000.0  # passes the 30 ks default
    # Bump the bar above 40 ks → no candidates remain.
    with pytest.raises(ValueError, match="LIVETIME threshold"):
        find_energy_calibration_file_for_time(Time("2023-01-01T12:00:00"), min_livetime=50_000 * u.s)


def test_find_energy_calibration_window_uses_module_defaults(cal_env):
    """The Fido query window matches the module-level default window."""
    cal_env.files = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    find_energy_calibration_file_for_time(Time("2023-01-15T12:00:00"))
    # The first positional search arg is a.Time(...) with .start / .end.
    args, _ = cal_env.searches[0]
    time_attr = args[0]
    day_midnight = Time("2023-01-15T00:00:00")
    assert Time(time_attr.start).isclose(day_midnight - cal._DEFAULT_WINDOW_PAST)
    # Closing bound = day + window_future + 1 day (exclusive).
    assert Time(time_attr.end).isclose(day_midnight + cal._DEFAULT_WINDOW_FUTURE + 1 * u.day)


def test_find_energy_calibration_downloads_only_winner(cal_env):
    """The LIVETIME walk inspects candidates in rank order and downloads only the
    winner: top-ranked fails, second-ranked passes -> two header reads, one fetch."""
    cal_env.files = [
        Path(_make_filename("20230101T000000", "20230102T000000")),
        Path(_make_filename("20230102T000000", "20230103T000000")),
    ]
    # First-ranked (closest to noon Jan 2) fails, second-ranked passes.
    lifetimes = {cal_env.files[1]: 100.0, cal_env.files[0]: 50_000.0}
    cal_env.livetime_of = lambda path: lifetimes[path]
    find_energy_calibration_file_for_time(Time("2023-01-02T13:00:00"))
    assert len(cal_env.livetime_reads) == 2  # one per inspected candidate
    assert len(cal_env.fetches) == 1  # only the winning candidate downloaded


def test_read_livetime_uses_fsspec_range_read(monkeypatch):
    """``_read_livetime`` opens the file with ``use_fsspec=True`` (an HTTP Range
    read of the header) and returns LIVETIME in seconds."""
    captured = {}

    def fake_open(url, *args, **kwargs):
        captured["kwargs"] = kwargs
        cm = MagicMock()
        cm.__enter__.return_value = [MagicMock(header={"LIVETIME": 42.0})]
        return cm

    monkeypatch.setattr(cal.fits, "open", fake_open)
    livetime = cal._read_livetime("https://example/cal.fits")
    assert livetime == 42.0 * u.s
    assert captured["kwargs"].get("use_fsspec") is True


# ---------------------------------------------------------------------------
# Remote-data smoke test
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_find_energy_calibration_picks_2024_04_01_for_2024_03_28_flare():
    """
    Real-archive smoke: at the 2024-03-28T06:26:45 flare, the closer-in-time
    cal files happen to fall below the LIVETIME threshold, so the LIVETIME
    walk must fall back to the longer (good live-time) file dated
    2024-03-31 → 2024-04-01.
    """
    path = find_energy_calibration_file_for_time(Time("2024-03-28T06:26:45"))
    # The selected file's filename-derived END time should land on 2024-04-01.
    _start, end = _parse_cal_filename(path.name)
    assert end.utc.iso.startswith("2024-04-01"), f"Expected the 2024-04-01-ending cal file, got {path.name}"


@pytest.mark.remote_data
def test_fido_finds_and_loads_cal_file():
    """End-to-end: Fido search → download → load → product props are sane."""
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    res = Fido.search(
        a.Time("2021-10-09T00:00:00", "2021-10-09T23:59:59"),
        a.Instrument.stix,
        a.Level("CAL"),
        a.stix.DataType.cal,
        a.stix.DataProduct.cal_energy,
    )
    assert len(res["stix"]) >= 1

    res["stix"].filter_for_latest_version()
    files = Fido.fetch(res["stix"])
    assert len(files) >= 1
    assert len(files.errors) == 0

    cal = Product(files[0])
    assert isinstance(cal, EnergyCalibration)
    # Daily CAL files cover roughly 24 hours of acquisition; tolerate a
    # ~1 h margin for files near operational transitions.
    duration_h = (cal.time_range.end - cal.time_range.start).to_value(u.h)
    assert duration_h == pytest.approx(24, abs=1)
    # ELUT name is the on-board ELUT in use during recording.
    assert cal.ob_elut_name.startswith("elut_table_")
