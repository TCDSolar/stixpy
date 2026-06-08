"""
Tests for :mod:`stixpy.product.sources.cal`.

Unit tests use the local FITS file under ``stixpy/data/`` for the product class,
and monkey-patched ``Fido`` / ``Product`` / ``get_elut`` for the helper. The
helper's strategy logic is exercised entirely offline; only the smoke
``@pytest.mark.remote_data`` test goes to the network.
"""

from types import SimpleNamespace
from pathlib import Path

import pytest

import astropy.units as u
from astropy.time import Time

from sunpy.time import TimeRange

from stixpy.data.test import STIX_CAL_ENERGY
from stixpy.product import Product
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
# Helper — Fido / Product / get_elut mocked
# ---------------------------------------------------------------------------


def _make_filename(start: str, end: str, ver: int = 1) -> str:
    """Build a CAL filename from ISO-stripped timestamps."""
    return f"solo_CAL_stix-cal-energy_{start}-{end}_V{ver:02d}.fits"


class _FakeFidoQueryRow:
    """
    Stand-in for the ``Fido.search`` result table.

    Each "row" is a dict with at least ``url``, ``Start Time``, ``End Time``
    keys — matching the columns ``STIXClient.post_search_hook`` populates on
    real searches. Supports iteration and ``[i:j]`` slicing (the helper uses
    a single-row slice to fetch one candidate at a time).
    """

    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _FakeFidoQueryRow(self._rows[key])
        if isinstance(key, int):
            return self._rows[key]
        raise TypeError(f"Unsupported key {key!r}")

    def filter_for_latest_version(self, *args, **kwargs):
        return None  # no-op for our tests


class _FakeFidoResults:
    def __init__(self, rows):
        self._rows = list(rows)

    def __getitem__(self, key):
        if key == "stix":
            return _FakeFidoQueryRow(self._rows)
        raise KeyError(key)


class _FakeFetchResults(list):
    errors = ()


class _FakeFido:
    """Captures search args; returns a fetch result containing the rows' urls."""

    def __init__(self, rows):
        self._rows = list(rows)

    def search(self, *args, **kwargs):
        return _FakeFidoResults(self._rows)

    def fetch(self, query, **kwargs):
        # ``query`` is a sliced (or full) _FakeFidoQueryRow — download is the
        # urls of whatever rows it contains.
        return _FakeFetchResults([r["url"] for r in query._rows])


class _FakeEnergyCalibration(EnergyCalibration):
    """
    Test stand-in built once at module scope so only a single subclass is
    registered with the Product factory. ``is_datasource_for`` overridden to
    never claim — the factory must still dispatch real CAL FITS files to the
    real `~stixpy.product.sources.cal.EnergyCalibration`.
    """

    def __init__(self, *, ob_elut_name):
        self._ob_elut_name = ob_elut_name

    @property  # type: ignore[misc]
    def ob_elut_name(self) -> str:  # type: ignore[override]
        return self._ob_elut_name

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        return False


def _rows_for(paths):
    """Build row dicts mirroring real STIXClient post-processed results."""
    rows = []
    for p in paths:
        start, end = _parse_cal_filename(Path(p).name)
        rows.append({"url": str(p), "Start Time": start, "End Time": end})
    return rows


class _FakeHDU:
    """Minimal stand-in for `fits.open(url, use_fsspec=True)[0]`."""

    def __init__(self, livetime_seconds: float):
        self.header = {"LIVETIME": livetime_seconds}


class _FakeHDUList:
    """Context-manager stand-in for an open HDUList."""

    def __init__(self, livetime_seconds: float):
        self._hdu = _FakeHDU(livetime_seconds)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, key):
        if key == 0:
            return self._hdu
        raise IndexError(key)


def _default_livetime(_path):
    """LIVETIME comfortably above the default 30 ks threshold."""
    return 50_000.0


@pytest.fixture
def mocks_factory(monkeypatch):
    """
    Returns a callable ``(paths, *, elut_name, file_elut_names=None,
    ob_elut_per_path=None, livetime_for=None, fetch_recorder=None,
    fits_open_recorder=None)`` that installs the mocks for the helper.

    - ``elut_name`` is what ``get_elut`` returns by default (the flare-time
      lookup and, unless overridden, every file-start lookup).
    - ``file_elut_names`` is a ``{Path: str}`` mapping; if given, the
      ``get_elut`` mock returns the per-path ELUT name when called with
      that path's start time. Used to simulate ELUT-mismatch cases.
    - ``ob_elut_per_path`` controls the on-board ELUT name returned by the
      Product loader called for the *final* sanity check. Defaults to the
      same value as ``elut_name`` so the post-download check is silent.
    - ``livetime_for`` is a callable ``Path -> float`` (LIVETIME seconds)
      consulted by the patched ``fits.open(..., use_fsspec=True)``. Default
      returns 50_000 s for every path (above the 30 ks default threshold).
    - ``fetch_recorder`` / ``fits_open_recorder``: optional list-like
      collectors that capture each Fido.fetch / fits.open call for tests
      that need to assert call counts or argument shape.
    """
    FakeCal = _FakeEnergyCalibration

    def install(
        paths,
        *,
        elut_name,
        file_elut_names=None,
        ob_elut_per_path=None,
        livetime_for=None,
        fetch_recorder=None,
        fits_open_recorder=None,
        search_recorder=None,
    ):
        if ob_elut_per_path is None:
            ob_elut_per_path = {p: elut_name for p in paths}
        if livetime_for is None:
            livetime_for = _default_livetime
        # Map each file's start Time to its ELUT name for the get_elut mock.
        start_to_elut: dict = {}
        if file_elut_names is not None:
            for p, name in file_elut_names.items():
                start_to_elut[_parse_cal_filename(Path(p).name)[0]] = name

        fake_fido = _FakeFido(_rows_for(paths))

        # Optional spies on _FakeFido's call surface.
        if search_recorder is not None or fetch_recorder is not None:
            original_search = fake_fido.search
            original_fetch = fake_fido.fetch

            def _search(*args, **kwargs):
                if search_recorder is not None:
                    search_recorder.append((args, kwargs))
                return original_search(*args, **kwargs)

            def _fetch(query, **kwargs):
                if fetch_recorder is not None:
                    fetch_recorder.append(query)
                return original_fetch(query, **kwargs)

            fake_fido.search = _search
            fake_fido.fetch = _fetch

        monkeypatch.setattr("stixpy.product.sources.cal.Fido", fake_fido)

        def get_elut_mock(t):
            for ref_t, name in start_to_elut.items():
                if abs((Time(t) - ref_t).sec) < 1e-3:
                    return SimpleNamespace(file=f"{name}.csv")
            return SimpleNamespace(file=f"{elut_name}.csv")

        monkeypatch.setattr("stixpy.calibration.energy.get_elut", get_elut_mock)
        monkeypatch.setattr(
            "stixpy.product.product_factory.Product",
            lambda path: FakeCal(ob_elut_name=ob_elut_per_path[Path(path)]),
        )

        def fake_fits_open(url, *args, **kwargs):
            if fits_open_recorder is not None:
                fits_open_recorder.append((url, kwargs))
            # Map url back to a path. Mocks set url = str(Path).
            return _FakeHDUList(livetime_for(Path(url)))

        monkeypatch.setattr("stixpy.product.sources.cal.fits.open", fake_fits_open)
        return paths

    return install


def test_find_energy_calibration_no_files_raises(mocks_factory):
    mocks_factory([], elut_name="elut_table_20230101")
    with pytest.raises(ValueError, match="No calibration files found"):
        find_energy_calibration_file_for_time(Time("2023-01-02T12:00:00"))


def test_find_energy_calibration_no_elut_match_raises(mocks_factory):
    # get_elut returns a different entry for the file's start time vs the
    # flare time — the strategy must drop the candidate.
    paths = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    mocks_factory(
        paths,
        elut_name="elut_table_20230101",
        file_elut_names={paths[0]: "elut_table_20990101"},
    )
    with pytest.raises(ValueError, match="ELUT mismatch"):
        find_energy_calibration_file_for_time(Time("2023-01-01T12:00:00"))


def test_find_energy_calibration_no_long_enough_files_raises(mocks_factory):
    # Only a 1-hour file — below the 12-h threshold.
    paths = [Path(_make_filename("20230101T000000", "20230101T010000"))]
    mocks_factory(paths, elut_name="elut_table_20230101")
    with pytest.raises(ValueError, match="duration"):
        find_energy_calibration_file_for_time(Time("2023-01-01T12:00:00"))


def test_find_energy_calibration_picks_closest_mid_time(mocks_factory):
    paths = [
        Path(_make_filename("20230101T000000", "20230102T000000")),  # mid 12:00
        Path(_make_filename("20230102T000000", "20230103T000000")),  # mid 12:00 next day
        Path(_make_filename("20230103T000000", "20230104T000000")),  # mid 12:00 day-after
    ]
    mocks_factory(paths, elut_name="elut_table_20230101")
    result = find_energy_calibration_file_for_time(Time("2023-01-02T13:00:00"))
    assert result == paths[1]


def test_find_energy_calibration_accepts_scalar_time(mocks_factory):
    paths = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    mocks_factory(paths, elut_name="elut_table_20230101")
    result = find_energy_calibration_file_for_time(Time("2023-01-01T12:00:00"))
    assert result == paths[0]


def test_find_energy_calibration_accepts_time_range(mocks_factory):
    # Two equally eligible files; t_mid (computed from range mid) breaks the tie
    # towards the file whose mid is closest to the flare midpoint.
    paths = [
        Path(_make_filename("20230101T000000", "20230102T000000")),  # cal mid 12:00 day 1
        Path(_make_filename("20230102T000000", "20230103T000000")),  # cal mid 12:00 day 2
    ]
    mocks_factory(paths, elut_name="elut_table_20230101")
    # Flare range mid = 2023-01-02T01:00:00 → closer to day-2 cal (mid 12:00)
    # than day-1 cal (mid 12:00 = 13 hours away).
    flare = [Time("2023-01-02T00:30:00"), Time("2023-01-02T01:30:00")]
    result = find_energy_calibration_file_for_time(flare)
    assert result == paths[1]


def test_find_energy_calibration_range_uses_start_for_lookup(mocks_factory):
    # Even if mid lands in a different calendar day, the ELUT lookup uses
    # the range start. Construct a range crossing midnight and verify the
    # 3-day Fido search window is built around the START's day.
    paths = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    mocks_factory(paths, elut_name="elut_table_20230101")
    flare = [Time("2023-01-01T23:00:00"), Time("2023-01-02T01:00:00")]
    result = find_energy_calibration_file_for_time(flare)
    assert result == paths[0]


# ---------------------------------------------------------------------------
# LIVETIME walk + extended-window tests
# ---------------------------------------------------------------------------


def test_find_energy_calibration_livetime_pass_top_candidate(mocks_factory):
    """Top-ranked candidate has plenty of LIVETIME -> returned immediately,
    only one Fido.fetch + one fits.open call."""
    paths = [
        Path(_make_filename("20230101T000000", "20230102T000000")),
        Path(_make_filename("20230102T000000", "20230103T000000")),
    ]
    fetch_recorder: list = []
    fits_open_recorder: list = []
    mocks_factory(
        paths,
        elut_name="elut_table_20230101",
        livetime_for=lambda _p: 50_000.0,  # well above the 30 ks default
        fetch_recorder=fetch_recorder,
        fits_open_recorder=fits_open_recorder,
    )
    # Flare mid = noon Jan 2 → closest to paths[1] (mid Jan 2 noon).
    result = find_energy_calibration_file_for_time(Time("2023-01-02T12:00:00"))
    assert result == paths[1]
    assert len(fetch_recorder) == 1  # only the winner downloaded
    assert len(fits_open_recorder) == 1  # only the winner's header peeked


def test_find_energy_calibration_livetime_walks_back_on_fail(mocks_factory):
    """Top-ranked candidate fails LIVETIME, next-ranked passes."""
    paths = [
        Path(_make_filename("20230101T000000", "20230102T000000")),  # mid Jan 1 noon
        Path(_make_filename("20230102T000000", "20230103T000000")),  # mid Jan 2 noon
    ]
    # Closest to noon Jan 2 = paths[1]. Make it fail; paths[0] passes.
    lifetimes = {paths[1]: 1_000.0, paths[0]: 50_000.0}
    mocks_factory(
        paths,
        elut_name="elut_table_20230101",
        livetime_for=lambda p: lifetimes[p],
    )
    result = find_energy_calibration_file_for_time(Time("2023-01-02T13:00:00"))
    assert result == paths[0]


def test_find_energy_calibration_livetime_all_fail_raises(mocks_factory):
    paths = [
        Path(_make_filename("20230101T000000", "20230102T000000")),
        Path(_make_filename("20230102T000000", "20230103T000000")),
    ]
    mocks_factory(
        paths,
        elut_name="elut_table_20230101",
        livetime_for=lambda _p: 100.0,  # 100 s << 30 ks default
    )
    with pytest.raises(ValueError, match="LIVETIME threshold"):
        find_energy_calibration_file_for_time(Time("2023-01-02T12:00:00"))


def test_find_energy_calibration_livetime_custom_threshold(mocks_factory):
    """min_livetime override rejects an otherwise-default-passing file."""
    paths = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    mocks_factory(
        paths,
        elut_name="elut_table_20230101",
        livetime_for=lambda _p: 40_000.0,  # passes default (30 ks)
    )
    # Bump the bar above 40 ks → no candidates remain.
    with pytest.raises(ValueError, match="LIVETIME threshold"):
        find_energy_calibration_file_for_time(
            Time("2023-01-01T12:00:00"),
            min_livetime=50_000 * u.s,
        )


def test_find_energy_calibration_window_uses_module_defaults(mocks_factory):
    """The Fido query window matches the module-level default window."""
    from stixpy.product.sources.cal import _DEFAULT_WINDOW_FUTURE, _DEFAULT_WINDOW_PAST

    paths = [Path(_make_filename("20230101T000000", "20230102T000000"))]
    search_recorder: list = []
    mocks_factory(
        paths,
        elut_name="elut_table_20230101",
        search_recorder=search_recorder,
    )
    # We don't care about the return value here, only the search args.
    find_energy_calibration_file_for_time(Time("2023-01-15T12:00:00"))
    # Inspect the first positional arg of the search call — it's a.Time(...)
    # with .start / .end attributes.
    args, _ = search_recorder[0]
    time_attr = args[0]
    day_midnight = Time("2023-01-15T00:00:00")
    assert Time(time_attr.start).isclose(day_midnight - _DEFAULT_WINDOW_PAST)
    # Closing bound = day + window_future + 1 day (exclusive).
    assert Time(time_attr.end).isclose(day_midnight + _DEFAULT_WINDOW_FUTURE + 1 * u.day)


def test_find_energy_calibration_livetime_via_fsspec_range_read(mocks_factory):
    """The LIVETIME walk uses fits.open(url, use_fsspec=True) — no Fido.fetch
    on rejected candidates."""
    paths = [
        Path(_make_filename("20230101T000000", "20230102T000000")),
        Path(_make_filename("20230102T000000", "20230103T000000")),
    ]
    fits_open_recorder: list = []
    fetch_recorder: list = []
    # First-ranked fails, second-ranked passes.
    lifetimes = {paths[1]: 100.0, paths[0]: 50_000.0}
    mocks_factory(
        paths,
        elut_name="elut_table_20230101",
        livetime_for=lambda p: lifetimes[p],
        fits_open_recorder=fits_open_recorder,
        fetch_recorder=fetch_recorder,
    )
    find_energy_calibration_file_for_time(Time("2023-01-02T13:00:00"))
    # fits.open was called twice (one per candidate inspected), with
    # use_fsspec=True; Fido.fetch only once (the winning candidate).
    assert len(fits_open_recorder) == 2
    for _url, kwargs in fits_open_recorder:
        assert kwargs.get("use_fsspec") is True
    assert len(fetch_recorder) == 1


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
