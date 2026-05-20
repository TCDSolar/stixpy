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


@pytest.fixture
def mocks_factory(monkeypatch):
    """
    Returns a callable ``(paths, *, elut_name, file_elut_names=None,
    ob_elut_per_path=None)`` that installs the mocks for the helper.

    - ``elut_name`` is what ``get_elut`` returns by default (the flare-time
      lookup and, unless overridden, every file-start lookup).
    - ``file_elut_names`` is a ``{Path: str}`` mapping; if given, the
      ``get_elut`` mock returns the per-path ELUT name when called with
      that path's start time. Used to simulate ELUT-mismatch cases.
    - ``ob_elut_per_path`` controls the on-board ELUT name returned by the
      Product loader called for the *final* sanity check. Defaults to the
      same value as ``elut_name`` so the post-download check is silent.
    """
    FakeCal = _FakeEnergyCalibration

    def install(paths, *, elut_name, file_elut_names=None, ob_elut_per_path=None):
        if ob_elut_per_path is None:
            ob_elut_per_path = {p: elut_name for p in paths}
        # Map each file's start Time to its ELUT name for the get_elut mock.
        start_to_elut: dict = {}
        if file_elut_names is not None:
            for p, name in file_elut_names.items():
                start_to_elut[_parse_cal_filename(Path(p).name)[0]] = name

        monkeypatch.setattr(
            "stixpy.product.sources.cal.Fido",
            _FakeFido(_rows_for(paths)),
        )

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
# Remote-data smoke test
# ---------------------------------------------------------------------------


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
