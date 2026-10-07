"""
Tests for the time and energy helpers in `stixpy.product.sources._selection`, using synthetic
time and energy arrays so no data files are needed.
"""

import numpy as np
import pytest

import astropy.units as u
from astropy.table import QTable
from astropy.time import Time

from stixpy.product.sources._selection import (
    _check_index_limits,
    _check_time_limits,
    _energy_indices_format,
    _energy_indices_from_flat_edges,
    _energy_indices_from_range_pairs,
    _find_bin_index,
    _handle_datetime_strings,
    _time_indices_format,
)

# Ten 4 s bins from 2024-01-01T00:00:00 to 00:00:40, so bin i runs from 4i to 4(i + 1) s.
T0 = Time("2024-01-01T00:00:00")
DT = np.full(10, 4.0) * u.s
TIMES = T0 + (np.arange(10) * 4 + 2) * u.s

# Energy bins 4-5, 5-6, ..., 9-10, 10-12, 12-15, 15-20 and the open top bin 20-nan keV.
EDGES = np.array([4, 5, 6, 7, 8, 9, 10, 12, 15, 20, np.nan], dtype=float)
E_LOW = EDGES[:-1]
E_HIGH = EDGES[1:]
ENERGIES = QTable({"e_low": E_LOW * u.keV, "e_high": E_HIGH * u.keV})


def iso(seconds):
    return (T0 + seconds * u.s).isot


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("indices", [[0, 3, 9], np.array([0, 3, 9]), [np.int64(2)]])
def test_time_flat_indices_returned_unchanged(indices):
    assert _time_indices_format(indices, TIMES, DT) is indices


@pytest.mark.parametrize("indices", [[[0, 3], [5, 9]], [(1, 2)], np.array([[0, 4]])])
def test_time_index_pairs_returned_unchanged(indices):
    assert _time_indices_format(indices, TIMES, DT) is indices


@pytest.mark.parametrize("indices", [[10], [-1], [[0, 10]], [[-1, 3]]])
def test_time_indices_outside_file_raise(indices):
    with pytest.raises(ValueError, match="outside the file, which has time indices 0-9"):
        _time_indices_format(indices, TIMES, DT)


def test_time_string_pairs():
    result = _time_indices_format([[iso(0), iso(12)], [iso(20), iso(40)]], TIMES, DT)
    assert result == [[0, 2], [5, 9]]


def test_time_flat_strings_are_range_edges():
    result = _time_indices_format([iso(0), iso(12), iso(40)], TIMES, DT)
    assert result == [[0, 2], [3, 9]]


def test_time_objects_accepted():
    result = _time_indices_format([[T0 + 4 * u.s, T0 + 16 * u.s]], TIMES, DT)
    assert result == [[1, 3]]
    result = _time_indices_format(T0 + [0, 8, 16] * u.s, TIMES, DT)
    assert result == [[0, 1], [2, 3]]


def test_time_range_takes_only_complete_bins():
    # 1 s - 15 s covers bins 1 and 2 completely, and bins 0 and 3 only in part
    assert _handle_datetime_strings([[iso(1), iso(15)]], TIMES, DT) == [[1, 2]]


def test_time_range_edges_rounded_to_milliseconds():
    # ISO strings round to the millisecond, which must not drop the bins on the edges
    times = TIMES + 0.4 * u.ms
    assert _handle_datetime_strings([[(T0 + 0.4 * u.ms).isot, (T0 + 8.0004 * u.s).isot]], times, DT) == [[0, 1]]


def test_time_range_without_complete_bin_raises():
    with pytest.raises(ValueError, match="does not contain a complete data bin"):
        _time_indices_format([[iso(1), iso(3)]], TIMES, DT)


def test_time_range_with_wrong_length_raises():
    with pytest.raises(ValueError, match="exactly 2 elements"):
        _handle_datetime_strings([[iso(0), iso(4), iso(8)]], TIMES, DT)


def test_single_flat_time_raises():
    with pytest.raises(ValueError, match="need at least two"):
        _time_indices_format([iso(0)], TIMES, DT)


@pytest.mark.parametrize("pair", [[-1, 8], [0, 41]])
def test_times_outside_file_raise(pair):
    with pytest.raises(ValueError, match="fall outside the file time range"):
        _time_indices_format([[iso(pair[0]), iso(pair[1])]], TIMES, DT)


def test_time_limits_tolerance():
    file_start, file_end = T0, T0 + 40 * u.s
    _check_time_limits([[T0 - 0.5 * u.ms, T0 + 40.0005 * u.s]], file_start, file_end)
    with pytest.raises(ValueError, match="fall outside"):
        _check_time_limits([[T0 - 2 * u.ms, T0 + 4 * u.s]], file_start, file_end)


def test_index_limits():
    _check_index_limits([[0, 9]], 10)
    with pytest.raises(ValueError, match=r"Time indices \[10, 12\]"):
        _check_index_limits([0, 10, 12, 10], 10)


@pytest.mark.parametrize("time_indices", [[1.5, 2.5], [[1.5, 2.5]], [[0, 1, 2]], [None]])
def test_time_unknown_format_raises(time_indices):
    with pytest.raises(ValueError, match="Cannot determine format|Nested lists must be"):
        _time_indices_format(time_indices, TIMES, DT)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (4, 10, [0, 5]),  # centres 4.5 to 9.5
        (5, 6, [1, 1]),
        (4.5, 5.5, [0, 1]),  # range edges on the centres are included
        (10, 20, [6, 8]),
    ],
)
def test_find_bin_index(start, end, expected):
    assert _find_bin_index(start, end, E_LOW, E_HIGH) == expected


def test_find_bin_index_returns_ints():
    assert all(type(i) is int for i in _find_bin_index(4, 10, E_LOW, E_HIGH))


def test_find_bin_index_never_selects_open_top_bin():
    with pytest.raises(ValueError, match="does not contain the centre of any product energy bin"):
        _find_bin_index(20, 1000, E_LOW, E_HIGH)


def test_find_bin_index_empty_range_raises():
    with pytest.raises(ValueError, match=r"Energy range \[5.1 - 5.4\] keV"):
        _find_bin_index(5.1, 5.4, E_LOW, E_HIGH)


def test_energy_from_flat_edges():
    assert _energy_indices_from_flat_edges(np.array([4, 7, 10, 20]), E_LOW, E_HIGH) == [[0, 2], [3, 5], [6, 8]]


def test_energy_from_range_pairs():
    assert _energy_indices_from_range_pairs(np.array([[4, 6], [10, 15]]), E_LOW, E_HIGH) == [[0, 1], [6, 7]]


@pytest.mark.parametrize("energy_indices", [None, [0, 2, 5], [[0, 3]], np.array([1, 2])])
def test_energy_non_quantity_returned_unchanged(energy_indices):
    assert _energy_indices_format(energy_indices, ENERGIES) is energy_indices


def test_energy_quantity_flat_and_pairs():
    assert _energy_indices_format([4, 7, 10] * u.keV, ENERGIES) == [[0, 2], [3, 5]]
    assert _energy_indices_format([[4, 6], [10, 15]] * u.keV, ENERGIES) == [[0, 1], [6, 7]]


def test_energy_quantity_converted_to_kev():
    assert _energy_indices_format([[4000, 6000]] * u.eV, ENERGIES) == [[0, 1]]


@pytest.mark.parametrize("energy_indices", [[[3, 6]] * u.keV, [[10, 25]] * u.keV, [2, 6] * u.keV])
def test_energy_outside_file_raises(energy_indices):
    # nanmax skips the NaN upper edge of the open top bin, so the file covers 4 - 20 keV
    with pytest.raises(ValueError, match=r"energy range of the file \[4.0 - 20.0\] keV"):
        _energy_indices_format(energy_indices, ENERGIES)


def test_energy_quantity_wrong_dimensions_raises():
    with pytest.raises(ValueError, match="must be either 1D"):
        _energy_indices_format(np.full((1, 1, 2), 5) * u.keV, ENERGIES)
