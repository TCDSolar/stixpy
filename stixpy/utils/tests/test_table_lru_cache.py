from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

from stixpy.coordinates.transforms import STIX_EPHEMERIS_CACHE, get_hpc_info, load_ephemeris_fits_to_cache
from stixpy.utils.table_lru import TableLRUCache


@pytest.fixture
def cache():
    """Fixture to create a fresh TableLRUCache instance."""
    return TableLRUCache("testcache", maxsize=10, default_bin_duration=1 * u.s)


@pytest.fixture
def mock_data_table():
    """Fixture to create mock ephemeris data."""
    now = Time(datetime.now())
    times = now + np.arange(10) * timedelta(seconds=1)
    data = QTable({"time": times, "value": np.arange(10)})
    return data


def test_initialization_default():
    """Test default initialization of TableLRUCache."""
    cache = TableLRUCache("testcache")
    assert cache.maxsize == 300000
    assert cache.default_bin_duration == 64 * u.s
    assert len(cache.cache) == 0


def test_initialization_custom_maxsize():
    """Test initialization with a custom maxsize."""
    cache = TableLRUCache("testcache", maxsize=100)
    assert cache.maxsize == 100


def test_clear_cache(cache, mock_data_table):
    """Test clearing the cache."""
    cache.put(mock_data_table)
    assert len(cache.cache) == len(mock_data_table)
    cache.clear()
    assert len(cache.cache) == 0


def test_put_data(cache, mock_data_table):
    """Test adding data to the cache."""
    cache.put(mock_data_table)
    assert len(cache.cache) == len(mock_data_table)
    assert "__lcu" in cache.cache.colnames


def test_get_data(cache, mock_data_table):
    """Test retrieving data from the cache."""
    cache.put(mock_data_table)
    start_time = mock_data_table["time"][2]
    end_time = mock_data_table["time"][5]
    result = cache.get(start_time, end_time)
    assert len(result) == 4
    assert all(result["time"] >= start_time)
    assert all(result["time"] <= end_time)
    assert all(result["__lcu"] == 1)  # code for HIT


def test_get_data_no_match(cache, mock_data_table):
    """Test retrieving data when no matching time range exists."""
    cache.put(mock_data_table)
    start_time = Time(datetime.now() + timedelta(days=1))
    result = cache.get(start_time)
    assert result is None


def test_get_data_single_time(cache, mock_data_table):
    """Test retrieving data for a single time point."""
    cache.put(mock_data_table)
    start_time = mock_data_table["time"][3]
    result = cache.get(start_time)
    assert len(result) == 1
    assert result["time"][0] == start_time
    assert result["__lcu"][0] == 1  # code for HIT


def test_prune_logic(cache, mock_data_table):
    """Test internal pruning logic.

    pruning should be cald if the cache is full and new data is added
    """
    cache.maxsize = len(mock_data_table) - 3
    cache.put(mock_data_table)
    assert len(cache.cache) <= cache.maxsize


def test_put_new_data_overrides_same_old_data(cache):
    now = Time(datetime.now())
    times = now + np.arange(5) * timedelta(seconds=1)
    data_first = QTable({"time": times, "value": np.full((5,), 1)})
    data_last = QTable({"time": times, "value": np.full((5,), 2)})

    cache.put(data_first)
    assert len(cache.cache) == len(data_first)

    found = cache.get(times[0])
    assert len(found) == 1
    assert found["value"][0] == 1

    # Add new data with the same time but different value
    # This should override the old data
    cache.put(data_last)

    found = cache.get(times[0])
    assert len(found) == 1
    assert found["value"][0] == 2


def test_get_range_in_block(cache):
    """Test retrieving data for a range with time gaps.
    This test checks if the cache can handle time gaps correctly.
    should return all data in the range or None if any gaps in range
    """
    now = Time(datetime.now())
    times = now + np.arange(10) * u.s
    times[5:] += 2 * u.s  # introduce a gap in the data
    data = QTable({"time": times, "value": np.full((10,), 1)})
    cache.put(data)
    start_time = data["time"][1]
    end_time = data["time"][3]
    result = cache.get(start_time, end_time)
    assert len(result) == 3
    assert all(result["time"] >= start_time)
    assert all(result["time"] <= end_time)

    # Test with a range that includes a gap
    end_time = data["time"][6]
    result = cache.get(start_time, end_time)
    assert result is None


def test_lru_data_stays(cache):
    """Test that the LRU data stays in the cache."""
    now = Time(datetime.now())
    times = now + np.arange(10) * timedelta(seconds=1)
    data = QTable({"time": times, "value": np.arange(10)})
    cache.put(data)
    assert len(cache.cache) == len(data)

    # Simulate accessing some data to update the LRU
    assert cache.get(times[2])["value"] == 2
    assert cache.get(times[5])["value"] == 5

    # reduce the cache size to trigger pruning
    cache.maxsize = 3
    cache._prune()

    # Check that the accessed data is still in the cache
    assert len(cache.cache) > 0
    assert len(cache.cache) <= 3
    assert cache.get(times[2]) is not None
    assert cache.get(times[5]) is not None


def test_gloabl_ephemeris_cache():
    """Test that the global ephemeris cache is initialized."""
    assert STIX_EPHEMERIS_CACHE is not None
    assert isinstance(STIX_EPHEMERIS_CACHE, TableLRUCache)


@pytest.mark.remote_data
def test_get_hpc_info_fills_cache():
    STIX_EPHEMERIS_CACHE.clear()
    assert len(STIX_EPHEMERIS_CACHE.cache) == 0
    res = get_hpc_info(Time("2023-01-01T00:00:00"), Time("2023-01-01T00:00:10"))
    assert res is not None
    assert len(STIX_EPHEMERIS_CACHE.cache) > 0


@pytest.mark.remote_data
def test_get_hpc_info_cache_hit():
    """Test that the cache is used when available."""
    STIX_EPHEMERIS_CACHE.clear()
    assert len(STIX_EPHEMERIS_CACHE.cache) == 0
    res1 = get_hpc_info(Time("2023-01-01T12:00:00"), Time("2023-01-01T12:00:10"))
    assert res1 is not None
    assert len(STIX_EPHEMERIS_CACHE.cache) > 0

    # Call again with the same time range to check if cache is used
    res2 = STIX_EPHEMERIS_CACHE.get(Time("2023-01-01T12:00:00"), Time("2023-01-01T12:30:00"))
    assert res2 is not None
    assert res2["__lcu"][0] == 1  # code for HIT


@pytest.mark.remote_data
def test_get_hpc_cache_same_as_first_call():
    """Test that the cache is used when available."""
    STIX_EPHEMERIS_CACHE.clear()
    assert len(STIX_EPHEMERIS_CACHE.cache) == 0

    # just on time point
    res1 = get_hpc_info(Time("2023-01-01T12:00:00"))
    assert res1 is not None
    assert len(STIX_EPHEMERIS_CACHE.cache) > 0

    res2 = get_hpc_info(Time("2023-01-01T12:00:00"))
    assert res1[0] == res2[0]  # should be the same data


@pytest.mark.remote_data
def test_get_hpc_bypass_cache():
    """Test that the cache is used when available."""
    STIX_EPHEMERIS_CACHE.clear()
    assert len(STIX_EPHEMERIS_CACHE.cache) == 0

    # just on time point
    res1 = get_hpc_info(Time("2023-01-01T12:00:00"))
    assert res1 is not None
    assert len(STIX_EPHEMERIS_CACHE.cache) > 0

    # corrupt the cache to test bypass
    for row_idx in range(len(STIX_EPHEMERIS_CACHE.cache)):
        for angle_idx in range(3):
            STIX_EPHEMERIS_CACHE.cache["roll_angle_rpy"][row_idx][angle_idx] = -9999 * u.deg
    res2 = get_hpc_info(Time("2023-01-01T12:00:00"))

    assert res2[0] == -9999 * u.deg  # should be from cache
    STIX_EPHEMERIS_CACHE.bypass_cache_read = True
    res_fresh = get_hpc_info(Time("2023-01-01T12:00:00"))
    assert res_fresh[0] == res1[0]  # should be fresh data not from cache so the same as res1

    STIX_EPHEMERIS_CACHE.bypass_cache_read = False
    res_refreshed = get_hpc_info(Time("2023-01-01T12:00:00"))
    assert res_refreshed[0] == res_fresh[0]


@pytest.mark.remote_data
def test_get_hpc_source_cache():
    """Test that the cache is used when available."""
    STIX_EPHEMERIS_CACHE.clear()
    assert len(STIX_EPHEMERIS_CACHE.cache) == 0

    hpc_source = list()
    res1 = get_hpc_info(Time("2023-01-01T12:00:00"), Time("2023-01-03T13:00:00"), return_source=hpc_source)
    assert res1 is not None
    assert len(STIX_EPHEMERIS_CACHE.cache) > 0
    # multiple days so multiple source files
    assert len(hpc_source[0]) > 1
    assert "solo_ANC_stix-asp-ephemeris" in hpc_source[0]


@pytest.mark.remote_data
def test_load_anc_file():
    """Test loading an ANC file."""
    start_time = Time("2023-01-01T12:00:00")
    end_time = Time("2023-01-01T12:30:00")
    query = Fido.search(
        a.Time(start_time, end_time),
        a.Instrument.stix,
        a.Level.anc,
        a.stix.DataType.asp,
        a.stix.DataProduct.asp_ephemeris,
    )
    if len(query["stix"]) == 0:
        raise ValueError(f"No STIX pointing data found for time range {start_time} to {end_time}.")
    else:
        query["stix"].filter_for_latest_version()
    aux_files = Fido.fetch(query["stix"])

    assert len(aux_files) > 0

    STIX_EPHEMERIS_CACHE.clear()
    for file in aux_files:
        load_ephemeris_fits_to_cache(file)

    assert len(STIX_EPHEMERIS_CACHE.cache) > 0

    # Call again with the same time range to check if cache is used
    res2 = STIX_EPHEMERIS_CACHE.get(Time("2023-01-01T12:00:00"), Time("2023-01-01T12:30:00"))
    assert res2 is not None
    assert res2["__lcu"][0] == 1  # code for HIT
