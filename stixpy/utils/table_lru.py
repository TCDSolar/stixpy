from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.table import QTable, unique, vstack

from stixpy.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["TableLRUCache"]


class TableLRUCache:
    """
    A LRU cache based on astropy.table.Table.
    Search and uniqueness is working on the 'time' column.
    """

    def __init__(self, name: str, *, maxsize=300000, default_bin_duration=64 * u.s):
        """Creates and initialize a LRU cache table.

        Parameters
        ----------
        name : str
            name of the cache
            this is used for logging only
        maxsize : int, optional
            max rows in the cache table, by default 300000 ~= 6 month of data for 64s time resolution
        default_bin_duration : int, optional
            default time duration of a data entry, by default 64*u.s
            this is used to determine if the search results might contain data gaps
        """
        self.name = name
        self.maxsize = maxsize
        self.cache = QTable()
        self.default_bin_duration = default_bin_duration

    def clear(self):
        """
        Clear the cache.
        """
        self.cache = QTable()
        logger.info(f"{self.name}: Cleared LRU cache.")

    def put(self, anc_data):
        """
        Add data to the cache.
        """
        # create a primary index by time if not set so far
        if len(self.cache.indices) > 0:
            self.cache.add_index("time")

        anc_data["__lcu"] = datetime.now().timestamp()
        logger.info(f"{self.name}: Adding {len(anc_data)} rows to STIX ephemeris cache.")
        self.cache = vstack([self.cache, anc_data])
        self.cache = unique(self.cache, keys=["time"], keep="last")
        self._prune()

    def __len__(self):
        """
        Return the number of rows in the cache.
        """
        return len(self.cache)

    def _prune(self):
        """
        Remove the not used data from the cache.
        new size will be 80% of the maxsize to avoid frequent pruning
        """
        if len(self.cache) > self.maxsize:
            logger.info(f"{self.name}:Pruning LRU cache.")

            sorted_by_lcu_time_idx = self.cache.argsort(keys=["__lcu", "time"])
            # remove the oldest data
            self.cache = self.cache[sorted_by_lcu_time_idx[-int(self.maxsize * 0.8) :]]

    def get(self, start_time, end_time=None):
        r"""
        Get data from the cache.

        Parameters
        ----------
        start_time : `astropy.time.Time`
            Time or start of a time interval.

        Returns
        -------
        data : `astropy.table.Table`
            a deep copy of the cached data for the given time range.

            none if no data is found in the cache.
            none if data in the cache might have time gaps (based on the default bin duration)
        """

        # return None

        if len(self.cache) == 0:
            return None

        if end_time is None:
            end_time = start_time

        # TODO: should we use the time_end column for a better search
        # indices = np.argwhere(
        #     ((self.cache["time"] >= start_time) | (self.cache["time_end"] >= start_time))
        #     & ((self.cache["time_end"] <= end_time) | (self.cache["time"] <= end_time))
        # )

        if end_time is not None:
            indices = np.argwhere((self.cache["time"] >= start_time) & (self.cache["time"] <= end_time))
        else:
            indices = np.argwhere((self.cache["time"] >= start_time) & (self.cache["time"] <= start_time))

        indices = indices.flatten()

        if len(indices) == 0:
            return None

        self.cache["__lcu"][indices] = datetime.now().timestamp()

        if len(indices) == 1 and end_time == start_time:
            # single time row copy
            logger.info(f"{self.name}: Using cached data for time point {start_time} to {end_time}")
            ret_val = self.cache[indices].copy(copy_data=True)
            ret_val["__lcu"] = 1  # mark as a cache hit
            return ret_val

        # multiple times simple heuristic if we have a all data within time range
        duration = abs((end_time - start_time).to_value(u.s))
        frames = np.ceil(duration / self.default_bin_duration.to_value(u.s) * 0.98)  # allow for minor error
        if len(indices) < frames:
            logger.info(f"{self.name}: not enough data in cache for time range {start_time} to {end_time}")
            return None

        # return a copy of the cached data
        logger.info(f"{self.name}: Using cached data for time range {start_time} to {end_time}")
        ret_val = self.cache[indices].copy(copy_data=True)
        ret_val["__lcu"] = 1  # mark as a cache hit
        return ret_val
