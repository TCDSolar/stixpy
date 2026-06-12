import warnings
from datetime import datetime

import numpy as np

import astropy.units as u
from astropy.table import QTable, unique, vstack
from astropy.time import Time

from stixpy.utils.logging import get_logger
from stixpy.utils.table import drop_fits_checksums

logger = get_logger(__name__)

__all__ = ["TableLRUCache"]


class TableLRUCache:
    """
    LRU cache of time-indexed astropy.table rows.

    The cache stores rows of a QTable keyed by the ``time`` column. Two
    query modes are supported:

    * Point query (``get_point``): returns the row whose ``time`` is closest
      to a requested instant, provided the distance is within
      ``max_point_distance``.
    * Range query (``get_range``): returns rows in ``[start - pad, end + pad]``
      iff the cache covers ``[start, end]`` without holes larger than
      ``max_gap``.

    The cache knows nothing about ANC files, Fido, or daily file boundaries.
    Those concerns live in the fetcher / orchestrator layers.

    Indexing
    --------
    The ``time`` column is kept sorted ascending (enforced by :meth:`put` and
    :meth:`_prune`). A float mirror of it in unix seconds, ``_times_unix``, is
    held alongside the table and refreshed by :meth:`_rebuild_index` whenever
    the cache is rebuilt or re-sorted. Both query modes binary-search this
    array with ``numpy.searchsorted`` (O(log n)) rather than scanning every
    row (O(n)): ``get_point`` locates the insertion point and inspects only the
    two straddling neighbours, while ``get_range`` resolves its window to a
    contiguous index slice. Precise, leap-second-aware ``Time`` arithmetic is
    then applied only to those few candidate rows. This makes the common usage
    pattern — one bulk day-add followed by many queries against a large cache —
    cheap; an astropy table index (``add_index``) is deliberately *not* used,
    as it would be invalidated by the ``vstack``/``unique``/``sort`` in
    :meth:`put` and does not serve nearest-neighbour or gap-aware range lookups.
    """

    def __init__(
        self,
        name: str,
        *,
        maxsize: int = 300000,
        nominal_bin: u.Quantity = 64 * u.s,
        max_point_distance: u.Quantity = 90 * u.s,
        max_gap: u.Quantity = 150 * u.s,
        default_bin_duration: u.Quantity | None = None,
    ):
        """
        Parameters
        ----------
        name
            Cache identifier used in log messages.
        maxsize
            Maximum number of rows retained. Default is ~6 months at 64 s.
        nominal_bin
            Nominal time resolution of the underlying data. Informational;
            used for log messages. Real spacing varies (the doc explicitly
            notes 64 s is not exact).
        max_point_distance
            Maximum |row.time - query_time| accepted for a point query.
            Beyond this, ``get_point`` returns ``None``.
        max_gap
            Maximum consecutive inter-row time gap accepted inside a range
            query; also the edge tolerance for considering a range covered.
        default_bin_duration
            Deprecated alias for ``nominal_bin``.
        """
        if default_bin_duration is not None:
            warnings.warn(
                "'default_bin_duration' is deprecated; use 'nominal_bin' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            nominal_bin = default_bin_duration

        self.name = name
        self.maxsize = maxsize
        self.nominal_bin = nominal_bin
        self.max_point_distance = max_point_distance
        self.max_gap = max_gap
        self.cache = QTable()
        # Float (unix seconds) mirror of the sorted ``time`` column, kept in
        # sync by ``_rebuild_index``. Enables O(log n) ``np.searchsorted``
        # lookups instead of scanning every row on each query.
        self._times_unix = np.empty(0, dtype=float)
        self.bypass_cache_read = False

    @property
    def default_bin_duration(self) -> u.Quantity:
        """Deprecated alias for ``nominal_bin``."""
        return self.nominal_bin

    def clear(self) -> None:
        """Empty the cache."""
        self.cache = QTable()
        self._rebuild_index()
        logger.info(f"{self.name}: Cleared LRU cache.")

    def __len__(self) -> int:
        return len(self.cache)

    def _rebuild_index(self) -> None:
        """
        Refresh the binary-search lookup array from the cache.

        Must be called whenever ``self.cache`` is rebuilt or re-sorted (i.e.
        after :meth:`put` / :meth:`_prune` / :meth:`clear`). The ``time``
        column is assumed sorted ascending, matching the cache invariant.
        """
        if len(self.cache) == 0:
            self._times_unix = np.empty(0, dtype=float)
        else:
            self._times_unix = np.asarray(self.cache["time"].unix, dtype=float)

    def put(self, anc_data, *, source="unknown") -> None:
        """
        Add rows to the cache. Existing rows with the same ``time`` are
        replaced (``keep="last"``). Injects ``__lcu`` (LRU timestamp) and
        ``__source`` columns; pass ``source`` as a string (broadcast to all
        rows) or a per-row array.
        """
        anc_data = anc_data.copy()
        anc_data["__lcu"] = datetime.now().timestamp()
        anc_data["__source"] = source
        drop_fits_checksums(anc_data, self.cache)
        logger.info(f"{self.name}: Adding {len(anc_data)} rows.")
        self.cache = vstack([self.cache, anc_data])
        self.cache = unique(self.cache, keys=["time"], keep="last")
        self.cache.sort(keys=["time"])
        self._prune()
        self._rebuild_index()

    def _prune(self) -> None:
        """Evict least-recently-used rows down to 80% of maxsize."""
        if len(self.cache) > self.maxsize:
            logger.info(f"{self.name}: Pruning LRU cache.")
            sorted_idx = self.cache.argsort(keys=["__lcu", "time"])
            self.cache = self.cache[sorted_idx[-int(self.maxsize * 0.8) :]]
            self.cache.sort(keys=["time"])
            self._rebuild_index()

    def get_point(self, t: Time, *, _force: bool = False) -> QTable | None:
        """
        Nearest-row lookup.

        Returns a 1-row deep copy of the row whose ``time`` is closest to ``t``
        if that distance is within ``max_point_distance``; otherwise ``None``.

        Parameters
        ----------
        t
            Scalar `~astropy.time.Time`.
        _force
            Internal-only escape hatch that ignores ``bypass_cache_read``.
            Used by the orchestrator immediately after a fresh fetch to read
            back what was just put.

        Raises
        ------
        TypeError
            If ``t`` is not scalar.
        """
        if not isinstance(t, Time) or not t.isscalar:
            raise TypeError("get_point requires a scalar astropy.time.Time")

        if len(self.cache) == 0 or (self.bypass_cache_read and not _force):
            return None

        # Binary search for the insertion point; the nearest row is one of the
        # two straddling neighbours. Precise (leap-second aware) Time
        # arithmetic is then done on at most those two rows.
        pos = int(np.searchsorted(self._times_unix, t.unix))
        cand = [j for j in (pos - 1, pos) if 0 <= j < len(self._times_unix)]
        dt = np.abs((self.cache["time"][cand] - t).to_value(u.s))
        k = int(np.argmin(dt))
        i = cand[k]
        if dt[k] > self.max_point_distance.to_value(u.s):
            return None

        self.cache["__lcu"][i] = datetime.now().timestamp()
        logger.info(f"{self.name}: point hit for {t}, nearest row at {self.cache['time'][i]} (Δ={dt[k]:.2f} s).")
        ret = self.cache[[i]].copy(copy_data=True)
        ret["__lcu"] = 1
        return ret

    def get_range(self, start: Time, end: Time, *, pad: u.Quantity = 0 * u.s, _force: bool = False) -> QTable | None:
        """
        Range lookup with honest coverage detection.

        ``[start, end]`` is considered "covered" iff:

        * A cached row exists within ``max_gap`` of ``start``.
        * A cached row exists within ``max_gap`` of ``end``.
        * Between those two brackets, no consecutive inter-row gap exceeds
          ``max_gap``.

        This bracketing semantic — rather than strict containment — is what
        the ephemeris doc requires. It also makes short range queries
        (shorter than the nominal bin cadence, e.g. a 10 s query against
        64 s data) work correctly: the brackets straddle the window.

        Returns rows in ``[start - pad, end + pad]``. If that slice is
        empty (very short range with ``pad=0``), the bracketing rows from
        the coverage check are returned so callers always have something
        to interpolate from. Returns ``None`` on coverage failure.

        Parameters
        ----------
        start, end
            Scalar `~astropy.time.Time`. ``start <= end``.
        pad
            Extra time to widen the returned slice on each side. Useful to
            bracket the requested range for downstream interpolation.

        Raises
        ------
        TypeError
            If ``start`` or ``end`` is not scalar.
        """
        if not isinstance(start, Time) or not start.isscalar:
            raise TypeError("get_range requires a scalar astropy.time.Time for 'start'")
        if not isinstance(end, Time) or not end.isscalar:
            raise TypeError("get_range requires a scalar astropy.time.Time for 'end'")

        if len(self.cache) == 0 or (self.bypass_cache_read and not _force):
            return None

        times = self.cache["time"]
        times_unix = self._times_unix
        max_gap_s = self.max_gap.to_value(u.s)

        # Coverage uses an expanded window so rows just outside [start, end]
        # can satisfy the bracket condition (essential for short queries).
        # Binary search bounds the window to a contiguous index slice rather
        # than scanning every row.
        lo = int(np.searchsorted(times_unix, (start - self.max_gap).unix, side="left"))
        hi = int(np.searchsorted(times_unix, (end + self.max_gap).unix, side="right"))
        expanded_idx = np.arange(lo, hi)
        if expanded_idx.size == 0:
            logger.info(f"{self.name}: no rows near [{start}, {end}] — not covered.")
            return None

        expanded_times = times[lo:hi]

        left_dt = float(np.min(np.abs((expanded_times - start).to_value(u.s))))
        right_dt = float(np.min(np.abs((expanded_times - end).to_value(u.s))))
        if left_dt > max_gap_s or right_dt > max_gap_s:
            logger.info(
                f"{self.name}: edge not covered (left={left_dt:.1f}s, right={right_dt:.1f}s, max_gap={max_gap_s}s)."
            )
            return None

        if expanded_idx.size >= 2:
            diffs = np.diff(times_unix[lo:hi])
            if diffs.max() > max_gap_s:
                logger.info(f"{self.name}: interior gap {diffs.max():.1f}s exceeds max_gap={max_gap_s}s — not covered.")
                return None

        pad_s = pad.to_value(u.s)
        if pad_s > 0:
            # Avoid `start - 0*u.s` arithmetic: it's not bit-equal to `start`
            # in astropy.Time, which would exclude rows lying exactly on the
            # boundary.
            s_lo = int(np.searchsorted(times_unix, (start - pad).unix, side="left"))
            s_hi = int(np.searchsorted(times_unix, (end + pad).unix, side="right"))
        else:
            s_lo = int(np.searchsorted(times_unix, start.unix, side="left"))
            s_hi = int(np.searchsorted(times_unix, end.unix, side="right"))
        slice_idx = np.arange(s_lo, s_hi)
        if slice_idx.size == 0:
            # Very short range with no pad: fall back to the bracketing rows
            # so the caller still has something to interpolate from.
            slice_idx = expanded_idx

        self.cache["__lcu"][slice_idx] = datetime.now().timestamp()
        logger.info(f"{self.name}: range hit [{start}, {end}] — returning {slice_idx.size} rows (pad={pad_s}s).")
        ret = self.cache[slice_idx].copy(copy_data=True)
        ret["__lcu"] = 1
        return ret

    def get(self, start: Time, end: Time | None = None) -> QTable | None:
        """
        Back-compat dispatcher.

        * ``end is None`` or ``end == start`` → :meth:`get_point`.
        * Otherwise → :meth:`get_range` with ``pad=0``.
        """
        if end is None or end == start:
            return self.get_point(start)
        return self.get_range(start, end, pad=0 * u.s)
