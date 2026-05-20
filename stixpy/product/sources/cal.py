"""
STIX calibration (CAL) products.

Each daily energy-calibration file contains one row in the ``DATA`` extension
holding the per-pixel ``offset`` / ``gain`` vectors and the actual energy-edge
table for that day. The on-board ELUT in use during the recording is named in
the ``CONTROL`` extension's ``ob_elut_name`` field.

The :func:`find_energy_calibration_file_for_time` helper applies the documented
selection strategy to pick the correct daily CAL file for a given flare time
range (or point in time) — see the function's docstring for the full rules.
"""

import re
import warnings
from pathlib import Path

import astropy.units as u
from astropy.time import Time
from astropy.units import Quantity

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import TimeRange

from stixpy.product.product import L1Product
from stixpy.utils.logging import get_logger

__all__ = [
    "CalibrationProduct",
    "EnergyCalibration",
    "find_energy_calibration_file_for_time",
]

logger = get_logger(__name__)

# Energy-calibration filename pattern: capture both start and end timestamps.
# Example: solo_CAL_stix-cal-energy_20250302T230511-20250303T230512_V02.fits
_CAL_FILENAME_RE = re.compile(r"solo_CAL_stix-cal-energy_(?P<start>\d{8}T\d{6})-(?P<end>\d{8}T\d{6})_V\d+\.fits$")
_FILENAME_TIME_FORMAT = "%Y%m%dT%H%M%S"

# Strategy parameter: candidate cal files must span at least this long
# (filename-derived duration).
_MIN_CAL_DURATION = 12 * u.h


class CalibrationProduct(L1Product):
    """
    Base class for STIX calibration products (LEVEL == ``CAL``).

    Concrete subclasses (currently only :class:`EnergyCalibration`) implement
    ``is_datasource_for``; this base provides the shared time/ELUT accessors.
    """

    @property
    def time(self) -> Time:
        return self.data["time"]

    @property
    def exposure_time(self) -> Quantity[u.s]:
        return self.data["timedel"].to(u.s)

    @property
    def time_range(self) -> TimeRange:
        """`~sunpy.time.TimeRange` covered by the calibration acquisition."""
        return TimeRange(self.time[0], self.time[-1] + self.exposure_time[-1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n    {self.time_range}\n    ELUT: {self.ob_elut_name}"


class EnergyCalibration(CalibrationProduct):
    """Daily energy-calibration product (``a.stix.DataProduct.cal_energy``)."""

    @property
    def ob_elut_name(self) -> str:
        """On-board ELUT in use when this calibration was recorded."""
        return str(self.control["ob_elut_name"][0])

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        sst = tuple(meta[k] for k in ("STYPE", "SSTYPE", "SSID"))
        return sst == (21, 6, 41) and meta["level"] == "CAL"


def _parse_cal_filename(name: str) -> tuple[Time, Time]:
    """
    Extract (start, end) UTC times from an energy-calibration filename.

    Raises ``ValueError`` if ``name`` doesn't match the expected pattern.
    """
    match = _CAL_FILENAME_RE.search(str(name))
    if not match:
        raise ValueError(f"Not a recognised energy-calibration filename: {name!r}")
    start = Time.strptime(match["start"], _FILENAME_TIME_FORMAT)
    end = Time.strptime(match["end"], _FILENAME_TIME_FORMAT)
    return start, end


def _normalise(time) -> tuple[Time, Time]:
    """
    Reduce the input shape to ``(t_lookup, t_mid)``.

    * Scalar `~astropy.time.Time` → both equal to the input.
    * `~sunpy.time.TimeRange` or 2-element ``Time`` / 2-tuple →
      ``t_lookup = start``, ``t_mid = start + (end - start) / 2``.
    """
    if isinstance(time, Time) and time.isscalar:
        return time, time
    if isinstance(time, TimeRange):
        tr = time
    else:
        tr = TimeRange(*time)
    return Time(tr.start), Time(tr.start) + (Time(tr.end) - Time(tr.start)) / 2


def find_energy_calibration_file_for_time(time) -> Path:
    """
    Apply the documented selection strategy and return the daily energy-
    calibration FITS file path applicable at the given instant or range.

    Accepts either a scalar `~astropy.time.Time` (point query) or a
    `~sunpy.time.TimeRange` / 2-tuple / length-2 ``Time`` (range query —
    e.g. a flare time range). All filtering is done on local metadata
    (Fido search response + the local ``elut_index.csv``); the only CAL
    file actually downloaded is the one that wins.

    Strategy
    --------
    1. **Fido search** — ask for CAL files in
       ``[day(t_lookup) - 1 day, day(t_lookup) + 1 day]`` (the three
       calendar days centred on the day of ``t_lookup``). If zero rows are
       returned, raise ``ValueError``. No CAL file is downloaded yet.

    2. **Metadata-only filtering and ranking** — using ``Start Time`` /
       ``End Time`` (parsed from the filename by
       :meth:`stixpy.net.client.STIXClient.post_search_hook`) and
       :func:`stixpy.calibration.energy.get_elut`, for every search-result
       row:

       * Discard rows whose duration is below 12 hours.
       * Discard rows whose start time maps to a different entry in
         ``stixpy/config/data/elut/elut_index.csv`` than ``t_lookup``
         does — i.e. ``get_elut(start).file != get_elut(t_lookup).file``.
       * Rank survivors by ``|file_mid - t_mid|`` ascending.

       If nothing survives, raise ``ValueError``.

    3. **Single download** — `~sunpy.net.Fido.fetch` only the best-ranked
       row. Every other candidate stays remote.

    4. **Post-download sanity check** — open the downloaded file via
       `~stixpy.product.Product`; if its on-board ``ob_elut_name`` doesn't
       match the ELUT entry the strategy ranked it under, emit a
       ``UserWarning`` (the file is still returned because it remains the
       best metadata-based choice — the warning surfaces a divergence
       between the local index and the file's recorded ELUT).

    For a point query, ``t_lookup`` and ``t_mid`` collapse to the query
    time. For a range query, ``t_lookup`` is the range start and
    ``t_mid`` is the range midpoint.

    Parameters
    ----------
    time : `~astropy.time.Time` or `~sunpy.time.TimeRange` or 2-tuple
        Point in time, or time range bounding the data to be calibrated.

    Returns
    -------
    Path
        Local path to the chosen CAL FITS file (already downloaded).

    Raises
    ------
    ValueError
        If the Fido search returns no rows, if every row fails the
        duration / ELUT filters, or if the chosen download fails.

    Warns
    -----
    UserWarning
        If the downloaded file's on-board ``ob_elut_name`` disagrees with
        the ELUT entry the local index resolved for the file's start time.
    """
    # Import locally to avoid a top-level circular import:
    # stixpy.calibration.energy imports Product, which imports this module
    # during sources/__init__.py star-import.
    from stixpy.calibration.energy import get_elut
    from stixpy.product.product_factory import Product

    t_lookup, t_mid = _normalise(time)

    # Midnight of the day containing t_lookup, then ±1 / +2 days.
    day_midnight = Time(t_lookup.utc.isot[:10])
    window_start = day_midnight - 1 * u.day
    window_end = day_midnight + 2 * u.day

    logger.info(f"Searching CAL files for {t_lookup} (window {window_start} – {window_end})")
    query = Fido.search(
        a.Time(window_start, window_end),
        a.Instrument.stix,
        a.Level("CAL"),
        a.stix.DataType.cal,
        a.stix.DataProduct.cal_energy,
    )
    if len(query["stix"]) == 0:
        raise ValueError(
            f"No calibration files found in the 3-day window around {t_lookup} ({window_start} – {window_end})."
        )
    query["stix"].filter_for_latest_version()

    # Metadata-only filtering + ranking. The ELUT match is implemented as
    # "both the flare time and the candidate's start time map to the same
    # entry in stixpy/config/data/elut/elut_index.csv". get_elut(t).file is
    # the unique identifier of that entry, so a direct string compare is
    # exactly the same-row test — no CAL FITS file content is read.
    flare_elut_file = get_elut(t_lookup).file
    logger.debug(f"ELUT entry active at {t_lookup}: {flare_elut_file}")

    candidates: list[tuple[float, int, str]] = []  # (distance_to_mid, row_idx, filename)
    rejected_short: list[str] = []
    rejected_elut: list[str] = []
    for i, row in enumerate(query["stix"]):
        filename = str(row["url"]).rsplit("/", 1)[-1]
        start = Time(row["Start Time"])
        end = Time(row["End Time"])
        duration = end - start
        if duration < _MIN_CAL_DURATION:
            rejected_short.append(filename)
            continue
        try:
            file_elut_file = get_elut(start).file
        except ValueError:
            # No ELUT entry covers the file's start time → can't compare.
            rejected_elut.append(filename)
            continue
        if file_elut_file != flare_elut_file:
            rejected_elut.append(filename)
            continue
        cal_mid = start + duration / 2
        distance = abs((cal_mid - t_mid).sec)
        candidates.append((distance, i, filename))

    if not candidates:
        raise ValueError(
            f"No calibration files passed the selection filters for {t_lookup}. "
            f"Short duration: {rejected_short}. ELUT mismatch: {rejected_elut}."
        )

    # Single download — the chosen file. Everything before this was metadata-only.
    candidates.sort(key=lambda c: c[0])
    distance, idx, filename = candidates[0]
    single = query["stix"][idx : idx + 1]
    downloaded = Fido.fetch(single)
    if len(downloaded.errors) > 0:
        raise ValueError(f"Errors downloading {filename}.")
    path = Path(downloaded[0])

    # Final sanity check: the on-board ELUT recorded in the file's CONTROL
    # table should match the local elut_index entry we ranked it under. A
    # mismatch means either the local index is stale relative to operations
    # or the file's recorded ob_elut_name is wrong — worth flagging.
    product = Product(path)
    if isinstance(product, EnergyCalibration):
        expected_elut_stem = Path(flare_elut_file).stem
        if product.ob_elut_name != expected_elut_stem:
            warnings.warn(
                f"On-board ELUT in {path.name} is {product.ob_elut_name!r} but "
                f"stixpy/config/data/elut/elut_index.csv expects "
                f"{expected_elut_stem!r}; selection may be inconsistent."
            )

    logger.info(f"Selected calibration file: {path.name} (Δmid={distance:.0f} s)")
    return path
