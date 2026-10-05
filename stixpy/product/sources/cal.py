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
from astropy.io import fits
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

# Defaults for find_energy_calibration_file_for_time. All three are
# overridable per-call via keyword arguments.
_DEFAULT_WINDOW_PAST = 5 * u.day  # how far back from day(t) the Fido search reaches
_DEFAULT_WINDOW_FUTURE = 5 * u.day  # how far forward
_DEFAULT_MIN_LIVETIME = 30_000 * u.s  # ≈ 8 h; healthy daily files run ~44 ks


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


def _read_livetime(url: str) -> Quantity:
    """
    Read the ``LIVETIME`` primary-header value from a CAL FITS file
    without downloading the full file.

    Uses :func:`astropy.io.fits.open` with ``use_fsspec=True`` and a
    small ``block_size`` (16 KB), which issues an HTTP Range request
    for just the first block — enough to cover the FITS primary header
    (≥ 2880 bytes). For local ``file://`` URLs the read is equally
    cheap. The default fsspec block size of 5 MB would otherwise fetch
    the whole ~160 KB CAL file in a single Range request, defeating
    the partial-download benefit.

    Returns ``LIVETIME`` as `~astropy.units.Quantity` in seconds.
    """
    with fits.open(url, use_fsspec=True, fsspec_kwargs={"block_size": 16384}) as hdul:
        return float(hdul[0].header["LIVETIME"]) * u.s


def find_energy_calibration_file_for_time(
    time,
    *,
    min_livetime: Quantity = _DEFAULT_MIN_LIVETIME,
    window_past: Quantity = _DEFAULT_WINDOW_PAST,
    window_future: Quantity = _DEFAULT_WINDOW_FUTURE,
) -> Path:
    """
    Apply the documented selection strategy and return the daily energy-
    calibration FITS file path applicable at the given instant or range.

    Accepts either a scalar `~astropy.time.Time` (point query) or a
    `~sunpy.time.TimeRange` / 2-tuple / length-2 ``Time`` (range query
    such as a flare time range). All metadata filters run before any
    CAL FITS file is fully downloaded: the only `~sunpy.net.Fido.fetch`
    is on the file that wins.

    Strategy
    --------
    1. **Fido search** — ask for CAL files in
       ``[day(t_lookup) - window_past, day(t_lookup) + window_future + 1 day)``
       (12 calendar days by default). The wide window lets the helper
       fall back to older files when the most-recent ones fail the
       LIVETIME check. If zero rows are returned, raise ``ValueError``.

    2. **Metadata-only filtering and ranking** — using ``Start Time`` /
       ``End Time`` (parsed from the filename by
       :meth:`stixpy.net.client.STIXClient.post_search_hook`) and
       :func:`stixpy.calibration.energy.get_elut`, for every row:

       * Discard rows whose duration is below 12 hours.
       * Discard rows whose start time maps to a different entry in
         ``stixpy/config/data/elut/elut_index.csv`` than ``t_lookup``.
       * Rank survivors by ``|file_mid - t_mid|`` ascending.

       If nothing survives, raise ``ValueError``.

    3. **LIVETIME walk** — for each candidate in ranked order, read
       the primary-header ``LIVETIME`` keyword via a fsspec range
       request (no full download yet). The first candidate with
       ``LIVETIME >= min_livetime`` wins. If every candidate falls
       below the threshold, raise ``ValueError`` listing the measured
       live-times.

    4. **Single download** — `~sunpy.net.Fido.fetch` only the winning
       row.

    5. **Post-download sanity check** — open the downloaded file via
       `~stixpy.product.Product`; if its on-board ``ob_elut_name``
       doesn't match the ELUT entry the local index resolved for the
       file's start time, emit a ``UserWarning``.

    For a point query, ``t_lookup`` and ``t_mid`` collapse to the query
    time. For a range query, ``t_lookup`` is the range start and
    ``t_mid`` is the range midpoint.

    Parameters
    ----------
    time : `~astropy.time.Time` or `~sunpy.time.TimeRange` or 2-tuple
        Point in time, or time range bounding the data to be calibrated.
    min_livetime : `~astropy.units.Quantity`, optional
        Minimum acceptable ``LIVETIME`` (primary FITS header) for the
        returned file, in time units. Default: ``30 000 s`` (~8 h),
        roughly two thirds of a healthy daily file's typical
        ``LIVETIME`` of ~44 ks. Floats are interpreted as seconds.
    window_past : `~astropy.units.Quantity`, optional
        How far back from ``day(t_lookup)`` the Fido search reaches.
        Default: ``10 * u.day``. Floats are interpreted as days.
    window_future : `~astropy.units.Quantity`, optional
        How far forward. Default: ``1 * u.day``.

    Returns
    -------
    Path
        Local path to the chosen CAL FITS file (already downloaded).

    Raises
    ------
    ValueError
        If the Fido search returns no rows, if every row fails the
        duration / ELUT filters, if every candidate falls below
        ``min_livetime``, or if the chosen download fails.

    Warns
    -----
    UserWarning
        If the downloaded file's on-board ``ob_elut_name`` disagrees
        with the ELUT entry the local index resolved for the file's
        start time.
    """
    # Import locally to avoid a top-level circular import:
    # stixpy.calibration.energy imports Product, which imports this module
    # during sources/__init__.py star-import.
    from stixpy.calibration.energy import get_elut
    from stixpy.product.product_factory import Product

    # Coerce numeric inputs to Quantity in sensible units.
    if not isinstance(min_livetime, Quantity):
        min_livetime = float(min_livetime) * u.s
    if not isinstance(window_past, Quantity):
        window_past = float(window_past) * u.day
    if not isinstance(window_future, Quantity):
        window_future = float(window_future) * u.day

    t_lookup, t_mid = _normalise(time)

    # Day-aligned window. The closing bound is kept exclusive (one extra
    # day past `window_future`) so the Fido search captures any file
    # whose calendar-day folder is `day + window_future`.
    day_midnight = Time(t_lookup.utc.isot[:10])
    window_start = day_midnight - window_past
    window_end = day_midnight + window_future + 1 * u.day

    logger.info(f"Searching CAL files for {t_lookup} (window {window_start} – {window_end})")
    query = Fido.search(
        a.Time(window_start, window_end),
        a.Instrument.stix,
        a.Level("CAL"),
        a.stix.DataType.cal,
        a.stix.DataProduct.cal_energy,
    )
    if len(query["stix"]) == 0:
        raise ValueError(f"No calibration files found in the window around {t_lookup} ({window_start} – {window_end}).")
    query["stix"].filter_for_latest_version()

    # Metadata-only filtering + ranking. The ELUT match is implemented as
    # "both the flare time and the candidate's start time map to the same
    # entry in stixpy/config/data/elut/elut_index.csv". get_elut(t).file is
    # the unique identifier of that entry, so a direct string compare is
    # exactly the same-row test — no CAL FITS file content is read.
    flare_elut_file = get_elut(t_lookup.utc.datetime).file
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
            file_elut_file = get_elut(start.utc.datetime).file
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

    # Rank by closest mid-time. Walk down the list checking LIVETIME via
    # fsspec range reads; first candidate at-or-above the threshold wins.
    candidates.sort(key=lambda c: c[0])
    rejected_livetime: list[tuple[str, float]] = []  # (filename, observed_livetime_s)
    winner = None
    for distance, idx, filename in candidates:
        url = str(query["stix"][idx]["url"])
        try:
            livetime = _read_livetime(url)
        except Exception as exc:
            # A header read can fail (network blip, range-request not
            # supported, missing key). Treat as a fail and continue.
            logger.warning(f"Could not read LIVETIME from {filename}: {exc}")
            rejected_livetime.append((filename, float("nan")))
            continue
        if livetime >= min_livetime:
            logger.info(
                f"Candidate {filename} passed LIVETIME check: "
                f"{livetime.to_value(u.s):.0f} s >= {min_livetime.to_value(u.s):.0f} s"
            )
            winner = (distance, idx, filename)
            break
        rejected_livetime.append((filename, livetime.to_value(u.s)))
        logger.info(
            f"Candidate {filename} below LIVETIME threshold: "
            f"{livetime.to_value(u.s):.0f} s < {min_livetime.to_value(u.s):.0f} s"
        )

    if winner is None:
        raise ValueError(
            f"No calibration files passed the LIVETIME threshold "
            f"({min_livetime.to_value(u.s):.0f} s) for {t_lookup}. "
            f"Observed live-times (seconds): {rejected_livetime}."
        )

    # Single download — the winning file.
    distance, idx, filename = winner
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
