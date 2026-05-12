"""
ANC ephemeris fetcher.

Internal module that searches, downloads, and loads STIX ANC asp_ephemeris
FITS files into a single QTable. Owns Fido / version filtering / day-boundary
padding so the cache and orchestrator layers don't have to.
"""

from pathlib import Path

import astropy.units as u
from astropy.table import QTable, vstack
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

from stixpy.product.product_factory import Product
from stixpy.product.sources.anc import Ephemeris
from stixpy.utils.logging import get_logger
from stixpy.utils.table import drop_fits_checksums

logger = get_logger(__name__)

__all__ = ["fetch_ephemeris_for_range"]


def fetch_ephemeris_for_range(
    start: Time,
    end: Time,
    *,
    midnight_pad: u.Quantity = 90 * u.s,
) -> QTable:
    """
    Search, version-filter, download, and load ANC asp_ephemeris files
    covering ``[start, end]``.

    The Fido query window is extended by ``midnight_pad`` whenever ``start``
    or ``end`` falls within ``midnight_pad`` of UTC midnight, so a query
    near a day boundary pulls the neighbour-day file too. Per the ephemeris
    requirements document, the row closest to a query just after midnight
    may live in the previous day's file (and vice versa).

    Parameters
    ----------
    start, end
        Scalar `~astropy.time.Time` defining the requested range.
    midnight_pad
        Extension applied to the Fido query window near day boundaries.

    Returns
    -------
    QTable
        Vstacked-and-sorted ``Ephemeris.data`` rows from all loaded files,
        with a ``__source`` column (filename) per row.

    Raises
    ------
    ValueError
        If the Fido search returns no results or downloads fail.
    """
    query_start, query_end = _apply_midnight_pad(start, end, midnight_pad)

    logger.info(f"Fido searching ANC asp_ephemeris for {query_start} – {query_end}")
    query = Fido.search(
        a.Time(query_start, query_end),
        a.Instrument.stix,
        a.Level.anc,
        a.stix.DataType.asp,
        a.stix.DataProduct.asp_ephemeris,
    )

    if len(query["stix"]) == 0:
        raise ValueError(f"No STIX pointing data found for time range {query_start} to {query_end}.")

    query["stix"].filter_for_latest_version()
    logger.debug(f"Downloading {len(query['stix'])} ANC files")
    aux_files = Fido.fetch(query["stix"])
    if len(aux_files.errors) > 0:
        raise ValueError("There were errors downloading the ANC data.")

    tables = []
    for aux_file in aux_files:
        aux_file = Path(aux_file)
        anc = Product(aux_file, data_only=True)
        if not isinstance(anc, Ephemeris):
            logger.warning(f"Skipping non-Ephemeris ANC file: {aux_file.name}")
            continue
        anc.data["__source"] = aux_file.name
        tables.append(anc.data)

    if not tables:
        raise ValueError("No Ephemeris files could be loaded from the Fido response.")

    drop_fits_checksums(*tables)
    ephemeris = vstack(tables)
    ephemeris.sort(keys=["time"])
    return ephemeris


def _apply_midnight_pad(start: Time, end: Time, pad: u.Quantity) -> tuple[Time, Time]:
    """
    Widen ``[start, end]`` toward the neighbour day(s) when either endpoint
    is within ``pad`` of UTC midnight, so cross-day queries pick up the
    neighbouring daily file.
    """
    pad_s = pad.to_value(u.s)
    seconds_per_day = 86400.0

    start_sod = _seconds_of_day(start)
    end_sod = _seconds_of_day(end)

    query_start = start - pad if start_sod < pad_s else start
    query_end = end + pad if (seconds_per_day - end_sod) < pad_s else end
    return query_start, query_end


def _seconds_of_day(t: Time) -> float:
    """Return seconds elapsed since UTC midnight for ``t``."""
    ymdhms = t.utc.ymdhms
    return float(ymdhms.hour * 3600 + ymdhms.minute * 60 + ymdhms.second)
