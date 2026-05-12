"""
==============
Ephemeris Demo
==============

How to query STIX ephemeris (SOLO pointing, roll, and HEEQ position) via
`~stixpy.coordinates.transforms.get_hpc_info`, and how the internal cache
keeps subsequent calls cheap by avoiding redundant Fido searches.

ANC ephemeris FITS files are organised as daily files. The first query that
touches a given day downloads that whole file via Fido and writes every row
of it into the global cache. Any later query inside the same day is served
straight from memory — no network, no file I/O.

Imports
"""

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import Helioprojective
from sunpy.net import Fido
from sunpy.net import attrs as a

from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import STIX_EPHEMERIS_CACHE, get_hpc_info, load_ephemeris_fits_to_cache

logger = logging.getLogger(__name__)

###############################################################################
# Start from an empty cache so the demo's behaviour is reproducible.

STIX_EPHEMERIS_CACHE.clear()
print(f"Rows in cache: {len(STIX_EPHEMERIS_CACHE)}")

###############################################################################
# Case 1 — single time point
# --------------------------
#
# Pass a scalar `~astropy.time.Time`. ``get_hpc_info`` interpolates the
# ANC data onto that exact instant. The first call to any time in
# 2023-01-01 triggers a Fido search and caches the entire daily file.

t = Time("2023-01-01T12:00:00")
roll, solo_heeq, stix_pointing = get_hpc_info(t)
print(f"At {t}: roll={roll}, pointing={stix_pointing}")
print(f"Rows in cache after first query: {len(STIX_EPHEMERIS_CACHE)}")

###############################################################################
# Case 2 — time range
# -------------------
#
# A scalar ``times`` plus an ``end_time`` averages the ephemeris across
# ``[start, end]``. This is the calling pattern used by imaging code
# (see `~stixpy.calibration.visibility.calibrate_visibility`) where one
# averaged pointing is needed for an integration window.
#
# This call lands entirely inside the day already cached above, so no
# Fido search runs — the lookup is in-memory.

start = Time("2023-01-01T15:20:00")
end = Time("2023-01-01T15:23:00")
roll, solo_heeq, stix_pointing = get_hpc_info(start, end)
print(f"Averaged over [{start}, {end}]: roll={roll}, pointing={stix_pointing}")

###############################################################################
# Case 3 — array of times
# -----------------------
#
# Passing an array returns one interpolated value per input time. This is
# what `astropy` coordinate frame transforms use under the hood when
# transforming a `~stixpy.coordinates.frames.STIXImaging` `~astropy.coordinates.SkyCoord`
# with a vector ``obstime``.

times = Time("2023-01-01") + np.arange(0, 24, 2) * u.h  # 12 points across the day
roll_array, _, _ = get_hpc_info(times)
print(f"Roll across the day shape: {roll_array.shape}")

###############################################################################
# Inspect the cache
# -----------------
#
# After the three calls above, the cache holds the full content of the
# daily ANC file for 2023-01-01 — a single Fido download served all three
# queries.

print(f"Rows in cache: {len(STIX_EPHEMERIS_CACHE)}")
print(f"Source files in cache: {set(STIX_EPHEMERIS_CACHE.cache['__source'])}")

###############################################################################
# Plot the cached roll-angle across the day to visualise that the whole
# day was cached after the first query.

day_start = Time("2023-01-01T00:00:00")
cached_times = STIX_EPHEMERIS_CACHE.cache["time"]
cached_roll = STIX_EPHEMERIS_CACHE.cache["roll_angle_rpy"][:, 0]
cached_hours = (cached_times - day_start).to_value(u.h)
query_hours = (times - day_start).to_value(u.h)

fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
ax.plot(cached_hours, cached_roll.to_value(u.deg), ".", markersize=2, label="cached rows")
ax.plot(query_hours, roll_array.to_value(u.deg), "o", color="C1", label="get_hpc_info(times)")
ax.set_xlabel("Hours since 2023-01-01 00:00 UTC")
ax.set_ylabel("Roll [deg]")
ax.set_title("Full day cached after first get_hpc_info call")
ax.legend()

###############################################################################
# Coordinate transforms use the cache implicitly
# ----------------------------------------------
#
# Astropy frame transforms involving `~stixpy.coordinates.frames.STIXImaging`
# call ``get_hpc_info`` under the hood — see ``stixim_to_hpc`` and
# ``hpc_to_stixim`` in `~stixpy.coordinates.transforms`. Because the day
# is already cached above, every transform below is served from memory:
# the cache row count does not change.

n_rows_before = len(STIX_EPHEMERIS_CACHE)

# Sample STIX (0,0) every 30 minutes through 2023-01-01.
sample_times = Time("2023-01-01") + np.arange(0, 24, 0.5) * u.h
stix_zero = SkyCoord(
    np.zeros(sample_times.size) * u.arcsec,
    np.zeros(sample_times.size) * u.arcsec,
    frame=STIXImaging(obstime=sample_times),
)
hpc_zero = stix_zero.transform_to(Helioprojective(obstime=sample_times))

n_rows_after = len(STIX_EPHEMERIS_CACHE)
print(f"Cache rows before transforms: {n_rows_before}")
print(f"Cache rows after  transforms: {n_rows_after}  (unchanged → cache served the transforms)")

###############################################################################
# Plot where STIX (0,0) lands on the Helioprojective disk through the day.
# Pure SOLO orbit + roll evolution — every point uses cached ephemeris.

fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
hours = (sample_times - sample_times[0]).to_value(u.h)
tx = hpc_zero.Tx.to_value(u.arcsec)
ty = hpc_zero.Ty.to_value(u.arcsec)
sc = ax.scatter(tx, ty, c=hours, cmap="viridis")
ax.set_aspect("equal")
ax.set_xlabel("HPC Tx [arcsec]")
ax.set_ylabel("HPC Ty [arcsec]")
ax.set_title("STIX (0,0) → Helioprojective across 2023-01-01\n(every transform served from cache)")

ax.set_xlim(32, 34)
ax.set_ylim(54, 56)
fig.colorbar(sc, ax=ax, label="Hours since 00:00 UTC")

###############################################################################
# Pre-warm the cache from local FITS files
# ----------------------------------------
#
# If you already have ANC FITS files on disk (e.g. mounted archive, prior
# download, or an offline environment), pre-warm the cache directly with
# `~stixpy.coordinates.transforms.load_ephemeris_fits_to_cache`. After
# that, calls to ``get_hpc_info`` for times inside those days hit the
# cache without any Fido call.

# Clear and demonstrate pre-warming using files we fetch ourselves.
STIX_EPHEMERIS_CACHE.clear()

query = Fido.search(
    a.Time("2023-01-02", "2023-01-03"),
    a.Instrument.stix,
    a.Level.anc,
    a.stix.DataType.asp,
    a.stix.DataProduct.asp_ephemeris,
)
query["stix"].filter_for_latest_version()
aux_files = Fido.fetch(query["stix"])

for path in aux_files:
    load_ephemeris_fits_to_cache(path)

print(f"Rows in cache after pre-warming 2 days: {len(STIX_EPHEMERIS_CACHE)}")

# This query now hits the cache directly; no Fido search happens.
roll, _, _ = get_hpc_info(Time("2023-01-02T08:00:00"))
print(f"Pre-warmed lookup roll: {roll}")

###############################################################################
# Point Fido at a local archive (no custom client needed)
# -------------------------------------------------------
#
# The cache is the recommended way to bypass network downloads for known
# time ranges (pre-warm via
# `~stixpy.coordinates.transforms.load_ephemeris_fits_to_cache`). If you
# want a *system-wide* "always look in this local archive instead of the
# online server" override, override
# `~stixpy.net.client.STIXClient.baseurl` to a ``file://`` URL. The class
# attribute is what `~sunpy.net.Fido` reads when it instantiates the client
# with no ``source`` argument, so every `~sunpy.net.Fido.search` afterwards
# — including the one inside
# `~stixpy.coordinates._ephemeris_fetcher.fetch_ephemeris_for_range` —
# resolves files from your local mirror.

if False:  # disabled in the docs build; uncomment locally to redirect Fido
    from stixpy.net.client import STIXClient

    STIXClient.baseurl = "file:///data/solo/fits"  # ← your local mirror root

    # From here on, any get_hpc_info / coordinate-transform call that misses
    # the cache will resolve files from the local directory rather than the
    # public archive. No stixpy code change required.
    roll, _, _ = get_hpc_info(Time("2023-01-04T08:00:00"))

# A per-call alternative (no global state mutation) is to instantiate
# `~stixpy.net.client.STIXClient` directly and bypass Fido entirely — useful
# when you want to feed a specific set of files into the cache and stop
# there. See the pre-warming section above for the recommended pattern.

###############################################################################
# Cache controls
# --------------
#
# A few knobs are exposed on the global cache:
#
# * ``STIX_EPHEMERIS_CACHE.clear()`` — empty the cache.
# * ``STIX_EPHEMERIS_CACHE.bypass_cache_read = True`` — force the next read
#   to miss so Fido re-runs (useful if you suspect cached data is stale or
#   to compare cached vs fresh results in tests).
# * ``len(STIX_EPHEMERIS_CACHE)`` — current row count.
#
# Note: ``bypass_cache_read`` does the *opposite* of pre-warming. To skip
# Fido for a known range, pre-load the relevant files via
# ``load_ephemeris_fits_to_cache`` — the cache itself is the bypass
# mechanism.

STIX_EPHEMERIS_CACHE.bypass_cache_read = True
roll_fresh, _, _ = get_hpc_info(Time("2023-01-02T08:00:00"))  # re-downloads
STIX_EPHEMERIS_CACHE.bypass_cache_read = False
print(f"Forced-fresh roll: {roll_fresh}")

plt.show()
