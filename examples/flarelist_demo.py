"""
===============
Flare List Demo
===============

How to search, combine and explore STIX L3 flare list products.

This example uses :class:`~stixpy.net.client.STIXClient` (a Fido client) to find
three months of L3 flare list files, loads each one as a
:class:`~stixpy.product.sources.flarelist.FlareList` product, concatenates them
into a single catalogue, plots the peak count time series, then filters for
X-class flares close to the Earth-facing solar limb and plots their locations.

Imports
"""

import matplotlib.pyplot as plt
import numpy as np

from sunpy.net import attrs as a

from stixpy.net.client import STIXClient
from stixpy.product import Product

#############################################################################
# Search three months of flare list data.
#
# The flare list test products are currently hosted on a separate URL from the
# rest of the STIX archive, so we instantiate :class:`~stixpy.net.client.STIXClient`
# with that ``source``.

client = STIXClient(source="https://pub099.cs.technik.fhnw.ch/testing/p311_comp_test")

results = client.search(
    a.Time("2024-09-01T01:00:00", "2024-11-30T23:59"),
    a.Instrument.stix,
    a.stix.DataType.flarelist,
)
print(results)

#############################################################################
# Download the matching files and load each one as a FlareList product

paths = client.fetch(results)
products = [Product(p) for p in paths]

#############################################################################
# Concatenate the monthly products into a single FlareList covering Sep–Nov 2024

flares = products[0].concatenate(products[1:])
print(flares)

#############################################################################
# Plot the timeline of peak counts per energy band across all three months

flares.plot_timeseries()
plt.title(f"STIX peak counts — {len(flares)} flares (Sep–Nov 2024)")
plt.tight_layout()

#############################################################################
# Filter the catalogue for M- and X-class flares close to the Earth-facing limb.
#
# Seen from Earth the solar limb maps to HGS longitude ±90°. We keep flares
# whose absolute HGS longitude exceeds 70° and whose GOES classification starts
# with ``"M"`` or ``"X"``.

lon = flares.flare_location_hgs.lon.deg
is_strong = np.array([str(g).strip().upper().startswith(("M", "X")) for g in flares.goes_class])
near_limb = np.isfinite(lon) & (np.abs(lon) > 70)
mask = is_strong & near_limb
print(f"Selected {mask.sum()} M/X-class limb flares of {len(flares)} total.")

#############################################################################
# A boolean mask can be applied directly to the FlareList; the result is a new
# FlareList with the matching rows. Plot the filtered locations from Earth's
# point of view.

filtered = flares[mask]
print(filtered)

filtered.plot_timeseries()
plt.title(f"Peak counts — {len(filtered)} M/X-class limb flares")
plt.tight_layout()

filtered.plot_locations(observer="earth")
plt.tight_layout()

#############################################################################
# Masking is non-destructive: ``flares`` is the full catalogue, so we can
# apply a completely independent filter to the same object — here, strong
# flares within ±30° of disk centre as seen from Earth.

near_disk = np.isfinite(lon) & (np.abs(lon) < 30)
disk_flares = flares[is_strong & near_disk]
print(f"Selected {len(disk_flares)} M/X-class on-disk flares of {len(flares)} total.")

disk_flares.plot_timeseries()
plt.title(f"Peak counts — {len(disk_flares)} M/X-class on-disk flares")
plt.tight_layout()

disk_flares.plot_locations(observer="earth")
plt.tight_layout()
plt.show()
