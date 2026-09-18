"""
======================
Flare Location Demo
======================

How to estimate the location of a flare from STIX pixel data.

This example uses `~stixpy.imaging.flare_location.estimate_flare_location` to make a full disk
back projection image and take the location of the brightest pixel as an estimate of the flare
location. This is the same approach as the IDL routine ``stx_estimate_flare_location``.

Imports
"""

import matplotlib.pyplot as plt

import astropy.units as u

from sunpy.coordinates import Helioprojective, SphericalScreen, get_earth
from sunpy.time import TimeRange

from stixpy.imaging.flare_location import estimate_flare_location
from stixpy.product import Product

#############################################################################
# Read the science file as a Product

cpd_sci = Product(
    "http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/solo_L1_stix-sci-xray-cpd_20210923T152015-20210923T152639_V02_2109230030-62447.fits"
)
cpd_sci

#############################################################################
# Set the time and energy range to use. A relatively low energy range is used here as the
# thermal emission is normally the brightest part of the flare, and a time range around the
# peak of the flare gives the best signal to noise.

time_range = TimeRange("2021-09-23T15:20:00", "2021-09-23T15:23:00")
energy_range = [6, 15] * u.keV

#############################################################################
# Estimate the flare location. Passing ``plot=True`` also plots the back projection images in
# the STIX imaging and Helioprojective frames with the estimated location marked.

results = estimate_flare_location(cpd_sci, time_range=time_range, energy_range=energy_range, plot=True)

#############################################################################
# The results are returned as a dictionary. ``stx`` and ``hpc`` are the estimated location in
# the STIX imaging and Helioprojective frames respectively, ``vis_tr`` is the time range
# actually covered by the visibilities, which can differ slightly from the requested time range
# as only complete time bins are used.

flare_stix = results["stx"]
flare_hpc = results["hpc"]

print(flare_stix)
print(flare_hpc)
print(results["vis_tr"])

#############################################################################
# The sidelobes ratio gives an indication of how reliable the estimate is. The back projection
# image of a single compact source has one clear peak, so a large secondary peak suggests the
# image is dominated by sidelobes and the estimated location should be treated with caution.
# A warning is raised if the ratio is greater than or equal to 0.9.

print(f"Sidelobes ratio: {results['sidelobes_ratio']:.3f}")

#############################################################################
# The location is returned as a `~astropy.coordinates.SkyCoord` so it can be transformed to
# other frames, for example to compare with observations from Earth. Note the Helioprojective
# coordinates are as seen from Solar Orbiter, so a flare on the disk as seen by STIX may be
# behind the limb as seen from Earth, hence the use of a
# `~sunpy.coordinates.screens.SphericalScreen`.

earth_hpc = Helioprojective(observer=get_earth(results["vis_tr"].center), obstime=results["vis_tr"].center)
with SphericalScreen(flare_hpc.observer, only_off_disk=True):
    flare_earth = flare_hpc.transform_to(earth_hpc)
    flare_hgs = flare_hpc.transform_to("heliographic_stonyhurst")

print(flare_earth)
print(flare_hgs)

plt.show()
