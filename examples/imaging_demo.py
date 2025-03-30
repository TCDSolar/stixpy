"""
============
Imaging Demo
============

How to create visibility from pixel data and make images.

The example uses ``stixpy`` to obtain STIX pixel data and convert these into visibilities and ``xrayvisim``
to make the images.

Imports
"""

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.map import Map, make_fitswcs_header
from sunpy.time import TimeRange
from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_to_image, vis_to_map
from xrayvision.mem import mem, resistant_mean

from stixpy.calibration.visibility import calibrate_visibility, create_meta_pixels, create_visibility
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.imaging.em import em
from stixpy.map.stix import STIXMap  # noqa
from stixpy.product import Product

logger = logging.getLogger(__name__)

#############################################################################
# Read science file as Product

cpd_sci = Product(
    "http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/solo_L1_stix-sci-xray-cpd_20210923T152015-20210923T152639_V02_2109230030-62447.fits"
)
cpd_sci

#############################################################################
# Read background file as Product

cpd_bkg = Product(
    "http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/solo_L1_stix-sci-xray-cpd_20210923T095923-20210923T113523_V02_2109230083-57078.fits"
)
cpd_bkg

###############################################################################
# Set time and energy ranges which will be considered for the science and the background file

time_range_sci = ["2021-09-23T15:20:00", "2021-09-23T15:23:00"]
time_range_bkg = [
    "2021-09-23T09:00:00",
    "2021-09-23T12:00:00",
]  # Set this range larger than the actual observation time
energy_range = [25, 28] * u.keV

###############################################################################
# Create the meta pixel, A, B, C, D for the science and the background data

meta_pixels_sci = create_meta_pixels(
    cpd_sci, time_range=time_range_sci, energy_range=energy_range, flare_location=[0, 0] * u.arcsec, no_shadowing=True
)

meta_pixels_bkg = create_meta_pixels(
    cpd_bkg, time_range=time_range_bkg, energy_range=energy_range, flare_location=[0, 0] * u.arcsec, no_shadowing=True
)

###############################################################################
# Perform background subtraction

meta_pixels_bkg_subtracted = {
    **meta_pixels_sci,
    "abcd_rate_kev_cm": meta_pixels_sci["abcd_rate_kev_cm"] - meta_pixels_bkg["abcd_rate_kev_cm"],
    "abcd_rate_error_kev_cm": np.sqrt(
        meta_pixels_sci["abcd_rate_error_kev_cm"] ** 2 + meta_pixels_bkg["abcd_rate_error_kev_cm"] ** 2
    ),
}

###############################################################################
# Create visibilities from the meta pixels

vis = create_visibility(meta_pixels_bkg_subtracted)

###############################################################################
# Calibrate the visibilities

cal_vis = calibrate_visibility(vis)

###############################################################################
# Selected detectors 10 to 7

# order by sub-collimator e.g. 10a, 10b, 10c, 9a, 9b, 9c ....
isc_10_7 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]
idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc_10_7)).ravel()

###############################################################################
# Slice the visibilities to detectors 10 - 7

vis10_7 = cal_vis[idx]

###############################################################################
# Set up image parameters

imsize = [512, 512] * u.pixel  # number of pixels of the map to reconstruct
pixel = [10, 10] * u.arcsec / u.pixel  # pixel size in arcsec

###############################################################################
# Make a full disk back projection (inverse transform) map

bp_image = vis_to_image(vis10_7, imsize, pixel_size=pixel)

###############################################################################
# Obtain the necessary metadata to create a sunpy map in the STIXImaging frame

vis_tr = TimeRange(vis.meta["time_range"])
roll, solo_xyz, pointing = get_hpc_info(vis_tr.start, vis_tr.end)
solo = HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")
coord = STIXImaging(0 * u.arcsec, 0 * u.arcsec, obstime=vis_tr.start, obstime_end=vis_tr.end, observer=solo)
header = make_fitswcs_header(
    bp_image, coord, telescope="STIX", observatory="Solar Orbiter", scale=[10, 10] * u.arcsec / u.pix
)
fd_bp_map = Map((bp_image, header))

###############################################################################
# Convert the coordinates and make a map in Helioprojective and rotate so "North" is "up"

hpc_ref = coord.transform_to(Helioprojective(observer=solo, obstime=vis_tr.center))  # Center of STIX pointing in HPC
header_hp = make_fitswcs_header(bp_image, hpc_ref, scale=[10, 10] * u.arcsec / u.pix, rotation_angle=90 * u.deg + roll)
hp_map = Map((bp_image, header_hp))
hp_map_rotated = hp_map.rotate()

###############################################################################
# Plot the both maps


fig = plt.figure(layout="constrained", figsize=(3, 6))
ax = fig.subplot_mosaic(
    [["stix"], ["hpc"]], per_subplot_kw={"stix": {"projection": fd_bp_map}, "hpc": {"projection": hp_map_rotated}}
)
fd_bp_map.plot(axes=ax["stix"])
fd_bp_map.draw_limb()
fd_bp_map.draw_grid()

hp_map_rotated.plot(axes=ax["hpc"])
hp_map_rotated.draw_limb()
hp_map_rotated.draw_grid()

###############################################################################
# Estimate the flare location and plot on top of back projection map. Note the coordinates
# are automatically converted from the STIXImaging to Helioprojective

max_pixel = np.argwhere(fd_bp_map.data == fd_bp_map.data.max()).ravel() * u.pixel
# because WCS axes and array are reversed
max_stix = fd_bp_map.pixel_to_world(max_pixel[1], max_pixel[0])

ax["stix"].plot_coord(max_stix, marker=".", markersize=50, fillstyle="none", color="r", markeredgewidth=2)
ax["hpc"].plot_coord(max_stix, marker=".", markersize=50, fillstyle="none", color="r", markeredgewidth=2)
fig.tight_layout()

################################################################################
# Use estimated flare location to create more accurate visibilities

meta_pixels_sci = create_meta_pixels(
    cpd_sci, time_range=time_range_sci, energy_range=energy_range, flare_location=max_stix, no_shadowing=True
)

meta_pixels_bkg_subtracted = {
    **meta_pixels_sci,
    "abcd_rate_kev_cm": meta_pixels_sci["abcd_rate_kev_cm"] - meta_pixels_bkg["abcd_rate_kev_cm"],
    "abcd_rate_error_kev_cm": np.sqrt(
        meta_pixels_sci["abcd_rate_error_kev_cm"] ** 2 + meta_pixels_bkg["abcd_rate_error_kev_cm"] ** 2
    ),
}

vis = create_visibility(meta_pixels_bkg_subtracted)
cal_vis = calibrate_visibility(vis, flare_location=max_stix)

###############################################################################
# Selected detectors 10 to 3
# order by sub-collimator e.g. 10a, 10b, 10c, 9a, 9b, 9c ....
isc_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]
idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc_10_3)).ravel()

###############################################################################
# Create an ``xrayvsion`` visibility object

cal_vis.meta["offset"] = max_stix
vis10_3 = cal_vis[idx]

###############################################################################
# Set up image parameters

imsize = [129, 129] * u.pixel  # number of pixels of the map to reconstruct
pixel = [2, 2] * u.arcsec / u.pixel  # pixel size in arcsec

###############################################################################
# Create a back projection image with natural weighting

bp_nat = vis_to_image(vis10_3, imsize, pixel_size=pixel)

###############################################################################
# Create a back projection image with uniform weighting

bp_uni = vis_to_image(vis10_3, imsize, pixel_size=pixel, scheme="uniform")

###############################################################################
# Create a `sunpy.map.Map` with back projection

bp_map = vis_to_map(vis10_3, imsize, pixel_size=pixel)

###############################################################################
# Crete a clean image using the clean algorithm `vis_clean`

niter = 200  # number of iterations
gain = 0.1  # gain used in each clean iteration
beam_width = 20.0 * u.arcsec
clean_map, model_map, resid_map = vis_clean(
    vis10_3, imsize, pixel_size=pixel, gain=gain, niter=niter, clean_beam_width=20 * u.arcsec
)

###############################################################################
# Create a sunpy map for the clean image in Helioprojective

header = make_fitswcs_header(
    clean_map.data,
    max_stix.transform_to(Helioprojective(obstime=vis_tr.center, observer=solo)),
    telescope="STIX",
    observatory="Solar Orbiter",
    scale=pixel,
    rotation_angle=90 * u.deg + roll,
)

###############################################################################
# Crete a map using the MEM GE algorithm `mem`

snr_value, _ = resistant_mean((np.abs(vis10_3.visibilities) / vis10_3.amplitude_uncertainty).flatten(), 3)
percent_lambda = 2 / (snr_value**2 + 90)
mem_map = mem(vis10_3, shape=imsize, pixel_size=pixel, percent_lambda=percent_lambda)


###############################################################################
# Crete a map using the EM algorithm `EM`

em_map = em(
    meta_pixels_bkg_subtracted["abcd_rate_kev_cm"],
    cal_vis,
    shape=imsize,
    pixel_size=pixel,
    flare_location=max_stix,
    idx=idx,
)


clean_map = Map((clean_map.data, header)).rotate()
bp_map = Map((bp_nat, header)).rotate()
mem_map = Map((mem_map.data, header)).rotate()
em_map = Map((em_map, header)).rotate()

vmax = max([clean_map.data.max(), mem_map.data.max(), em_map.data.max()])

###############################################################################
# Finally compare the images from each algorithm

fig = plt.figure(figsize=(12, 12))
ax = fig.subplot_mosaic(
    [
        ["bp", "clean"],
        ["mem", "em"],
    ],
    subplot_kw={"projection": clean_map},
)
a = bp_map.plot(axes=ax["bp"])
ax["bp"].set_title("Back Projection")
b = clean_map.plot(axes=ax["clean"])
ax["clean"].set_title("Clean")
c = mem_map.plot(axes=ax["mem"])
ax["mem"].set_title("MEM GE")
d = em_map.plot(axes=ax["em"])
ax["em"].set_title("EM")
fig.colorbar(a, ax=ax.values())
plt.show()
