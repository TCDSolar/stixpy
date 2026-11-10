"""
==================
Spectroscopy Demo
==================

How to make a background-subtracted count spectrum and the matching spectral response
matrix (SRM) from STIX pixel data, ready for spectral fitting with ``sunkit-spex``.

This example uses `~stixpy.product.sources.science.ScienceData.get_data` with
``sunkit_spex_spectrum=True``. The spectrum and SRM follow the IDL routines
``stx_convert_pixel_data`` and ``stx_convert_spectrogram2ospex`` in STIX-GSW, so the result
can be compared directly with an OSPEX analysis.

Imports
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import astropy.units as u

from stixpy.imaging.flare_location import estimate_flare_location
from stixpy.product import Product

#############################################################################
# Read the science file and a background file as Products

cpd_sci = Product(
    "http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/solo_L1_stix-sci-xray-cpd_20210923T152015-20210923T152639_V02_2109230030-62447.fits"
)
cpd_sci

cpd_bkg = Product(
    "http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/solo_L1_stix-sci-xray-cpd_20210923T095923-20210923T113523_V02_2109230083-57078.fits"
)
cpd_bkg

#############################################################################
# Plot the time series in a low and a high energy band to choose the time range to fit.
# Energy ranges can be given in keV; each range takes the energy bins whose centres lie
# inside it.

cpd_sci.plot_timeseries(energy_indices=[[6, 10], [15, 28]] * u.keV)
plt.legend()

#############################################################################
# The SRM includes the transmission of the grids, which depends on where the flare is, so
# estimate the flare location first (see the flare location example for more detail).
#
# Requested times must lie inside the file, which covers 15:20:16 to 15:26:39.

time_range = ["2021-09-23T15:20:30", "2021-09-23T15:23:30"]

flare_location = estimate_flare_location(cpd_sci, time_range=time_range, energy_range=[6, 15] * u.keV)

#############################################################################
# Make the spectrum. The counts are summed over the top 24 detectors and the time range,
# corrected for livetime and the energy look-up table (ELUT), and the background is
# subtracted after scaling it to the same livetime. ``sunkit_spex_systematic_error`` adds
# the standard STIX systematic uncertainty (7% below 7 keV, 5% from 7 to 10 keV and 3%
# above).
#
# The CFL detector (index 8) can not be used for spectroscopy, and the BKG detector
# (index 9) only on its own, as in STIX-GSW.

spec = cpd_sci.get_data(
    time_indices=time_range,
    detector_indices="top24",
    bkg=cpd_bkg,
    flare_location=flare_location,
    elut_correction=True,
    sunkit_spex_spectrum=True,
    sunkit_spex_systematic_error=True,
)
spec

#############################################################################
# The spectrum holds the counts and their uncertainty, with the count energy bin edges as
# the spectral axis. Everything needed to fit it is in the metadata: the exposure time, the
# SRM with its photon energy edges, the geometric area and the Sun-spacecraft distance.

print(spec.meta["exposure_time"])
print(spec.meta["geo_area"])
print(spec.meta["distance"])
print(spec.meta["srm"].shape, spec.meta["ph_axis"].shape)

#############################################################################
# Plot the background-subtracted count rate spectrum.

ct_edges = spec.spectral_axis.bin_edges
ct_width = np.diff(ct_edges)
exposure = spec.meta["exposure_time"]

rate = spec.data / exposure / ct_width
rate_err = spec.uncertainty.array / exposure / ct_width
ct_mid = ct_edges[:-1] + ct_width / 2

fig, ax = plt.subplots()
ax.stairs(rate.value, ct_edges.value, label="Top 24 detectors")
ax.errorbar(ct_mid.value, rate.value, yerr=rate_err.value, fmt="none", color="C0")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [keV]")
ax.set_ylabel(f"Count rate [{rate.unit}]")
ax.legend()

#############################################################################
# Plot the SRM. Each row is a photon energy bin and each column a count energy bin, in
# counts per photon. The photopeak is the diagonal; the fainter lines below it are the Cd
# and Te escape peaks, and hole tailing spreads counts to lower energies. By default the
# photon axis starts at 3.5 keV (``srm_e_min``).

srm = spec.meta["srm"]
ph_edges = spec.meta["ph_axis"]

fig, ax = plt.subplots()
mesh = ax.pcolormesh(ct_edges.value, ph_edges.value, srm, norm=LogNorm(vmin=srm.max() * 1e-5, vmax=srm.max()))
ax.set_xlabel("Count energy [keV]")
ax.set_ylabel("Photon energy [keV]")
ax.set_ylim(ph_edges[0].value, 100)
fig.colorbar(mesh, label="Counts / photon")

#############################################################################
# Giving several time ranges returns one spectrum per range as an
# `~ndcube.NDCubeSequence`. Each spectrum has the SRM for its own attenuator (RCR) state,
# so a sequence can cross an attenuator insertion, but a single range can not.

time_ranges = [
    ["2021-09-23T15:20:30", "2021-09-23T15:21:30"],
    ["2021-09-23T15:21:30", "2021-09-23T15:22:30"],
    ["2021-09-23T15:22:30", "2021-09-23T15:23:30"],
]

spec_seq = cpd_sci.get_data(
    time_indices=time_ranges,
    detector_indices="top24",
    bkg=cpd_bkg,
    flare_location=flare_location,
    sunkit_spex_spectrum=True,
)

spec_seq_bkg = cpd_sci.get_data(
    time_indices=time_ranges,
    detector_indices="bkg",
    bkg=cpd_bkg,
    flare_location=flare_location,
    sunkit_spex_spectrum=True,
)

n = len(time_ranges)
blues = plt.colormaps["Blues"](np.linspace(0.45, 0.9, n))
oranges = plt.colormaps["Oranges"](np.linspace(0.45, 0.9, n))

fig, ax = plt.subplots()
for seq, name, colours, ls in [
    (spec_seq, "Top 24", blues, "-"),
    (spec_seq_bkg, "BKG", oranges, "--"),
]:
    for s, (start, end), c in zip(seq.data, time_ranges, colours):
        edges = s.spectral_axis.bin_edges
        s_rate = s.data / s.meta["exposure_time"] / np.diff(edges)
        ax.stairs(
            s_rate.value,
            edges.value,
            baseline=None,
            color=c,
            linestyle=ls,
            label=f"{name} {start[11:19]} - {end[11:19]}",
        )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [keV]")
ax.set_ylabel("Count rate [ct / (keV s)]")
ax.legend(fontsize="small", ncol=2)


time_range = ["2021-09-23T15:20:30", "2021-09-23T15:23:30"]

spec_sub = cpd_sci.get_data(
    time_indices=time_range,
    detector_indices="top24",
    bkg=cpd_bkg,
    flare_location=flare_location,
    sunkit_spex_spectrum=True,
)

spec_nosub = cpd_sci.get_data(
    time_indices=time_range,
    detector_indices="top24",
    flare_location=flare_location,
    sunkit_spex_spectrum=True,
)

edges = spec_sub.spectral_axis.bin_edges
ct_width = np.diff(edges)
ct_mid = edges[:-1] + ct_width / 2

rate_sub = spec_sub.data * spec_sub.unit / spec_sub.meta["exposure_time"] / ct_width
err_sub = spec_sub.uncertainty.array * spec_sub.unit / spec_sub.meta["exposure_time"] / ct_width
rate_nosub = spec_nosub.data * spec_nosub.unit / spec_nosub.meta["exposure_time"] / ct_width
err_nosub = spec_nosub.uncertainty.array * spec_nosub.unit / spec_nosub.meta["exposure_time"] / ct_width

with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(rate_nosub > 0, rate_sub / rate_nosub, np.nan).to(u.one)
    ratio_err = np.where(rate_nosub > 0, np.sqrt(err_sub**2 + (ratio * err_nosub) ** 2) / rate_nosub, np.nan).to(u.one)

rate_unit = rate_sub.unit

fig, (ax, ax_ratio) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)

ax.stairs(rate_sub.to_value(rate_unit), edges.to_value(u.keV), baseline=None, color="C0", label="BKG subtracted")
ax.errorbar(
    ct_mid.to_value(u.keV), rate_sub.to_value(rate_unit), yerr=err_sub.to_value(rate_unit), fmt="none", color="C0"
)
ax.stairs(
    rate_nosub.to_value(rate_unit),
    edges.to_value(u.keV),
    baseline=None,
    color="C1",
    linestyle="--",
    label="No BKG subtraction",
)
ax.errorbar(
    ct_mid.to_value(u.keV), rate_nosub.to_value(rate_unit), yerr=err_nosub.to_value(rate_unit), fmt="none", color="C1"
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(edges[0].to_value(u.keV), edges[-1].to_value(u.keV))
ax.set_ylabel(f"Count rate [{rate_unit}]")
ax.set_title(f"Top 24 detectors, {time_range[0][11:19]} - {time_range[1][11:19]}")
ax.legend()

ax_ratio.stairs(ratio.value, edges.to_value(u.keV), baseline=None, color="C0")
ax_ratio.errorbar(ct_mid.to_value(u.keV), ratio.value, yerr=ratio_err.value, fmt="none", color="C0")
ax_ratio.axhline(1, color="0.5", linewidth=0.8, linestyle=":")
ax_ratio.set_ylim(0, 1.1)
ax_ratio.set_xlabel("Energy [keV]")
ax_ratio.set_ylabel("Sub / no sub")

#############################################################################
# With ``sunkit_spex_detector_sum=False`` each detector gets its own spectrum and SRM,
# returned as an `~ndcube.NDCollection` keyed by detector index. This is useful to check
# the detectors agree before summing them.

spec_dets = cpd_sci.get_data(
    time_indices=time_range,
    detector_indices=[5, 6, 7],
    bkg=cpd_bkg,
    flare_location=flare_location,
    sunkit_spex_spectrum=True,
    sunkit_spex_detector_sum=False,
)

fig, ax = plt.subplots()
for det, s in spec_dets.items():
    edges = s.spectral_axis.bin_edges
    s_rate = s.data / s.meta["exposure_time"] / np.diff(edges)
    ax.stairs(s_rate.value, edges.value, label=f"Detector {det}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [keV]")
ax.set_ylabel("Count rate [ct / (keV s)]")
ax.legend()

plt.show()
