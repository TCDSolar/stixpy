import astropy.units as u
import numpy as np

from stixcore.calibration.visibility import calibrate_visibility, create_visibility
from stixpy.science import ScienceData
from xrayvision.clean import clean_wrapper
from xrayvision.imaging import back_project
from xrayvision.transform import idft_map
from xrayvision.visibility import Visibility

cpd = ScienceData.from_fits('/Users/shane/Downloads/solo_L1_stix-sci-xray-cpd_20200607T213709-20200607T215208_V01_1178428688-49155.fits')

time_range = ['2020-06-07T21:39:00', '2020-06-07T21:42:49']
energy_range = [6, 10]
flare_xy = [-1625, -700] * u.arcsec


vis = create_visibility(cpd, time_range=time_range, energy_range=energy_range, phase_center=flare_xy)

cal_vis = calibrate_visibility(vis)

# Phase projection correction
L1 = 550 * u.mm
L2 = 47 * u.mm
# TOdo check how this works out to degress
cal_vis.phase -= (flare_xy[1].to_value('arcsec') * 360. * np.pi
                  / (180. * 3600. * 8.8*u.mm) * (L2 + L1 / 2.0))*u.deg

cal_vis.u *= -cal_vis.phase_sense
cal_vis.v *= -cal_vis.phase_sense

# Compute real and imaginary part of the visibility
cal_vis.obsvis = cal_vis.amplitude * (np.cos(cal_vis.phase.to('rad')) + np.sin(cal_vis.phase.to('rad'))*1j)

# Add phase factor for shifting the flare_xy
map_center = flare_xy - [26.1, 58.2] * u.arcsec
# Subtract Frederic's mean shift values

# TODO check the u, v why is flare_xy[1] used with u (prob stix vs spacecraft orientation
phase_mapcenter = -2 * np.pi * (map_center[1] * cal_vis.u - map_center[0] * cal_vis.v) * u.rad
cal_vis.obsvis *= (np.cos(phase_mapcenter)+np.sin(phase_mapcenter) *1j)

isc_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]
idx = np.searchsorted(cal_vis.isc, isc_10_3)

imsize = [129, 129]*u.pixel  # number of pixels of the map to reconstruct
pixel = [2.,2.]*u.arcsec   # pixel size in aresec

stix_vis = Visibility(uv=np.vstack([cal_vis.u[idx], cal_vis.v[idx]]), vis=cal_vis.obsvis[idx],
                      xyoffset=flare_xy, pixel_size=pixel)

# back projection (inverse transform) natural weighting
bp = back_project(stix_vis, imsize, pixel_size=pixel)

# back project (inverse transform) uniform weighting


# Clean
niter = 60  # number of iterations
gain = 0.1  # gain used in each clean iteration
beam_width = 20. * u.arcsec
dirty_map = clean_wrapper(stix_vis, imsize, pixel=pixel,
                          gain=gain, niter=niter, clean_beam_width=20*u.arcsec)
print(dirty_map.shape)
