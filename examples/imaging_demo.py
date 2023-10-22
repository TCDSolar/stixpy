import logging
import astropy.units as u
import numpy as np


from stixpy.imaging.em import em
from stixpy.calibration.visibility import calibrate_visibility, create_meta_pixels, \
    correct_phase_projection, create_visibility, get_visibility_info_giordano
from stixpy.science import ScienceData
from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_to_image, vis_to_map
from xrayvision.mem import mem
from xrayvision.visibility import Visibility

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

# cpd = ScienceData.from_fits('/Users/shane/Downloads/'
#                             'solo_L1_stix-sci-xray-cpd_20200607T213709-20200607T215208'
#                             '_V01_1178428688-49155.fits')

cpd = ScienceData.from_fits('/Users/sm/Downloads/solo_L1_stix-sci-xray-cpd_20200607T213709-20200607T215208_V01_1178428688-49155.fits')


time_range = ['2020-06-07T21:38:58', '2020-06-07T21:43:00']
energy_range = [6, 10]
flare_xy = [-1625, -700] * u.arcsec

meta_pixels = create_meta_pixels(cpd, time_range=time_range,
                                 energy_range=energy_range, phase_center=[0, 0] * u.arcsec, no_shadowing=True)

vis = create_visibility(meta_pixels)

uu, vv = get_visibility_info_giordano()
vis.u = uu
vis.v = vv

cal_vis = calibrate_visibility(vis)

# cal_vis = correct_phase_projection(cal_vis, [0,0] * u.arcsec)

# order by sub-collimator e.g 10a, 10b, 10c, 9a, 9b, 9c ....
isc_10_7 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]
isc_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]
idx = np.argwhere(np.isin(cal_vis.isc, isc_10_7)).ravel()

imsize = [512, 512]*u.pixel  # number of pixels of the map to reconstruct
pixel = [10, 10]*u.arcsec   # pixel size in aresec

stix_vis = Visibility(vis=cal_vis.obsvis[idx], u=cal_vis.u[idx], v=cal_vis.v[idx])
skeys = stix_vis.__dict__.keys()
for k, v in cal_vis.__dict__.items():
    if k not in skeys:
        setattr(stix_vis, k, v[idx])

# back projection (inverse transform) natural weighting
bp_nat = vis_to_image(stix_vis, imsize, pixel_size=pixel, natural=False)
bp_map = vis_to_map(stix_vis, imsize, pixel_size=pixel, natural=False)
bp_map.meta['rsun_obs'] = cpd.meta['rsun_arc']

max_pixel = np.argwhere(bp_map.data == bp_map.data.max()).ravel() *u.pixel
# because WCS axes are follow the reverse ordering
max_hpc = bp_map.pixel_to_world(max_pixel[1], max_pixel[0])

# TODO rename doesn't really create a complex vis just the real and imag parts
meta_pixels = create_meta_pixels(cpd, time_range=time_range, energy_range=energy_range,
                                 phase_center=[max_hpc.Tx, max_hpc.Ty], no_shadowing=False)

vis = create_visibility(meta_pixels)

uu, vv = get_visibility_info_giordano()
vis.u = uu
vis.v = vv

cal_vis = calibrate_visibility(vis)

cal_vis = correct_phase_projection(cal_vis, [max_hpc.Tx, max_hpc.Ty])

# order by sub-collimator e.g 10a, 10b, 10c, 9a, 9b, 9c ....
isc_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]
idx = np.argwhere(np.isin(cal_vis.isc, isc_10_3)).ravel()

imsize = [129, 129]*u.pixel  # number of pixels of the map to reconstruct
pixel = [2, 2]*u.arcsec   # pixel size in aresec

stix_vis1 = Visibility(vis=cal_vis.obsvis[idx], u=cal_vis.u[idx], v=cal_vis.v[idx],
                      offset=np.array([max_hpc.Tx, max_hpc.Ty]).reshape(-1)*u.arcsec)
skeys = stix_vis1.__dict__.keys()
for k, v in cal_vis.__dict__.items():
    if k not in skeys:
        setattr(stix_vis1, k, v[idx])

# back projection (inverse transform) natural weighting
bp_nat = vis_to_image(stix_vis1, imsize, pixel_size=pixel)

# back project (inverse transform) uniform weighting
bp_uni = vis_to_image(stix_vis1, imsize, pixel_size=pixel, natural=False)

bp_map = vis_to_map(stix_vis1, imsize, pixel_size=pixel)

# Clean
niter = 60  # number of iterations
gain = 0.1  # gain used in each clean iteration
beam_width = 20. * u.arcsec
clean_map, model_map, resid_map = vis_clean(stix_vis1, imsize, pixel=pixel, gain=gain, niter=niter,
                                            clean_beam_width=20*u.arcsec, natural=False)

mem_map = mem(stix_vis1, shape=imsize, pixel=pixel)

vis = create_visibility(cpd, time_range=time_range, energy_range=energy_range, phase_center=flare_xy)

em_map = em(vis['rate'], stix_vis, shape=imsize, pixel_size=pixel, flare_xy=flare_xy, idx=idx)
print(em_map)
