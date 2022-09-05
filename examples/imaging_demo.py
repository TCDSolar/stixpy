import logging
import astropy.units as u
import numpy as np


from stixcore.imaging.em import em
from stixcore.calibration.visibility import calibrate_visibility, create_visibility, \
    correct_phase_projection
from stixpy.science import ScienceData
from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_to_image
from xrayvision.mem import mem
from xrayvision.visibility import Visibility

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

cpd = ScienceData.from_fits('/Users/shane/Downloads/'
                            'solo_L1_stix-sci-xray-cpd_20200607T213709-20200607T215208'
                            '_V01_1178428688-49155.fits')


time_range = ['2020-06-07T21:39:00', '2020-06-07T21:42:49']
energy_range = [6, 10]
flare_xy = [-1625, -700] * u.arcsec

# TODO rename doesn't really create a complex vis just the real and imag parts
vis = create_visibility(cpd, time_range=time_range,
                        energy_range=energy_range, phase_center=flare_xy, no_shadowing=False)


# TODO rename
cal_vis = calibrate_visibility(vis)

cal_vis = correct_phase_projection(cal_vis, flare_xy)

# order by sub-collimator e.g 10a, 10b, 10c, 9a, 9b, 9c ....
isc_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]
idx = np.searchsorted(cal_vis.isc, isc_10_3)

imsize = [129, 129]*u.pixel  # number of pixels of the map to reconstruct
pixel = [2, 2]*u.arcsec   # pixel size in aresec

stix_vis = Visibility(vis=cal_vis.obsvis[idx], u=cal_vis.u[idx], v=cal_vis.v[idx])
skeys = stix_vis.__dict__.keys()
for k, v in cal_vis.__dict__.items():
    if k not in skeys:
        setattr(stix_vis, k, v[idx])

# back projection (inverse transform) natural weighting
bp_nat = vis_to_image(stix_vis, imsize, pixel_size=pixel)

# back project (inverse transform) uniform weighting
bp_uni = vis_to_image(stix_vis, imsize, pixel_size=pixel, natural=False)

# Clean
niter = 60  # number of iterations
gain = 0.1  # gain used in each clean iteration
beam_width = 20. * u.arcsec
clean_map, model_map, resid_map = vis_clean(stix_vis, imsize, pixel=pixel, gain=gain, niter=niter,
                                            clean_beam_width=20*u.arcsec, natural=False)

mem_map = mem(stix_vis, shape=imsize, pixel=pixel)

vis = create_visibility(cpd, time_range=time_range, energy_range=energy_range, phase_center=flare_xy)

em_map = em(vis['rate'], stix_vis, shape=imsize, pixel_size=pixel, flare_xy=flare_xy, idx=idx)
print(em_map)
