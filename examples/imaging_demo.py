import logging
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.map import make_fitswcs_header, Map

from stixpy.frames import get_hpc_info
from stixpy.imaging.em import em
from stixpy.calibration.visibility import calibrate_visibility, create_meta_pixels, create_visibility, get_visibility_info_giordano
from stixpy.product import Product

from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_to_image, vis_to_map
from xrayvision.mem import mem
from xrayvision.visibility import Visibility

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


cpd = Product('http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/solo_L1_stix-sci-xray-cpd_20210923T152015-20210923T152639_V02_2109230030-62447.fits')
time_range = ['2021-09-23T15:21:00', '2021-09-23T15:24:00']
energy_range = [28, 40]


meta_pixels = create_meta_pixels(cpd, time_range=time_range,
                                 energy_range=energy_range, phase_center=[0, 0] * u.arcsec, no_shadowing=True)

vis = create_visibility(meta_pixels)

uu, vv = get_visibility_info_giordano()
vis.u = uu
vis.v = vv

cal_vis = calibrate_visibility(vis)

# order by sub-collimator e.g 10a, 10b, 10c, 9a, 9b, 9c ....
isc_10_7 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]
idx = np.argwhere(np.isin(cal_vis.isc, isc_10_7)).ravel()

imsize = [512, 512]*u.pixel  # number of pixels of the map to reconstruct
pixel = [10, 10]*u.arcsec   # pixel size in aresec

stix_vis = Visibility(vis=cal_vis.obsvis[idx], u=cal_vis.u[idx], v=cal_vis.v[idx])
skeys = stix_vis.__dict__.keys()
for k, v in cal_vis.__dict__.items():
    if k not in skeys:
        setattr(stix_vis, k, v[idx])

# back projection (inverse transform) natural weighting
fd_bp_nat = vis_to_image(stix_vis, imsize, pixel_size=pixel)
fd_bp_map = vis_to_map(stix_vis, imsize, pixel_size=pixel)
fd_bp_map.meta['rsun_obs'] = cpd.meta['rsun_arc']

max_pixel = np.argwhere(fd_bp_map.data == fd_bp_map.data.max()).ravel() *u.pixel
# because WCS axes are reverse order
max_hpc = fd_bp_map.pixel_to_world(max_pixel[1], max_pixel[0])

meta_pixels = create_meta_pixels(cpd, time_range=time_range, energy_range=energy_range,
                                 phase_center=[max_hpc.Tx, max_hpc.Ty], no_shadowing=False)

vis = create_visibility(meta_pixels)

uu, vv = get_visibility_info_giordano()
vis.u = uu
vis.v = vv

cal_vis = calibrate_visibility(vis, [max_hpc.Tx, max_hpc.Ty])

# order by sub-collimator e.g 10a, 10b, 10c, 9a, 9b, 9c ....
isc_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]
idx = np.argwhere(np.isin(cal_vis.isc, isc_10_3)).ravel()

imsize = [129, 129]*u.pixel  # number of pixels of the map to reconstruct
pixel = [2, 2]*u.arcsec   # pixel size in aresec

stix_vis1 = Visibility(vis=cal_vis.obsvis[idx], u=cal_vis.u[idx], v=cal_vis.v[idx],
                       offset=np.array([max_hpc.Tx.value, max_hpc.Ty.value])*u.arcsec)
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
niter = 200  # number of iterations
gain = 0.1  # gain used in each clean iteration
beam_width = 20. * u.arcsec
clean_map, model_map, resid_map = vis_clean(stix_vis1, imsize, pixel=pixel, gain=gain, niter=niter,
                                            clean_beam_width=20*u.arcsec)

mem_map = mem(stix_vis1, shape=imsize, pixel=pixel)

stix_vis1 = Visibility(vis=cal_vis.obsvis[idx], u=cal_vis.u[idx], v=cal_vis.v[idx],
                       center=np.array([max_hpc.Tx.value, max_hpc.Ty.value])*u.arcsec)
skeys = stix_vis1.__dict__.keys()
for k, v in cal_vis.__dict__.items():
    if k not in skeys:
        setattr(stix_vis1, k, v[idx])

em_map = em(meta_pixels['abcd_rate_kev_cm'], stix_vis1, shape=imsize, pixel_size=pixel, flare_xy=
            np.array([max_hpc.Tx.value, max_hpc.Ty.value])*u.arcsec, idx=idx)

vmax = max([clean_map.data.max(), mem_map.data.max(), em_map.value.max()])

fig = plt.figure()
axes = fig.add_subplot(projection=fd_bp_map)
fd_bp_map.plot(axes=axes)
axes.plot_coord(max_hpc, marker='.', markersize=50, fillstyle='none', color='r')
axes.set_xlabel('STIX Y')
axes.set_ylabel('STIX X')
axes.set_title(f'STIX {" ".join(time_range)} {"-".join([str(e) for e in energy_range])} keV')


fig, axes = plt.subplots(2, 2)
a = axes[0, 0].imshow(bp_nat.value)
axes[0, 0].set_title('Back Projection')
fig.colorbar(a)
b = axes[1, 0].imshow(clean_map.data, vmin=0, vmax=vmax)
axes[1,0].set_title('Clean')
fig.colorbar(b)
c = axes[0, 1].imshow(mem_map.data, vmin=0, vmax=vmax)
axes[0,1].set_title('MEM GE')
fig.colorbar(c)
d = axes[1, 1].imshow(em_map.value, vmin=0, vmax=vmax)
axes[1,1].set_title('EM')
fig.colorbar(d)
fig.tight_layout()
print(1)


roll, pos, ref = get_hpc_info(Time(time_range[0]))
solo_coord = SkyCoord(*pos, frame='heliocentricearthecliptic', representation_type='cartesian', obstime=Time(time_range[0]))
ref_coord = SkyCoord(ref[0], ref[1], obstime=Time(time_range[0]), observer=solo_coord, frame='helioprojective')
header = make_fitswcs_header(fd_bp_nat.value, ref_coord, rotation_angle=-roll+90*u.deg, scale=[10,10]*u.arcsec/u.pix)
stix_hpc_map = Map(fd_bp_nat.value, header)
print(1)
