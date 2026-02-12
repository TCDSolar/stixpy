
import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, frames
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



__all__ = [
    "estimate_flare_location",
]

def estimate_flare_location(sci_file,time_range_sci):

    energy_range = [6.,10.] * u.keV
    imsize = [512,512] * u.pixel
    pixel = [10, 10] * u.arcsec / u.pixel

    cpd_sci = Product(sci_file)

    meta_pixels_sci = create_meta_pixels(
        cpd_sci, time_range=time_range_sci, energy_range=energy_range, flare_location=[0, 0] * u.arcsec, no_shadowing=True
    )

    vis = create_visibility(meta_pixels_sci)

    vis_tr = TimeRange(vis.meta["time_range"])
    roll, solo_xyz, pointing = get_hpc_info(vis_tr.start, vis_tr.end)
    solo = HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")
    center_hpc = SkyCoord(0 * u.deg, 0 * u.deg, frame=Helioprojective(obstime=vis_tr.center, observer=solo))

    cal_vis = calibrate_visibility(vis, flare_location=center_hpc)

    # order by sub-collimator e.g. 10a, 10b, 10c, 9a, 9b, 9c ....
    isc_10_7 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]
    idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc_10_7)).ravel()

    vis10_7 = cal_vis[idx]

    bp_image = vis_to_image(vis10_7, imsize, pixel_size=pixel)

    vis_tr = TimeRange(vis.meta["time_range"])
    roll, solo_xyz, pointing = get_hpc_info(vis_tr.start, vis_tr.end)
    solo = HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")
    coord_stix = center_hpc.transform_to(STIXImaging(obstime=vis_tr.start, obstime_end=vis_tr.end, observer=solo))
    header = make_fitswcs_header(
        bp_image, coord_stix, telescope="STIX", observatory="Solar Orbiter", scale=[10, 10] * u.arcsec / u.pix
    )
    fd_bp_map = Map((bp_image, header))

    max_pixel = np.argwhere(fd_bp_map.data == fd_bp_map.data.max()).ravel() * u.pixel
    # because WCS axes and array are reversed
    max_stix = fd_bp_map.pixel_to_world(max_pixel[1], max_pixel[0])

    # hpc_x = max_stix.transform_to(frames.Helioprojective).Tx.value
    # hpc_y = max_stix.transform_to(frames.Helioprojective).Ty.value

    hpc_x = max_stix.Tx.value
    hpc_y = max_stix.Ty.value

    return np.array([hpc_x,hpc_y])
