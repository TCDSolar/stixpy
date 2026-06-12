from stixpy.map.stix import STIXMap  
from stixpy.product import Product
from stixpy.product.sources.science import CompressedPixelData 
from stixpy.calibration.visibility import calibrate_visibility, create_meta_pixels, create_visibility
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from xrayvision.imaging import vis_to_image, vis_to_map

from sunpy.time import TimeRange, parse_time
from sunpy.coordinates import frames, SphericalScreen
import sunpy.map

import matplotlib.pyplot as plt
from astropy import units as u 
from astropy.coordinates import SkyCoord
import numpy as np 
import sunpy.sun.constants as sun_const
import warnings

__all__ = [
    "stx_estimate_flare_location",
]

def get_rsun_obs(observer):
    """
    Get the observed radius of the Sun from an observer location.
    """

    rsun_obs = ((sun_const.radius / (observer.spherical.distance - sun_const.radius)).decompose()
                    * u.radian).to(u.arcsec)
    return rsun_obs


def stx_estimate_flare_location(cpd_sci, time_range, energy_range=None, plot=False):
    """
    Estimate the flare location using STIX imaging data.

    This function processes the imaging data from the STIX instrument on Solar Orbiter to estimate the location of a solar flare.
    It is based on the IDL software `stx_estimate_flare_location`. 

    It creates back-projected images in both STIX imaging coordinates and Helioprojective coordinates, and finds the maximum location of the pixel.

    Optionally, it plots the results showing the maximum pixel locations in both coordinate systems.

    Parameters
    ----------
    pixel_path : str
        Path to the STIX pixel data product file.
    time_range : `sunpy.time.TimeRange`
        The time range over which to estimate the flare location.
    energy_range : `astropy.units.Quantity`
        The energy range (e.g., in keV) for the analysis.
    plot : bool, optional
        If True, the function plots the back-projected images in both STIX and Helioprojective frames. 
        Default is False.

    Returns
    -------
    max_stix : `astropy.coordinates.SkyCoord`
        The estimated flare location in STIX imaging coordinates.
    max_hpc : `astropy.coordinates.SkyCoord`
        The estimated flare location in Helioprojective Cartesian coordinates.

    Notes
    -----
    The function involves the following steps:
    - Reading STIX pixel data and generating meta pixels for a given time and energy range.
    - Creating visibility data from the meta pixels.
    - Obtaining solar observer coordinates and converting them to the Heliographic Stonyhurst frame.
    - Creating a back-projected image from the visibility data.
    - Transforming the coordinates of the maximum pixel in the image to Helioprojective coordinates.
    - Optionally, plotting the back-projected images and marking the estimated flare locations.
    
    """

    if energy_range is None:
        energy_range = [6,15]*u.keV 

    if not isinstance(cpd_sci,CompressedPixelData):
        cpd_sci = Product(cpd_sci)

    meta_pixels_sci = create_meta_pixels(cpd_sci, 
                                         time_range=time_range, 
                                         energy_range=energy_range, 
                                         flare_location=[0, 0] * u.arcsec, 
                                         no_shadowing=True)

    # create visibilities
    vis = create_visibility(meta_pixels_sci)
    vis_tr = TimeRange(vis.meta["time_range"])

    roll, solo_xyz, pointing = get_hpc_info(vis_tr.start, vis_tr.end)
    solo = frames.HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")

    center_map = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=frames.Helioprojective(observer=solo, obstime=solo.obstime))
    center_coord = center_map.transform_to(STIXImaging(obstime=vis_tr.start, obstime_end=vis_tr.end, observer=solo))
    
    # get calibrated visibilities - use center of Sun as phase center
    cal_vis = calibrate_visibility(vis, flare_location=center_coord)
    
    # order by sub-collimator e.g. 10a, 10b, 10c, 9a, 9b, 9c ....
    isc_10_7 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]
    idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc_10_7)).ravel()

    # only use subcolimators 7 - 10
    vis10_7 = cal_vis[idx]
    

    # set up image size
    imsize = [512, 512] * u.pixel  

    # to make sure the full Sun is within FOV - the 2.6 is taken to be the same as the IDL software
    pixel = get_rsun_obs(solo) * 2.6 / imsize 


    # get back projection image
    bp_image = vis_to_image(vis10_7, imsize, pixel_size=pixel)


    # Make a sunpy map from the bp_image, in STIX imaging frame
    header = sunpy.map.make_fitswcs_header(
        bp_image, center_coord, telescope="STIX", observatory="Solar Orbiter", scale=pixel
    )
    fd_bp_map = sunpy.map.Map((bp_image, header))


    sidelobes_ratio = calculate_sidelobes_ratio(fd_bp_map)

    if sidelobes_ratio >= 0.9:
        warnings.warn(f'Flare location may be unreliable. Sidelobes ratio = {np.round(sidelobes_ratio,3)}.')

    # Make a sunpy map from the bp_image, in HPC from STIX observer
    hpc_ref = center_coord.transform_to(frames.Helioprojective(observer=solo, obstime=vis_tr.center)) 
    header_hp = sunpy.map.make_fitswcs_header(bp_image, hpc_ref, scale=pixel, rotation_angle=90 * u.deg + roll)
    hp_map = sunpy.map.Map((bp_image, header_hp))

    # get the position of the max pixel
    max_pixel = np.argwhere(fd_bp_map.data == fd_bp_map.data.max()).ravel() * u.pixel
    # get the world coord of the max pixel - (note WCS axes and array are reversed)
    max_stix = fd_bp_map.pixel_to_world(max_pixel[1], max_pixel[0])

    # get the coordinate of the max pixel in HPC - if coordinate is off limb, assume spherical screen for transform
    with SphericalScreen(hp_map.observer_coordinate, only_off_disk=True):
        max_hpc = max_stix.transform_to(hp_map.coordinate_frame)

    # if plot True, then plot maps in STIX + HPC frames, with max coord. 
    if plot:
        hp_map_rotated = hp_map.rotate()
        fig = plt.figure(figsize=(12, 8))
        ax0 = fig.add_subplot(1, 2, 1, projection=fd_bp_map)
        ax1 = fig.add_subplot(1, 2, 2, projection=hp_map_rotated)
        fd_bp_map.plot(axes=ax0, cmap="viridis")
        fd_bp_map.draw_limb()
        fd_bp_map.draw_grid(annotate=False)
        
        hp_map_rotated.plot(axes=ax1, cmap="viridis")
        hp_map_rotated.draw_limb()
        hp_map_rotated.draw_grid(annotate=False)
        
        
        ax0.plot_coord(max_stix, marker=".", markersize=50, fillstyle="none", color="r", markeredgewidth=2)
        with SphericalScreen(hp_map.observer_coordinate, only_off_disk=True):
            ax1.plot_coord(max_stix, marker=".", markersize=50, fillstyle="none", color="r", markeredgewidth=2)
        plt.tight_layout()
        plt.show()

    results = {'stx':max_stix, 
                'hpc':max_hpc,
                'sidelobes_ratio':sidelobes_ratio}

    return results


def calculate_sidelobes_ratio(bp_nat_map, threshold=200*u.arcsec):
    """

    Calculate the sidelobes ratio for a back-projected image map.

    The sidelobes ratio is a measure of the relative strength of the sidelobes compared to the main peak of the image.
    This ratio helps determine the reliability of the flare location. A sidelobes ratio close to or above 0.9 suggests
    that the flare location may not be reliable due to significant sidelobe interference.

    Parameters
    ----------
    bp_nat_map : `sunpy.map.Map`
        The back-projected image map (in natural units) to analyze. This map is typically generated from visibility data
        and contains the image of the flare.
    threshold : `astropy.units.Quantity`, optional
        The angular separation threshold (in arcseconds) around the peak within which sidelobes are excluded from the calculation.
        Default is 200 arcseconds.

    Returns
    -------
    sidelobes_ratio : float
        The ratio of the maximum sidelobe intensity to the peak intensity in the back-projected image.
        A value close to 1 indicates significant sidelobes, potentially making the flare location unreliable.

    Notes
    -----
    - This is based upon the methodology in the STIX-GSW IDL software.
    """
    max_bp = np.max(bp_nat_map.data)
    ind_max = np.unravel_index(np.argmax(bp_nat_map.data, axis=None), bp_nat_map.data.shape)
    max_bp_coord = bp_nat_map.pixel_to_world(ind_max[1] * u.pix, ind_max[0] * u.pix)

    yy, xx = np.indices(bp_nat_map.data.shape)
    world_coords = bp_nat_map.pixel_to_world(xx * u.pix, yy * u.pix)
    
    distance_wrt_peak = world_coords.separation(max_bp_coord)
    
    bp_image_masked = np.copy(bp_nat_map.data)
    mask = distance_wrt_peak <= threshold
    bp_image_masked[mask] = 0

    sidelobes_ratio = np.max(bp_image_masked) / max_bp
    
    return sidelobes_ratio


