import warnings

import numpy as np
from xrayvision.imaging import vis_to_image

from astropy import units as u
from astropy.coordinates import SkyCoord

import sunpy.map
import sunpy.sun.constants as sun_const
from sunpy.coordinates import SphericalScreen, frames
from sunpy.time import TimeRange

from stixpy.calibration.visibility import calibrate_visibility, create_meta_pixels, create_visibility
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.product import Product
from stixpy.product.sources.science import CompressedPixelData
from stixpy.visualisation.plotters import plot_flare_location

__all__ = [
    "estimate_flare_location",
]


def get_rsun_obs(observer):
    """
    Get the angular radius of the Sun as seen from an observer location.

    Parameters
    ----------
    observer : `astropy.coordinates.BaseCoordinateFrame` or `astropy.coordinates.SkyCoord`
        Observer location, must have a well-defined distance from Sun centre
        (e.g. `sunpy.coordinates.frames.HeliographicStonyhurst`).

    Returns
    -------
    `astropy.units.Quantity`
        The angular radius of the Sun, ``arcsin(R_sun / distance)``.
    """
    rsun_obs = np.arcsin((sun_const.radius / observer.spherical.distance).decompose()).to(u.arcsec)
    return rsun_obs


def estimate_flare_location(cpd_sci, time_range, energy_range=None, plot=False):
    """
    Estimate the flare location from STIX imaging data of the Solar Orbiter STIX instrument.

    Based on the IDL software ``stx_estimate_flare_location``, this creates back-projected images
    in the STIX imaging and Helioprojective frames and finds the location of the maximum-intensity
    pixel.

    Optionally, plots the results showing the estimated location in both coordinate systems.

    Parameters
    ----------
    cpd_sci : `str` or `~stixpy.product.sources.science.CompressedPixelData`
        Path to the STIX compressed pixel data product file, or an already loaded product.
    time_range : `sunpy.time.TimeRange`
        The time range over which to estimate the flare location.
    energy_range : `astropy.units.Quantity`, optional
        The energy range (e.g., in keV) for the analysis. Defaults to 6-15 keV.
    plot : bool, optional
        If True, plot the back-projected images in both STIX and Helioprojective frames using
        `~stixpy.visualisation.plotters.plot_flare_location`. Default is False.

    Returns
    -------
    `dict`
        Dictionary containing:

        * ``stx`` (`~astropy.coordinates.SkyCoord`) -- estimated flare location in STIX imaging coordinates.
        * ``hpc`` (`~astropy.coordinates.SkyCoord`) -- estimated flare location in Helioprojective Cartesian
          coordinates.
        * ``sidelobes_ratio`` (float) -- ratio used to assess the reliability of the location
          (see `calculate_sidelobes_ratio`).
        * ``vis_tr`` (`~sunpy.time.TimeRange`) -- actual time range covered by the visibilities used to make
          the image, which can differ slightly from the input ``time_range``.

    Notes
    -----
    The function involves the following steps:

    - Reading STIX pixel data and generating meta pixels for a given time and energy range.
    - Creating visibility data from the meta pixels.
    - Obtaining solar observer coordinates and converting them to the Heliographic Stonyhurst frame.
    - Creating a back-projected image from the visibility data.
    - Transforming the coordinates of the maximum pixel in the image to Helioprojective coordinates.
    - Optionally, plotting the back-projected images and marking the estimated flare location.

    """

    if energy_range is None:
        energy_range = [6, 15] * u.keV

    if not isinstance(cpd_sci, CompressedPixelData):
        cpd_sci = Product(cpd_sci)

    # normalise to a TimeRange then unpack to a plain [start, end] as expected by create_meta_pixels
    time_range = TimeRange(time_range)

    meta_pixels_sci = create_meta_pixels(
        cpd_sci,
        time_range=[time_range.start, time_range.end],
        energy_range=energy_range,
        flare_location=[0, 0] * u.arcsec,
        no_shadowing=True,
    )

    # create visibilities
    vis = create_visibility(meta_pixels_sci)
    vis_tr = TimeRange(vis.meta["time_range"])

    roll, solo_xyz, _ = get_hpc_info(vis_tr.start, vis_tr.end)
    solo = frames.HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")

    center_map = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective(observer=solo, obstime=solo.obstime))
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
        warnings.warn(f"Flare location may be unreliable. Sidelobes ratio = {np.round(sidelobes_ratio, 3)}.")

    # Make a sunpy map from the bp_image, in HPC from STIX observer
    hpc_ref = center_coord.transform_to(frames.Helioprojective(observer=solo, obstime=vis_tr.center))
    header_hp = sunpy.map.make_fitswcs_header(bp_image, hpc_ref, scale=pixel, rotation_angle=90 * u.deg + roll)
    hp_map = sunpy.map.Map((bp_image, header_hp))

    # get the position of the max pixel, argmax/unravel_index picks a single deterministic
    # pixel even if multiple pixels are tied for the maximum value
    ind_max = np.unravel_index(np.argmax(fd_bp_map.data, axis=None), fd_bp_map.data.shape)
    # get the world coord of the max pixel - (note WCS axes and array are reversed)
    max_stix = fd_bp_map.pixel_to_world(ind_max[1] * u.pix, ind_max[0] * u.pix)

    # get the coordinate of the max pixel in HPC - if coordinate is off limb, assume spherical screen for transform
    with SphericalScreen(hp_map.observer_coordinate, only_off_disk=True):
        max_hpc = max_stix.transform_to(hp_map.coordinate_frame)

    # if plot True, then plot maps in STIX + HPC frames, with max coord.
    if plot:
        plot_flare_location(fd_bp_map, hp_map, max_stix)

    results = {"stx": max_stix, "hpc": max_hpc, "sidelobes_ratio": sidelobes_ratio, "vis_tr": vis_tr}

    return results


def calculate_sidelobes_ratio(bp_nat_map, threshold=200 * u.arcsec):
    """
    Calculate the sidelobes ratio for a back-projected image map.

    The sidelobes ratio is a measure of the relative strength of the sidelobes compared to the main peak of the image.
    This ratio helps determine the reliability of the flare location. A sidelobes ratio close to or above 0.9 suggests
    that the flare location may not be reliable due to significant sidelobe interference.

    Parameters
    ----------
    bp_nat_map : `sunpy.map.Map`
        The back-projected image map (in natural units) to analyse. This map is typically generated from visibility data
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
    This is based on the methodology in the STIX-GSW IDL software.
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
