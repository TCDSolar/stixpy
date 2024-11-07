import astropy.units as u
import sunpy.map
import sunpy.time
import xrayvision.imaging
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst
from xrayvision.visibility import Visibilities

import stixpy.calibration.visibility
import stixpy.coordinates.transforms
from stixpy.coordinates.transforms import STIXImaging


def estimate_stix_flare_location(cpd_sci, time_range=None, energy_range=None, subcollimators=None,
                                 cpd_bkg=None, time_range_bkg=None):
    """
    Estimates flare location from STIX compressed pixel data.

    FLare location is assumed to be the location of the brightest pixel in the backprojection
    map calculated from the input STIX pixel data.

    Parameters
    ----------
    cpd_sci: `stixpy.product.Product`
        The STIX pixel data. Assumed to be already background subtracted.
    time_range: `sunpy.time.TimeRange` (optional)
        The time range over which to estimate the flare location.
        Default is all times in cpd_sci.
    energy_range: `astropy.units.Quantity` in spectral units (optional)
        Length-2 quantity giving the lower and upper bounds of the energy range to use for imaging.
        Default is all finite energies in cpd_sci.
    subcollimators: `iterable` of `str` (optional)
        The labels of the subcollimators to include in the output Visibilities object, e.g.
        ``['10a', '10b', '10c',...]``
        Default is all subcollimators in cpd_sci
    cpd_bkg: stixpy.product.Product` (optional)
        The background to subtract from the pixel data before determining the flare location.
        Passed to construct_stix_calibrated_visibilities().
    time_range_bkg: `sunpy.time.TimeRange`
        The time range within cpd_bkg to use for the background subtraction.
        Passed to construct_stix_calibrated_visibilities().

    Returns
    -------
    flare_loc: `astropy.coordinates.SkyCoord`
        The estimated flare location.
    map_bp: `sunpy.map.Map`
        The backprojection map from whose brightest pixel the flare location was estimated.
    """
    if time_range is None:
        time_range = cpd_sci.time_range
    if subcollimators is None:
        subcollimators = ["10a", "10b", "10c", "9a", "9b", "9c",
                          "8a", "8b", "8c", "7a", "7b", "7c"]
    # Construct STIX location and centre of FOV.
    roll, solo_xyz, pointing = stixpy.coordinates.transforms.get_hpc_info(time_range.start,
                                                                          time_range.end)
    solo = HeliographicStonyhurst(*solo_xyz, obstime=time_range.center,
                                  representation_type="cartesian")
    fov_centre = STIXImaging(0 * u.arcsec, 0 * u.arcsec, obstime=time_range.start,
                             obstime_end=time_range.end, observer=solo)
    # Generate calibrated visibilities using coarser subcollimators.
    vis = construct_stix_calibrated_visibilities(
        cpd_sci, fov_centre, time_range=time_range, energy_range=energy_range,
        subcollimators=subcollimators)
    # Produce backprojected image and find brightest pixel. Use this for flare location.
    imsize = [512, 512] * u.pixel  # number of pixels of the map to reconstruct
    plate_scale = [10, 10] * u.arcsec / u.pixel  # pixel size in arcsec
    bp_image = xrayvision.imaging.vis_to_image(vis, imsize, pixel_size=plate_scale)
    max_idx = np.argwhere(bp_image == bp_image.max()).ravel()
    # Calculate WCS for backprojected image in STIXImaging and HPC frames.
    # Recalculate STIX HPC info as slightly different times will have been used
    # than input times due to onboard STIX time binning.
    vis_tr = sunpy.time.TimeRange(vis.meta["time_range"])
    roll, solo_xyz, pointing = stixpy.coordinates.transforms.get_hpc_info(vis_tr.start, vis_tr.end)
    solo = HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")
    coord = STIXImaging(0 * u.arcsec, 0 * u.arcsec, obstime=vis_tr.start, obstime_end=vis_tr.end, observer=solo)
    header_bp = sunpy.map.make_fitswcs_header(bp_image, coord, telescope="STIX",
                                              observatory="Solar Orbiter", scale=plate_scale)
    map_bp = sunpy.map.Map(bp_image, header_bp).wcs
    wcs_bp = map_bp.wcs
    # Estimate flare location from brightest pixel in backprojection image
    flare_loc = wcs_bp.array_index_to_world(*max_idx)
    flare_loc = STIXImaging(flare_loc[0].to(u.arcsec), flare_loc[1].to(u.arcsec),
                            obstime=vis_tr.start, obstime_end=vis_tr.end, observer=solo)
    return flare_loc, map_bp


def construct_stix_calibrated_visibilities(cpd_sci, flare_location, time_range=None,
                                           energy_range=None, subcollimators=None,
                                           cpd_bkg=None, time_range_bkg=None, **kwargs):
    """
    Constructs calibrated STIX visibilities from STIX compressed pixel data.

    Extra kwargs are passed to `stixpy.calibration.visibility.create_meta_pixels`

    Parameters
    ----------
    cpd_sci: `stixpy.product.Product`
        The STIX pixel data. Assumed to be already background subtracted.
    flare_location: `astropy.coordinates.SkyCoord`
        The flare location. Frame must be convertible to `stixpy.coordinates.transdforms.STIXImaging`.
    time_range: `sunpy.time.TimeRange` (optional)
        The time range over which to estimate the flare location.
        Default is all times in cpd_sci.
    energy_range: `astropy.units.Quantity` in spectral units (optional)
        Length-2 quantity giving the lower and upper bounds of the energy range to use for imaging.
        Default is all finite energies in cpd_sci.
    subcollimators: `iterable` of `str`
        The labels of the subcollimators to use in estimating the flare locations, e.g.
        ``['10a', '10b', '10c',...]``
        Default is all subcollimators in cpd_sci.
    cpd_bkg: stixpy.product.Product` (optional)
        The background to subtract from the pixel data before determining the flare location.
        If None and time_range_bkg is also None, no background is subtracted.
    time_range_bkg: `sunpy.time.TimeRange`
        The time range within cpd_bkg to use for the background subtraction.
        If None, entire time range of cpd_bkg is used to determine background.
        If not None, and cpd_bkg is None, the background is determined from this time range
        applied to cpd_sci.

    Returns
    -------
    vis: `xrayvision.visibility.Visibilities`
        The calibrated STIX visibilities.
    """
    # Sanitze inputs.
    if time_range is None:
        time_range = cpd_sci.time_range
    times = Time([time_range.start, time_range.end])
    if energy_range is None:
        energy_range = u.Quantity([cpd_sci.energies["e_low"][np.isfinite(cpd_sci.energies["e_low"])][0],
                                   cpd_sci.energies["e_high"][np.isfinite(cpd_sci.energies["e_high"])][-1]])
    no_shadowing = kwargs.pop("no_shadowing", True)
    # Generate meta_pixels from pixel_data.
    meta_pixels = stixpy.calibration.visibility.create_meta_pixels(
        cpd_sci, time_range=times, energy_range=energy_range,
        flare_location=flare_location, no_shadowing=no_shadowing, **kwargs)
    # Subtract background if background pixel data provided.
    if cpd_bkg is None and time_range_bkg is not None:
        cpd_bkg = cpd_sci
    if cpd_bkg is not None:
        if time_range_bkg is None:
            time_range_bkg = cpd_bkg.time_range
        times_bkg = Time([time_range_bkg.start, time_range_bkg.end])
        meta_pixels_bkg = stixpy.calibration.visibility.create_meta_pixels(
            cpd_bkg, time_range=times_bkg, energy_range=energy_range,
            flare_location=[0, 0] * u.arcsec, no_shadowing=no_shadowing, **kwargs)
        meta_pixels = _subtract_background_from_stix_meta_pixels(meta_pixels, meta_pixels_bkg)
    # Generate and calibrate visibilities.
    vis = stixpy.calibration.visibility.create_visibility(meta_pixels)
    vis = stixpy.calibration.visibility.calibrate_visibility(vis, flare_location=flare_location)
    if subcollimators is not None:
        idx_subcol = np.argwhere(np.isin(vis.meta["vis_labels"], subcollimator_labels)).ravel()
        vis = vis[idx_subcol]
    return vis


def _subtract_background_from_stix_meta_pixels(meta_pixels_sci, meta_pixels_bkg):
    """
    Estimates flare location from STIX pixel data.

    Parameters
    ----------
    meta_pixels_sci: `dict`
        The STIX meta pixel representing the observations. Format must be
        same as output from `stixpy.calibration.visibility.create_meta_pixels`.
    meta_pixels_bkg: `dict`
        The STIX meta pixels representing the background. Format must be
        same as output from `stixpy.calibration.visibility.create_meta_pixels`.
    """
    meta_pixels_bkg_sub = {
        **meta_pixels_sci,                   
        "abcd_rate_kev_cm": meta_pixels_sci["abcd_rate_kev_cm"] - meta_pixels_bkg["abcd_rate_kev_cm"],
        "abcd_rate_error_kev_cm": np.sqrt(meta_pixels_sci["abcd_rate_error_kev_cm"] ** 2
                                          + meta_pixels_bkg["abcd_rate_error_kev_cm"] ** 2),
    }
    return meta_pixels_bkg_sub
