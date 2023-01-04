"""

Examples
-------
An AIA map

.. plot::
   :include-source: true

   import astropy.units as u
   from astropy.coordinates import SkyCoord
   from sunpy.net import Fido, attrs as a
   from sunpy.map import Map
   from sunpy.coordinates.frames import HeliographicStonyhurst
   from stixpy.vis.map_reprojection import reproject_map, plot_map_reproj, get_solo_position

   # Search and download map using FIDO
   aia = (a.Instrument.aia &
       a.Sample(24 * u.hour) &
       a.Time('2021-04-17', '2021-04-18'))
   wave = a.Wavelength(19.3 * u.nm, 19.3 * u.nm)
   query = Fido.search(wave, aia)
   results = Fido.fetch(query[0][0])

   # Create map and resample to speed up calculations
   map = Map(results)
   map = map.resample([512, 512]*u.pix)

   # Get SOLO position observer
   # observer = get_solo_position(map) # broken on RTD
   observer = SkyCoord(-97.01771373*u.deg, 21.20931737*u.deg, 1.25689226e+08*u.km,
                       frame=HeliographicStonyhurst, obstime='2021-04-17 00:00:04.840000')

   # Reproject Map
   reprojected_map = reproject_map(map, observer, out_shape=(768, 768))

   # Run plotting function
   plot_map_reproj(map, reprojected_map)


A HMI map

.. plot::
   :include-source: true

   import astropy.units as u
   from astropy.coordinates import SkyCoord
   from sunpy.net import Fido, attrs as a
   from sunpy.map import Map
   from sunpy.coordinates.frames import HeliographicStonyhurst
   from stixpy.vis.map_reprojection import reproject_map, plot_map_reproj, get_solo_position

   # Search and download map using FIDO
   query = Fido.search(a.Time('2020/06/12 13:20:00', '2020/06/12 13:40:00'),
                       a.Instrument.hmi, a.Physobs.los_magnetic_field)
   result = Fido.fetch(query[0][0])

   # Create map and resample to speed up calculations
   map = Map(result)
   map = map.resample([512, 512] * u.pix)

   # Set SOLO as observer
   # observer = get_solo_position(map) # broke on RTD
   observer = SkyCoord(56.18727061*u.deg, 3.5358653*u.deg, 77369481.8542484*u.km,
                       frame=HeliographicStonyhurst, obstime='2020-06-12 13:20:08.300000')

   # Reproject Map
   reprojected_map = reproject_map(map, observer, out_shape=(1024, 1024))

   # Run plotting function
   plot_map_reproj(map, reprojected_map)


"""


from pathlib import Path

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import stixcore.data.test
import sunpy
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from reproject import reproject_interp
from stixcore.ephemeris.manager import Spice
from stixcore.data.test import test_data
from sunpy.coordinates import frames
from sunpy.coordinates import get_body_heliographic_stonyhurst
import drms
import warnings

__all__ = ['get_solo_position', 'get_sdo_position', 'reproject_map', 'create_header', 'plot_map_reproj']


def get_solo_position(map):
    """
    Return the position of SOLO at the time the map was observed

    Parameters
    ----------
    map : `sunpy.map.Map`
        Map to reproject to be as seen from SOLO

    Returns
    -------
    solo_hgs : `astropy.coordinates.SkyCoord`
        Reference coordinate in a helioprojective frame, with observer at position of SOLO
    """
    p = Spice.instance.get_position(date=map.date.datetime, frame='SOLO_HEE')
    solo_hee = SkyCoord(*p, frame=frames.HeliocentricEarthEcliptic, representation_type='cartesian',
                        obstime=map.date.datetime)
    # Converting HeliocentricEarthEcliptic coords of SOLAR ORBITER position to
    # HeliographicStonyhurst frame
    solo_hgs = solo_hee.transform_to(frames.HeliographicStonyhurst)
    return SkyCoord(0*u.arcsec,0*u.arcsec, obstime=map.date.datetime,observer=solo_hgs,frame='helioprojective')
    
def get_sdo_position(map):
    """
    Return the position of SDO at the time the map was observed

    Parameters
    ----------
    map : `sunpy.map.Map`
        Any solar map

    Returns
    -------
    sdo_hgs : `astropy.coordinates.SkyCoord`
        Reference coordinate in a helioprojective frame, with observer at position of SDO
    """
    date_obs_str=map.meta['date-obs']
    client = drms.Client()
    kstr='CUNIT1,CUNIT2,CRVAL1,CRVAL2,CDELT1,CDELT2,CRPIX1,CRPIX2,CTYPE1,CTYPE2,HGLN_OBS,HGLT_OBS,DSUN_OBS,RSUN_OBS,DATE-OBS,RSUN_REF,CRLN_OBS,CRLT_OBS,EXPTIME,INSTRUME,WAVELNTH,WAVEUNIT,TELESCOP,LVL_NUM,CROTA2'
    df = client.query(f'aia.lev1_uv_24s[{date_obs_str}/1m@30s]', key=kstr)
    if df.empty:  # Event not in database yet or other error
        observer=get_observer(date_obs,obs='Earth',sc=sc)
        warnings.warn(f"FITS headers for {date_obs} not available, using Earth observer" )
    else:
        try:
            meta=df.where(df.WAVELNTH==1600).dropna().iloc[0].to_dict()
        except IndexError:
            meta=df.iloc[0].to_dict()
        if np.isnan(meta['CRPIX1']):
            meta['CRPIX1']=0.
        if np.isnan(meta['CRPIX2']):
            meta['CRPIX2']=0.
        sdo_hgs = SkyCoord(meta['HGLN_OBS']*u.deg, meta['HGLT_OBS']*u.deg,meta['DSUN_OBS']*u.m, frame=frames.HeliographicStonyhurst, obstime=meta['DATE-OBS'])

    return SkyCoord(0*u.arcsec,0*u.arcsec, obstime=map.date.datetime,observer=sdo_hgs,frame='helioprojective')


def create_header(obs_ref_coord, map, out_shape=None, out_scale=None):
    """
    Generates MetaDict header for reprojected map

    Parameters
    ----------
    obs_ref_coord : `astropy.coordinates.SkyCoord`
        Target WCS reference coordinate (as seen by observer).
        Generated in get_SPACECRAFT_position function.
        
    map : `sunpy.map.Map`
        Map to be reprojected.

    out_shape : `tuple`
        The shape of the reprojected map - defaults to input shape for full-disk input maps, otherwise will be calculated based on reprojected shape of submap

    out_scale : `Quantity`
        The scale of the output map. Defaults to 1":1px

    Returns
    -------
    obs_metadict_header : `MetaDict`
        MetaDict header for reprojected map
    """
    if out_scale is None:
        out_scale = u.Quantity(map.scale)

    if out_shape is None:
        out_shape = (1,1) # So that shape can be modified later
        
    target_observer = obs_ref_coord
    target_wcs=WCS(sunpy.map.make_fitswcs_header(out_shape,target_observer)) #wcs_utils.reference_coordinate_from_frame(wcs_utils.solar_wcs_frame_mapping(target_wcs)) # Must be SkyCoord ie reference coordinate
    edges_pix = np.concatenate(sunpy.map.map_edges(map))
    edges_coord = map.pixel_to_world(edges_pix[:, 0], edges_pix[:, 1])
    new_edges_coord = edges_coord.transform_to(target_observer)
    new_edges_xpix, new_edges_ypix = target_wcs.world_to_pixel(new_edges_coord)
    
    # Check for NaNs
    if new_edges_xpix[np.isnan(new_edges_xpix)] != np.array([]) or new_edges_ypix[np.isnan(new_edges_ypix)] != np.array([]):
         cc = sunpy.map.all_coordinates_from_map(map).transform_to(target_observer)
         on_disk = sunpy.map.coordinate_is_on_solar_disk(cc)
         on_disk_coordinates = cc[on_disk]
         nan_percent = 1. - len(on_disk_coordinates)/len(cc.flatten())
         if nan_percent > 0.5:
             warnings.warn(f"{nan_percent*100:.1f}% of pixels in reprojected map are NaN!")

    # Determine the extent needed - use of nanmax/nanmin means only on-disk coords are considered
    left, right = np.nanmin(new_edges_xpix), np.nanmax(new_edges_xpix)
    bottom, top = np.nanmin(new_edges_ypix), np.nanmax(new_edges_ypix)

    # Adjust the CRPIX and NAXIS values
    target_header = sunpy.map.make_fitswcs_header(out_shape, target_observer,instrument=map.detector) # What to do about scale?
    target_header['crpix1'] -= left
    target_header['crpix2'] -= bottom
    target_header['naxis1'] = int(np.ceil(right - left))
    target_header['naxis2'] = int(np.ceil(top - bottom))
    
    return target_header


def reproject_map(map, observer, out_shape=None):
    """
    Reproject a map as viewed from a different observer.

    Parameters
    ----------
    map : `sunpy.map.Map`
        The input map to be reprojected
    observer : `astropy.coordinates.SkyCoord`
        The reference coordinate of the observer in HeliographicStonyhurst frame
    out_shape : `tuple`
        The shape of the reprojected map - defaults to same size as input map if map is full-disk, otherwise the reprojected shape will be calculated based on the submap coordinates

    Returns
    -------
    outmap : `sunpy.map.Map`
        Reprojected map
    """
    if out_shape is None and sunpy.map.contains_full_disk(map):
        out_shape = map.data.shape
        
    # Make sure observer is a reference coordinate, not just an observer coordinate
    if not isinstance(observer, SkyCoord):
        observer = SkyCoord(0*u.arcsec,0*u.arcsec, obstime=map.date.datetime,observer=observer,frame='helioprojective') # What if user inputs WCS? I know I wrote a observer_from_wcs somewhere ... find it and add here
        
    obs_metadict_header = create_header(observer, map, out_shape=out_shape)
    # Don't try to reproject to a huge array
    if obs_metadict_header['naxis1'] > 2048 or obs_metadict_header['naxis2'] > 2048: # Reprojection will take at least 30s in this case.
        warnings.warn(f"Reprojection to {obs_metadict_header['naxis1']}x{obs_metadict_header['naxis2']} array not permitted! Check that your desired observer is correct, use a submap of the region, or scale down. Alternatively, proceed at your own risk using new_header=create_header(observer,map) and map.reproject_to(new_header) .")
    outmap = map.reproject_to(obs_metadict_header)
    outmap.plot_settings = map.plot_settings
    return outmap


def plot_map_reproj(map, reprojected_map):
    """
    Plot the original map, reprojected map and observer locations

    Parameters
    ----------
    map : `sunpy.map.Map`
        The input map to be reprojected
    reprojected_map :  `sunpy.map.Map`
        The reprojected map

    Returns
    -------
    `matplotlib.Figure`
        Figure showing the original map, reprojected map, and the observer locations
    """
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(1, 3, 1, projection=map)
    map.plot(axes=ax1, title=f'Input {map.detector} map {map.date}')
    map.draw_grid(annotate=False, color='w')
    ax2 = fig.add_subplot(1, 3, 2, projection=reprojected_map)
    reprojected_map.plot(axes=ax2, title=f'Map as seen by observer {map.date}')
    reprojected_map.draw_grid(annotate=False, color='k')
    ax2.axes.get_yaxis().set_visible(False)

    new_observer = reprojected_map.observer_coordinate
    original_observer = map.observer_coordinate
    # Plotting position of the Sun
    sun_coords = get_body_heliographic_stonyhurst("sun", map.date)

    # Plotting polar positions
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")
    ax3.plot(new_observer.lon.to(u.rad),
             new_observer.radius.to(u.AU),
             'o', ms=10,
             label="New Map Observer")
    ax3.plot(original_observer.lon.to(u.rad),
             original_observer.radius.to(u.AU),
             'o', ms=10,
             label="Original Map Observer")
    ax3.plot(sun_coords.lon.to(u.rad),
             sun_coords.radius.to(u.AU),
             'o', ms=10,
             label="Sun")
    fig.legend()
    plt.show()
    return fig
