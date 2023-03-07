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
   from stixpy.visualisation.map_reprojection import reproject_map, plot_map_reproj, get_solo_position

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
   from stixpy.visualisation.map_reprojection import reproject_map, plot_map_reproj, get_solo_position

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

import astropy.units as u
import matplotlib.pyplot as plt

import sunpy
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from reproject import reproject_interp
try:
    from stixcore.ephemeris.manager import Spice
except (ImportError, ModuleNotFoundError):
    Spice = None
from sunpy.coordinates import frames
from sunpy.coordinates import get_body_heliographic_stonyhurst

from stixpy.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ['get_solo_position', 'reproject_map', 'create_headers', 'plot_map_reproj']


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
        The position of SOLO in HeliographicStonyhurst frame
    """
    if Spice is not None:
        logger.info('Using Spice')
        p = Spice.instance.get_position(date=map.date.datetime, frame='SOLO_HEE')
        solo_hee = SkyCoord(*p, frame=frames.HeliocentricEarthEcliptic, representation_type='cartesian',
                            obstime=map.date.datetime)
        # Converting HeliocentricEarthEcliptic coords of SOLAR ORBITER position to
        # HeliographicStonyhurst frame
        solo_hgs = solo_hee.transform_to(frames.HeliographicStonyhurst)
    else:
        logger.info('Spice not configured falling back to JPL ')
        solo_hgs = get_horizons_coord('solo', time=map.date)


def create_headers(obs_ref_coord, map, out_shape=None, out_scale=None):
    """
    Generates MetaDict and WCS headers for reprojected map

    Parameters
    ----------
    obs_ref_coord : `sunpy.map.Map`
        Target WCS reference coordinate (as seen by observer).
        Generated in map_reproject_to_observer function.

    out_shape : `tuple`
        The shape of the reprojected map - defaults to (512, 512)

    out_scale : `Quantity`
        The scale of the output map

    Returns
    -------
    obs_wcs_header : `astropy.wcs`
        WCS header for reprojected map
    obs_metadict_header : `MetaDict`
        MetaDict header for reprojected map
    """
    if out_scale is None:
        out_scale = u.Quantity(map.scale)

    if out_shape is None:
        out_shape = map.data.shape

    if map.wavelength.unit.to_string() == '':
        obs_metadict_header = sunpy.map.make_fitswcs_header(
            out_shape,
            obs_ref_coord,
            scale=out_scale,
            rotation_matrix=map.rotation_matrix,
            instrument=map.detector)
    else:
        obs_metadict_header = sunpy.map.make_fitswcs_header(
            out_shape,
            obs_ref_coord,
            scale=out_scale,
            rotation_matrix=map.rotation_matrix,
            instrument=map.detector,
            wavelength=map.wavelength)
    obs_wcs_header = WCS(obs_metadict_header)
    return obs_wcs_header, obs_metadict_header


def reproject_map(map, observer, out_shape=None):
    """
    Reproject a map as viewed from a different observer.

    Parameters
    ----------
    map : `sunpy.map.Map`
        The input map to be reprojected
    observer : `astropy.coordinates.SkyCoord`
        The coordinates of the observer in HeliographicStonyhurst frame
    out_shape : `tuple`
        The shape of the reprojected map - defaults to same size as input map

    Returns
    -------
    map : `sunpy.map.Map`
        Reprojected map
    """
    if out_shape is None:
        out_shape = map.data.shape

    obs_ref_coord = SkyCoord(map.reference_coordinate.Tx, map.reference_coordinate.Ty,
                             obstime=map.reference_coordinate.obstime,
                             observer=observer, frame="helioprojective")
    obs_wcs_header, obs_metadict_header = create_headers(obs_ref_coord, map, out_shape=out_shape)
    output, footprint = reproject_interp(map, obs_wcs_header, out_shape)
    outmap = sunpy.map.Map(output, obs_metadict_header)
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
    reprojected_map.draw_grid(annotate=False, color='w')
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
