from datetime import datetime
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import stixcore.data.test
import sunpy
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from reproject import reproject_interp
from stixcore.ephemeris.manager import Position
from sunpy.coordinates import frames
from sunpy.coordinates import get_body_heliographic_stonyhurst
from sunpy.map import Map
from sunpy.net import Fido, attrs as a

# __________________________________
def get_SOLO_Pos(map):
    """
    Returns position of SOLO at the time the map was observed

    Parameters
    ----------
    map : `sunpy.map.Map`
        Map to reproject to be as seen from SOLO

    Returns
    -------
    solo_hgs : `astropy.coordinates.SkyCoord`
        The position of SOLO in HeliographicStonyhurst frame
    """
    # Specifiying year, month, day
    date = map.date.value[0:10]
    date_year = int(date[0:4])
    date_month = int(date[5:7])
    date_day = int(date[8:10])

    mkp = Path(
        stixcore.ephemeris.manager.__file__).parent.parent / 'data' / 'test' / 'ephemeris' / 'test_position_20201001_V01.mk'
    with Position(meta_kernel_path=mkp) as pos:
        p = pos.get_position(date=datetime(date_year, date_month, date_day), frame='SOLO_HEE')
    print(p)
    solo_hee = SkyCoord(*p, frame=frames.HeliocentricEarthEcliptic, representation_type='cartesian', obstime=date)
    # Converting HeliocentricEarthEcliptic coords of SOLAR ORBTER position to HeliographicStonyhurst frame
    solo_hgs = solo_hee.transform_to(frames.HeliographicStonyhurst)
    return (solo_hgs)

# __________________________________
def create_headers(obs_ref_coord, map, out_shape=(512, 512)):
    """
    Generates MetaDict and WCS headers for reprojected map

    Parameters
    ----------
    obs_ref_coord : `sunpy.map.Map`
        Target WCS reference coordinate (as seen by observer).
        Generated in map_reproject_to_observer function.

    out_shape : `tuple`
        The shape of the reprojected map - defaults to (512, 512)

    Returns
    -------
    obs_wcs_header : `astropy.wcs`
        WCS header for reprojected map
    obs_metadict_header : `MetaDict`
        MetaDict header for reprojected map
    """
    if map.wavelength.unit.to_string() == '':
        obs_metadict_header = sunpy.map.make_fitswcs_header(
            out_shape,
            obs_ref_coord,
            scale=u.Quantity(map.scale) * 8,
            rotation_matrix=map.rotation_matrix,
            instrument=map.detector)
    else:
        obs_metadict_header = sunpy.map.make_fitswcs_header(
            out_shape,
            obs_ref_coord,
            scale=u.Quantity(map.scale) * 8,
            rotation_matrix=map.rotation_matrix,
            instrument=map.detector,
            wavelength=map.wavelength)
    obs_wcs_header = WCS(obs_metadict_header)
    return (obs_wcs_header, obs_metadict_header)


# __________________________________
def reproject_map(map, observer, out_shape=(512, 512)):
    """
    Reprojects a map to be viewed from a different observer

    Parameters
    ----------
    map : `sunpy.map.Map`
        The input map to be reprojected
    observer : `astropy.coordinates.SkyCoord`
        The coordinates of the observer in HeliographicStonyhurst frame
    out_shape : `tuple`
        The shape of the reprojected map - defaults to (512, 512)

    Returns
    -------
    map : `sunpy.map.Map`
        Reprojected map
    """
    obs_ref_coord = SkyCoord(map.reference_coordinate.Tx, map.reference_coordinate.Ty,
                             obstime=map.reference_coordinate.obstime,
                             observer=observer, frame="helioprojective")
    obs_wcs_header, obs_metadict_header = create_headers(obs_ref_coord, map, out_shape=out_shape)
    output, footprint = reproject_interp(map, obs_wcs_header, out_shape)
    outmap = sunpy.map.Map(output, obs_metadict_header)
    outmap.plot_settings = map.plot_settings
    return (outmap)


# __________________________________
def map_reproj_plot(map, observer):
    """
    Reprojects a map to be viewed from a different observer

    Parameters
    ----------
    map : `sunpy.map.Map`
        The input map to be reprojected
    observer : `astropy.coordinates.SkyCoord`
        The coordinates of the observer in HeliographicStonyhurst frame

    Returns
    -------
    Plots reprojected map, input map and polar positions of observer, input map observer, and The Sun
    """
    reprojected_map = reproject_map(map, observer)
    fig1 = plt.figure(figsize=(16, 4))
    ax1 = fig1.add_subplot(1, 3, 1, projection=map)
    map.plot(axes=ax1, title='Input ' + map.detector + ' map ' + map.date.value[0:10])
    reprojected_map.draw_grid(annotate=False, color='w')
    ax2 = fig1.add_subplot(1, 3, 2, projection=reprojected_map)
    ax2 = plt.gca()
    reprojected_map.plot(axes=ax2, title='Map as seen by observer ' + map.date.value[0:10])
    reprojected_map.draw_grid(annotate=False, color='k')
    ax2.axes.get_yaxis().set_visible(False)

    date = map.date.value[0:10]
    observer_coords = observer
    input_coords = map.observer_coordinate
    # Plotting position of the Sun
    sun_coords = get_body_heliographic_stonyhurst("sun", date)

    # Plotting polar positions
    ax3 = fig1.add_subplot(1, 3, 3, projection="polar")
    ax3.plot(observer_coords.lon.to(u.rad),
             observer_coords.radius.to(u.AU),
             'o', ms=10,
             label="Reprojected Map Observer")
    ax3.plot(input_coords.lon.to(u.rad),
             input_coords.radius.to(u.AU),
             'o', ms=10,
             label="Input Map Observer")
    ax3.plot(sun_coords.lon.to(u.rad),
             sun_coords.radius.to(u.AU),
             'o', ms=10,
             label="Sun")
    plt.legend()
    plt.show()
    return ()
# __________________________________

# Example using AIA map
'''
# Fetch map using FIDO
aia = (a.Instrument.aia &
       a.Sample(24 * u.hour) &
       a.Time('2021-04-17', '2021-04-18'))
wave = a.Wavelength(19.3 * u.nm, 19.3 * u.nm)
res = Fido.search(wave, aia)
files = Fido.fetch(res)

# Convert using to SunPy map
map = sunpy.map.Map(files)

# Set SOLO as observer
observer = get_SOLO_Pos(map)

# Reproject Map
reprojected_map = reproject_map(map, observer)

# Run plotting function
map_reproj_plot(map, observer)
'''
# __________________________________

# Example using HMI map
'''
# Fetch map using FIDO

result = Fido.search(a.Time('2020/06/12 13:20:00', '2020/06/12 13:40:00'),
                     a.Instrument.hmi, a.Physobs.los_magnetic_field)
jsoc_result = result[0][10]

downloaded_file = Fido.fetch(jsoc_result)

# Convert using to SunPy map
map = sunpy.map.Map(downloaded_file)

# Set SOLO as observer
observer = get_SOLO_Pos(map)

# Reproject Map
reprojected_map = reproject_map(map, observer)

# Run plotting function
map_reproj_plot(map, observer)
'''
