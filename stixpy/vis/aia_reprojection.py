from datetime import datetime
from stixcore.ephemeris.manager import Position
import stixcore.data.test
from pathlib import Path
import sunpy
import sys
from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.map import Map
from sunpy.coordinates import get_body_heliographic_stonyhurst
from sunpy.coordinates import frames
from astropy.wcs import WCS
from pathlib import Path
from reproject import reproject_interp
import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
plt.rcParams.update({'font.size': 7})


def get_aia_map(start_day, end_day, wavelength, path):
	'''
	e.g
	start_day = '2020-10-01'
	end_day = '2020-10-02'
	path = '/Users/Username/Desktop/'
	wavelength[nm] = 19.5 
	'''
	# Fetching AIA Data
	aia = (a.Instrument.aia &
	       a.Sample(24 * u.hour) &
	       a.Time(start_day, end_day))
	wave = a.Wavelength(wavelength, wavelength)
	res = Fido.search(wave, aia)
	files = Fido.fetch(res, path = path)

	# Converting to sunpy map
	aia_map  = sunpy.map.Map(files)
	return (aia_map)
#__________________________________
# ftp://spiftp.esac.esa.int/data/SPICE/ 

def get_SOLO_Pos(start_day, end_day):
	# Specifiying year, month, day
	start_year = int(start_day[0:4])
	start_month = int(start_day[5:7])
	start_day = int(start_day[8:10])

	from pathlib import Path
	mkp = Path(stixcore.ephemeris.manager.__file__).parent.parent / 'data' / 'test' / 'ephemeris' / 'test_position_20201001_V01.mk'
	with Position(meta_kernel_path=mkp) as pos:
	    p = pos.get_position(date=datetime(start_year, start_month, start_day), frame='SOLO_HEE')
	print(p)
	return(p)
#__________________________________


def convert_SOLO_coords(start_day, end_day, p, aiamap):
	# Converts return of get_SOLO_Pos to HeliographicStonyhurst frame 

	# Converting return of STIXCORE POISITION to HeliocentricEarthEcliptic frame
	solo_hee = SkyCoord(*p, frame=frames.HeliocentricEarthEcliptic, representation_type='cartesian', obstime=start_day)

	# Converting HeliocentricEarthEcliptic coords of SOLAR ORBTER position to HeliographicStonyhurst frame
	solo_hgs = solo_hee.transform_to(frames.HeliographicStonyhurst)

	# Creating reference frame (AIA as seen by SOLAR ORBITER)
	solo_hgs_ref_coord = SkyCoord(aiamap.reference_coordinate.Tx,aiamap.reference_coordinate.Ty,obstime=aiamap.reference_coordinate.obstime,
	observer=solo_hgs,frame="helioprojective")
	return (solo_hgs_ref_coord, solo_hgs)
#__________________________________


def create_wcs_header(solo_hgs_ref_coord, aia_map):
	# Creating wcs header for SOLAR ORBITER frame  

	out_shape = (512, 512)
	solo_header = sunpy.map.make_fitswcs_header(
	out_shape,
	solo_hgs_ref_coord,
	scale=u.Quantity(aia_map.scale)*8,
	rotation_matrix=aia_map.rotation_matrix,
	instrument="AIA",
	wavelength=aia_map.wavelength)
	
	solo_wcs = WCS(solo_header)
	print('header 4096: ', solo_wcs)
	return(solo_wcs, out_shape, solo_header)
#__________________________________


def aia_reproj_interp(aia_map, solo_wcs, out_shape, solo_header):
	# Performing reprojection to generate map of AIA as seen by Solar Orbiter
	output, footprint = reproject_interp(aia_map, solo_wcs, out_shape)
	outmap = sunpy.map.Map((output, solo_header))
	print('outmap 4096: ', outmap)

	outmap.plot_settings = aia_map.plot_settings
	return(outmap)
#__________________________________


def aia_reproj_plot(start_day, end_day, aia_map, outmap, solo_hgs):
	# Plotting maps

	fig1 = plt.figure(figsize=(16,6))
	ax1 = fig1.add_subplot(1, 3, 1, projection=aia_map)
	aia_map.plot(axes=ax1)
	outmap.draw_grid(color='w')
	ax2 = fig1.add_subplot(1, 3, 2, projection=outmap)
	ax2 = plt.gca()
	outmap.plot(axes=ax2, title='AIA observation as seen from Solar Orbiter')
	outmap.draw_grid(color='w')
	ax2.axes.get_yaxis().set_visible(False)


	date = start_day
	solo_coords = solo_hgs
	aia_coords = aia_map.observer_coordinate
	# Plotting position of the Sun
	sun = get_body_heliographic_stonyhurst("sun",  date)


	# Plotting polar positions
	ax3 = fig1.add_subplot(1, 3, 3, projection="polar")
	ax3.plot(solo_coords.lon.to(u.rad),
	         solo_coords.radius.to(u.AU),
	        'o', ms=10,
	        label="SOLO")
	ax3.plot(aia_coords.lon.to(u.rad),
	         aia_coords.radius.to(u.AU),
	        'o', ms=10,
	        label="AIA")
	ax3.plot(sun.lon.to(u.rad),
	         sun.radius.to(u.AU),
	        'o', ms=10,
	        label="Sun")
	plt.legend()
	plt.show()
	return()
#__________________________________


#EXAMPLE
sample_start_day = '2021-03-09'
sample_end_day = '2021-03-10'
sample_wavelength = 19.5*u.nm
sample_path = '/Users/brendanclarke/Desktop/'

aia_map = get_aia_map(sample_start_day, sample_end_day, sample_wavelength, sample_path)

p = get_SOLO_Pos(sample_start_day, sample_end_day)

solo_hgs_ref_coord, solo_hgs = convert_SOLO_coords(sample_start_day, sample_end_day, p, aia_map)

solo_wcs, out_shape, solo_header  = create_wcs_header(solo_hgs_ref_coord, aia_map)

outmap = aia_reproj_interp(aia_map, solo_wcs, out_shape, solo_header)

aia_reproj_plot(sample_start_day, sample_end_day, aia_map, outmap, solo_hgs)