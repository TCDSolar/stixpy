import numpy as np

from stixpy.vis.map_reprojection import *

# fetching test SunPy map
aia = (a.Instrument.aia &
       a.Sample(24 * u.hour) &
       a.Time('2021-04-17', '2021-04-18'))
wave = a.Wavelength(19.3 * u.nm, 19.3 * u.nm)
res = Fido.search(wave, aia)
files = Fido.fetch(res)

# Convert using to SunPy map
map = sunpy.map.Map(files)

# Testing to check if SOLO position data are correct.
def test_get_SOLO_Pos(map = map):
    aia_map = sunpy.map.Map(map)
    solo_pos = get_SOLO_Pos(aia_map)
    assert solo_pos.data.xyz[0].unit == "km"
    assert solo_pos.data.xyz[1].unit == "km"
    assert solo_pos.data.xyz[2].unit == "km"
    check_y = isinstance(solo_pos.data.xyz[0].value, np.float64)
    assert check_y == True
    check_y = isinstance(solo_pos.data.xyz[1].value, np.float64)
    assert check_y == True
    check_z = isinstance(solo_pos.data.xyz[2].value, np.float64)
    assert check_z == True
    return()

# Testing to check if reprojection is correct by doing reprojecting map onto itself.
def map_reproj_test(map=map, observer=map.observer_coordinate):
    self_reprojected_map = reproject_map(map, observer)
    assert np.allclose(map.observer_coordinate.lat, self_reprojected_map.observer_coordinate.lat)
    assert np.allclose(map.observer_coordinate.lon, self_reprojected_map.observer_coordinate.lon)
    assert np.allclose(map.observer_coordinate.radius, self_reprojected_map.observer_coordinate.radius)
    return()
