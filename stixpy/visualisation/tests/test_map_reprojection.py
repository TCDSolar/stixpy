import pytest
from unittest.mock import patch

import astropy.units as u
import numpy as np
from astropy.time.core import Time
from sunpy.data import sample
from sunpy.map import Map

from stixpy.visualisation.map_reprojection import get_solo_position, reproject_map

@pytest.mark.skip
@patch('sunpy.map.GenericMap')
def test_get_solo_position(mock_map):
    mock_map.date = Time('2021-04-17T00:00')
    solo_pos = get_solo_position(mock_map)

    # equivalent to to meter i.e. a lenght and finite
    assert solo_pos.data.xyz.unit.is_equivalent('m')
    assert all(np.isfinite(solo_pos.data.xyz))
    assert np.allclose(solo_pos.data.xyz, [-1.43160849e+07, -1.16297787e+08, 4.54713759e+07] * u.km)


# Testing to check if reprojection is correct by doing reprojecting map onto itself.
def test_map_reproject():
    map = Map(sample.AIA_094_IMAGE)
    observer = map.observer_coordinate
    reprojected_map = reproject_map(map, observer)
    assert np.allclose(map.observer_coordinate.lat, reprojected_map.observer_coordinate.lat)
    assert np.allclose(map.observer_coordinate.lon, reprojected_map.observer_coordinate.lon)
    assert np.allclose(map.observer_coordinate.radius, reprojected_map.observer_coordinate.radius)
    assert np.allclose(map.data, reprojected_map.data, atol=1e-7)
