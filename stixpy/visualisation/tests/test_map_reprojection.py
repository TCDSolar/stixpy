from unittest.mock import patch

import numpy as np
import pytest

import astropy.units as u
from astropy.time.core import Time

from stixpy.visualisation.map_reprojection import get_solo_position, reproject_map


@pytest.mark.skip
@patch("sunpy.map.GenericMap")
def test_get_solo_position(mock_map):
    mock_map.date = Time("2021-04-17T00:00")
    solo_pos = get_solo_position(mock_map)

    # equivalent to meter i.e. a length and finite
    assert solo_pos.data.xyz.unit.is_equivalent("m")
    assert all(np.isfinite(solo_pos.data.xyz))
    assert np.allclose(solo_pos.data.xyz, [-1.43160849e07, -1.16297787e08, 4.54713759e07] * u.km)


@pytest.mark.remote_data
def test_map_reproject():
    # Testing to check if reprojection is correct by doing reprojecting amap onto itself.
    from sunpy.data import sample
    from sunpy.map import Map

    amap = Map(sample.AIA_094_IMAGE)
    observer = amap.observer_coordinate
    reprojected_map = reproject_map(amap, observer)
    assert np.allclose(amap.observer_coordinate.lat, reprojected_map.observer_coordinate.lat)
    assert np.allclose(amap.observer_coordinate.lon, reprojected_map.observer_coordinate.lon)
    assert np.allclose(amap.observer_coordinate.radius, reprojected_map.observer_coordinate.radius)
    assert np.allclose(amap.data, reprojected_map.data, atol=1e-7)
