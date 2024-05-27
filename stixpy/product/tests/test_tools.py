import astropy.nddata
import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from ndcube import ExtraCoords, NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate
from ndcube.tests.helpers import assert_cubes_equal

from .. import tools


@pytest.fixture
def cube():
    data = np.ones((2, 5))
    mask = np.array([[False, False, False, True, False], [True, True, True, False, True]], dtype=bool)
    uncertainty = astropy.nddata.StdDevUncertainty(data * 0.1)

    time_coord = TimeTableCoordinate(
        Time("2000-01-01", scale="utc", format="iso") + np.arange(0, data.shape[0] * 10, 10) * u.s
    )
    energy_coord = QuantityTableCoordinate(
        np.arange(3, 3 + data.shape[1]) * u.keV, names="energy", physical_types="em.energy"
    )
    ec = ExtraCoords()
    ec.add("time", 1, time_coord)
    ec.add("energy", 0, energy_coord)
    wcs = ec.wcs

    return NDCube(data=data, wcs=wcs, mask=mask, uncertainty=uncertainty)


def test_rebin_by_edges(cube):
    # Run function.
    axes_idx_edges = [0, 2], [0, 1, 3, 5]
    output = tools.rebin_by_edges(
        cube,
        axes_idx_edges,
        operation=np.sum,
        operation_ignores_mask=False,
        handle_mask=np.all,
        propagate_uncertainties=True,
    )
    # Define expected outputs.
    data = np.array([[1.0, 2.0, 2.0]])
    mask = np.zeros(data.shape, dtype=bool)
    uncertainty = astropy.nddata.StdDevUncertainty([[0.1, 0.14142135623730953, 0.14142135623730953]])
    ec = ExtraCoords()
    ec.add("iso(utc; None", 1, [5.0] * u.s, physical_types="time")
    ec.add("energy", 0, [3.0, 4.5, 6.5] * u.keV, physical_types="em.energy")
    wcs = ec.wcs
    expected = NDCube(data, wcs, uncertainty=uncertainty, mask=mask)
    # Compare outputs with expected.
    assert_cubes_equal(output, expected)
    """
    np.testing.assert_allclose(data, expected_data)
    assert (mask == expected_mask).all()
    assert isinstance(uncertainty, type(expected_uncertainty))
    np.testing.assert_allclose(uncertainty.array, expected_uncertainty.array)
    assert_quantity_allclose(coords[0], expected_coords[0])
    assert_quantity_allclose(coords[1], expected_coords[1])
    """
