import astropy.nddata
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from ndcube import NDCube, NDMeta
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate

from stixpy.product.tools import rebin_by_edges


@pytest.fixture
def cube():
    data = np.ones((2, 5))
    mask = np.array([[False, False, False, True, False], [True, True, True, False, True]], dtype=bool)
    uncertainty = astropy.nddata.StdDevUncertainty(data * 0.1)

    time_coord = TimeTableCoordinate(
        Time("2000-01-01", scale="utc", format="iso") + np.arange(0, data.shape[0] * 10, 10) * u.s,
        names="time",
        physical_types="time",
    )
    energy_coord = QuantityTableCoordinate(
        np.arange(3, 3 + data.shape[1]) * u.keV, names="energy", physical_types="em.energy"
    )
    meta = NDMeta({"dt": [10, 10] * u.s, "e": np.arange(5), "other": 1}, axes={"dt": 0, "e": 1})
    return NDCube(
        data=data, wcs=(energy_coord & time_coord).wcs, mask=mask, uncertainty=uncertainty, meta=meta, unit=u.ct
    )


def test_rebin_by_edges(cube):
    data_before = cube.data.copy()
    rebinned = rebin_by_edges(
        cube,
        ([0, 2], [0, 1, 3, 5]),
        operation=np.sum,
        operation_ignores_mask=False,
        handle_mask=np.all,
        propagate_uncertainties=True,
    )
    np.testing.assert_allclose(rebinned.data, [[1.0, 2.0, 2.0]])
    assert (rebinned.mask == np.zeros((1, 3), dtype=bool)).all()
    assert isinstance(rebinned.uncertainty, astropy.nddata.StdDevUncertainty)
    np.testing.assert_allclose(rebinned.uncertainty.array, [[0.1, 0.14142135623730953, 0.14142135623730953]])
    assert rebinned.unit == u.ct
    assert rebinned.array_axis_physical_types == [("time",), ("em.energy",)]
    assert rebinned.axis_world_coords("time")[0].isot.tolist() == ["2000-01-01T00:00:05.000"]
    assert_quantity_allclose(rebinned.axis_world_coords("em.energy")[0], [3.0, 4.5, 6.5] * u.keV)
    # Generic NDMeta drops axis-awareness for rebinned axes
    assert dict(rebinned.meta.axes) == {}
    np.testing.assert_array_equal(rebinned.meta.data_shape, [1, 3])
    # Input is not modified
    np.testing.assert_array_equal(cube.data, data_before)


def test_rebin_by_edges_no_mask_skipped_axis(cube):
    cube._mask = None
    rebinned = rebin_by_edges(cube, (None, [1, 3]), operation=np.sum, propagate_uncertainties=True)
    assert rebinned.shape == (2, 1)
    assert rebinned.mask is None
    np.testing.assert_allclose(rebinned.data, [[2.0], [2.0]])
    assert_quantity_allclose(rebinned.axis_world_coords("em.energy")[0], [4.5] * u.keV)
    assert rebinned.axis_world_coords("time")[0].isot.tolist() == [
        "2000-01-01T00:00:00.000",
        "2000-01-01T00:00:10.000",
    ]
    assert dict(rebinned.meta.axes) == {"dt": (0,)}


def test_rebin_by_edges_keeps_extra_coords(cube):
    cube.extra_coords.add("duration", 0, [10, 20] * u.s, physical_types="time.duration")
    rebinned = rebin_by_edges(cube, ([0, 2], None), operation=np.sum)
    assert_quantity_allclose(rebinned.axis_world_coords("time.duration", wcs=rebinned.extra_coords)[0], [15] * u.s)


def test_rebin_by_edges_wrong_length(cube):
    with pytest.raises(ValueError, match="length of axes_idx_edges"):
        rebin_by_edges(cube, ([0, 2],))
