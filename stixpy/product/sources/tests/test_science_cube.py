import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from stixpy.data import test
from stixpy.product import Product
from stixpy.product.sources.science_cube import ScienceCube


@pytest.fixture
def cpd():
    return Product(test.STIX_SCI_XRAY_CPD)


@pytest.fixture
def cpd_cube(cpd):
    return ScienceCube.from_product(cpd)


def test_axes_and_coords(cpd, cpd_cube):
    assert cpd_cube.shape == (5, 32, 12, 32)
    assert cpd_cube.array_axis_physical_types == [
        ("time",),
        ("custom:detector",),
        ("custom:pixel",),
        ("em.energy",),
    ]
    assert np.all(cpd_cube.axis_world_coords("time")[0] == cpd.data["time"])
    # Overflow channel has e_high = inf, coordinate falls back to e_low
    assert_allclose(cpd_cube.axis_world_coords("em.energy")[0][[0, 1, -1]], [2, 4.5, 150] * u.keV)


def test_slice_keeps_meta_and_coords_aligned(cpd, cpd_cube):
    sliced = cpd_cube[1:3, 0:4, :, 5:10]
    assert isinstance(sliced, ScienceCube)
    assert sliced.meta["timedel"].shape == (2,)
    assert sliced.meta["detector_masks"].shape == (2, 4)
    assert sliced.meta["pixel_masks"].shape == (2, 12)
    assert sliced.meta["triggers"]["triggers"].shape == (2, 16)
    assert_allclose(sliced.meta["e_low"], cpd.energies["e_low"][5:10])
    assert np.all(sliced.times == cpd.data["time"][1:3])
    e_centres = (cpd.energies["e_low"] + cpd.energies["e_high"])[5:10] / 2
    assert_allclose(sliced.axis_world_coords("em.energy")[0], e_centres)


def test_sum_ranges(cpd, cpd_cube):
    summed = (
        cpd_cube.sum_ranges("detector", [[0, 31]])
        .sum_ranges("pixel", [[0, 11]])
        .sum_ranges("energy", [[1, 30]])
        .sum_ranges("time", [[0, 1], [2, 4]])
    )
    assert summed.shape == (2, 1, 1, 1)
    assert_allclose(summed.meta["timedel"], [8, 12] * u.s)
    assert_allclose(summed.meta["e_low"], [4] * u.keV)
    assert_allclose(summed.meta["e_high"], [150] * u.keV)
    assert summed.meta["detector_index"][0] == tuple(range(32))
    # Times are the centre of the combined bins
    assert np.all(summed.times == summed.axis_world_coords("time")[0])
    assert_allclose((summed.times - cpd.data["time"][[0, 2]]).to(u.s), [2, 4] * u.s, atol=1e-6 * u.s)

    counts = cpd.data["counts"].value.astype(float)[..., 1:31]
    expected = np.array([counts[0:2].sum(), counts[2:5].sum()])
    assert_allclose(summed.data.ravel(), expected)

    expected_err = np.sqrt(
        expected
        + np.array([(cpd.data["counts_err"].value[i, ..., 1:31] ** 2).sum() for i in [slice(0, 2), slice(2, 5)]])
    )
    assert_allclose(summed.uncertainty.array.ravel(), expected_err)


def test_to_rate(cpd, cpd_cube):
    summed = cpd_cube.sum_ranges("energy", [[1, 30]])
    dcr = summed.to_rate("dcr")
    assert dcr.unit == u.ct / (u.s * u.keV)
    expected = summed.data / (cpd.data["timedel"].to_value(u.s).reshape(-1, 1, 1, 1) * 146)
    assert_allclose(dcr.data, expected)
    assert cpd_cube.to_rate("cr").unit == u.ct / u.s
    with pytest.raises(ValueError, match="vtype"):
        cpd_cube.to_rate("x")


def test_summed_pixel_data():
    cube = ScienceCube.from_product(Product(test.STIX_SCI_XRAY_SCPD))
    sliced = cube[2:4]
    # Summed pixel masks (time, pixel_set, 12) are tied to the time axis only
    assert sliced.meta["pixel_masks"]["pixel_masks"].shape == (2, 4, 12)


def test_spectrogram():
    cube = ScienceCube.from_product(Product(test.STIX_SCI_XRAY_SPEC))
    assert cube.array_axis_physical_types == [("time",), ("em.energy",)]
    summed = cube.sum_ranges("energy", [[1, 10], [11, 20]])
    assert summed.shape == (5, 2)
    assert_allclose(summed.meta["e_low"], [4, 14] * u.keV)
    assert_allclose(summed.meta["e_high"], [14, 40] * u.keV)


def test_ndcube_rebin_uses_stix_meta(cpd, cpd_cube):
    data_before = cpd_cube.data.copy()
    rebinned = cpd_cube.rebin((5, 32, 12, 1))
    assert isinstance(rebinned, ScienceCube)
    assert rebinned.shape == (1, 1, 1, 32)
    assert_allclose(rebinned.meta["timedel"], [20] * u.s)
    assert rebinned.meta["detector_masks"].shape == (1, 1)
    assert rebinned.meta["triggers"]["triggers"].shape == (1, 16)
    assert_allclose(rebinned.meta["triggers"]["triggers"], cpd.data["triggers"].sum(axis=0, keepdims=True))
    assert rebinned.times == rebinned.axis_world_coords("time")[0]
    assert_allclose((rebinned.times - cpd.data["time"][0]).to(u.s), 8 * u.s, atol=1e-6 * u.s)
    assert_allclose(rebinned.data, cpd.data["counts"].value.sum(axis=(0, 1, 2), keepdims=True))
    np.testing.assert_array_equal(cpd_cube.data, data_before)


def test_rebin_by_edges_matches_sum_ranges(cpd_cube):
    by_edges = cpd_cube.rebin_by_edges([[0, 2, 5], [0, 32], [0, 12], [1, 31]])
    by_ranges = (
        cpd_cube.sum_ranges("time", [[0, 1], [2, 4]])
        .sum_ranges("detector", [[0, 31]])
        .sum_ranges("pixel", [[0, 11]])
        .sum_ranges("energy", [[1, 30]])
    )
    assert_allclose(by_edges.data, by_ranges.data)
    assert_allclose(by_edges.uncertainty.array, by_ranges.uncertainty.array)
    # Energy coordinate from the combined edges (4-150 keV), not the mean of the original centres
    assert_allclose(by_edges.axis_world_coords("em.energy")[0], [77] * u.keV)


def test_sum_ranges_with_gaps(cpd, cpd_cube):
    summed = cpd_cube.sum_ranges("time", [[0, 1], [3, 4]])
    assert summed.shape == (2, 32, 12, 32)
    assert_allclose(summed.meta["timedel"], [8, 8] * u.s)
    counts = cpd.data["counts"].value
    assert_allclose(summed.data[1], counts[3:5].sum(axis=0))
    assert_allclose((summed.times - cpd.data["time"][[0, 3]]).to(u.s), [2, 2] * u.s, atol=1e-6 * u.s)
    with pytest.raises(ValueError, match="non-overlapping"):
        cpd_cube.sum_ranges("time", [[0, 2], [1, 4]])
