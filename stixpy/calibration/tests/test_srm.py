import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u

from stixpy.calibration.srm import SRM, _count_edges_in_drm, _group_count_channels, get_srm
from stixpy.config.instrument import RCR_AREA

SCI_EDGES = [
    0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 70, 76, 84,
    100, 120, 150, np.inf,
] * u.keV  # fmt: skip
COARSE_EDGES = [4, 10, 20, 50, 84] * u.keV


@pytest.fixture(scope="module")
def srm_rcr0():
    return get_srm(COARSE_EDGES, [0, 1, 2], np.arange(12), 0)


def test_group_count_channels():
    ct_full = np.array([2.0, 3.0, 4.0, 5.0])
    drm = np.array([[1, 2, 100], [3, 4, 100]])

    assert_allclose(_group_count_channels(drm, ct_full, np.array([2.0, 4.0])), [[3], [7]])


def test_group_count_channels_edge_not_on_grid():
    with pytest.raises(ValueError, match="not on the saved DRM grid"):
        _group_count_channels(np.ones((4, 2)), np.array([2.0, 3.0, 4.0]), np.array([2.0, 3.5]))


def test_count_edges_in_drm():
    ct_full = np.array([4.0, 5.0, 150.0])
    assert_allclose(_count_edges_in_drm(SCI_EDGES.value, ct_full), SCI_EDGES.value[1:-1])
    assert_allclose(_count_edges_in_drm([0, 4, 10, 150, 200], ct_full), [4, 10, 150])
    with pytest.raises(ValueError, match="at least two edges"):
        _count_edges_in_drm([0, 4], ct_full)
    with pytest.raises(ValueError, match="increasing"):
        _count_edges_in_drm([4, 10, 5], ct_full)


def test_get_srm_shapes_and_units(srm_rcr0):
    assert isinstance(srm_rcr0, SRM)
    assert srm_rcr0.srm.unit == u.ct / (u.keV * u.ph)
    assert srm_rcr0.photon_edges.unit == u.keV
    assert srm_rcr0.geometric_area.unit == u.cm**2

    assert srm_rcr0.srm.shape == (srm_rcr0.photon_edges.size - 1, COARSE_EDGES.size - 1)
    assert_allclose(srm_rcr0.count_edges, COARSE_EDGES)
    assert srm_rcr0.photon_edges[0] >= 3.5 * u.keV
    assert np.all(np.isfinite(srm_rcr0.srm))
    assert np.all(srm_rcr0.srm >= 0)


def test_get_srm_science_channels():
    srm = get_srm(SCI_EDGES, 0, 0, 0)
    # the 0-4 keV and open top channels are left out
    assert_allclose(srm.count_edges, SCI_EDGES[1:-1])
    assert srm.srm.shape[1] == SCI_EDGES.size - 3


def test_get_srm_regroups_only_count_channels():
    fine = get_srm(SCI_EDGES, 0, 0, 0)
    coarse = get_srm([4, 6, 9] * u.keV, 0, 0, 0)
    assert_allclose(coarse.photon_edges, fine.photon_edges)
    integrated = fine.srm * np.diff(fine.count_edges)
    assert_allclose(
        coarse.srm * np.diff(coarse.count_edges),
        np.column_stack([integrated[:, :2].sum(axis=1), integrated[:, 2:5].sum(axis=1)]),
    )


def test_get_srm_ph_energy_min(srm_rcr0):
    full = get_srm(COARSE_EDGES, [0, 1, 2], np.arange(12), 0, ph_energy_min=None)
    n_cut = full.photon_edges.size - srm_rcr0.photon_edges.size
    assert n_cut > 0
    assert full.photon_edges[0] < 3.5 * u.keV
    assert_allclose(full.srm[n_cut:], srm_rcr0.srm)


def test_get_srm_geometric_area(srm_rcr0):
    # all pixels with no attenuator: the full area of each detector
    assert_allclose(srm_rcr0.geometric_area, 3 * RCR_AREA[0], rtol=1e-4)
    rcr3 = get_srm(COARSE_EDGES, [0, 1, 2], np.arange(12), 3)
    assert_allclose(rcr3.geometric_area, 3 * RCR_AREA[3], rtol=1e-4)


def test_get_srm_attenuator(srm_rcr0):
    rcr1 = get_srm(COARSE_EDGES, [0, 1, 2], np.arange(12), 1)
    low = srm_rcr0.photon_edges[:-1] < 10 * u.keV
    high = srm_rcr0.photon_edges[:-1] > 100 * u.keV
    # the attenuator cuts the response at low energies, and barely changes it at high energies
    assert np.all(rcr1.srm[low].sum(axis=1) < 0.5 * srm_rcr0.srm[low].sum(axis=1))
    assert_allclose(rcr1.srm[high], srm_rcr0.srm[high], rtol=0.05)


def test_get_srm_bkg_detector():
    # real_bkg_grid_transmission.txt: pixel 0 0.1341096, pixel 3 fully covered (0), pixel 7 0.1393076
    pix0 = get_srm(COARSE_EDGES, 9, 0, 0)
    pix3 = get_srm(COARSE_EDGES, 9, 3, 0)
    pix7 = get_srm(COARSE_EDGES, 9, 7, 0)
    assert np.all(pix3.srm == 0)
    assert_allclose(pix0.srm / 0.1341096, pix7.srm / 0.1393076)


def test_get_srm_cfl_detector():
    with pytest.raises(ValueError, match="CFL"):
        get_srm(COARSE_EDGES, [0, 8], 0, 0)


def test_get_srm_bkg_with_others():
    with pytest.raises(ValueError, match="BKG"):
        get_srm(COARSE_EDGES, [0, 9], 0, 0)


def test_get_srm_bad_rcr():
    with pytest.raises(ValueError, match="rcr"):
        get_srm(COARSE_EDGES, 0, 0, 8)
