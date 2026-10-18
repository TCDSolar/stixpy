import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import astropy.units as u
from astropy.table import QTable
from astropy.time import Time

from stixpy.product.sources._countdata import DETECTOR, ENERGY, PIXEL, TIME, _CountData

N_TIME, N_DET, N_PIX, N_ENERGY = 4, 3, 2, 5


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    shape = (N_TIME, N_DET, N_PIX, N_ENERGY)
    return _CountData(
        counts=rng.uniform(0, 100, shape) * u.ct,
        counts_err=rng.uniform(1, 10, shape) * u.ct,
        livefrac=rng.uniform(0.5, 1, (N_TIME, N_DET, 1, 1)),
        livefrac_err=rng.uniform(0, 1, shape) * u.ct,
        elut_cor_fac=rng.uniform(0.9, 1.1, (1, N_DET, 1, N_ENERGY)),
        times=Time("2024-03-10T12:00:00") + [0, 2, 6, 10] * u.s,
        durations=[2, 2, 6, 2] * u.s,
        energies=QTable(
            {
                "channel": np.arange(N_ENERGY),
                "e_low": [4, 5, 6, 8, 10] * u.keV,
                "e_high": [5, 6, 8, 10, 15] * u.keV,
            }
        ),
        detectors=np.eye(N_DET, 32, dtype=bool),
        pixels=np.eye(N_PIX, 12, dtype=bool),
        rcr=np.array([0, 0, 1, 1]),
    )


def test_getitem_flat_indices(data):
    sel = data[[0, 2], [2, 0], [1], [1, 3, 4]]
    assert_array_equal(sel.counts, data.counts.value[[0, 2]][:, [2, 0]][:, :, [1]][..., [1, 3, 4]] * u.ct)
    assert_array_equal(sel.counts_err, data.counts_err[[0, 2]][:, [2, 0]][:, :, [1]][..., [1, 3, 4]])
    assert_array_equal(sel.livefrac_err, data.livefrac_err[[0, 2]][:, [2, 0]][:, :, [1]][..., [1, 3, 4]])
    # length-1 axes are left alone
    assert_array_equal(sel.livefrac, data.livefrac[[0, 2]][:, [2, 0]])
    assert_array_equal(sel.times.mjd, data.times[[0, 2]].mjd)
    assert_array_equal(sel.durations, [2, 6] * u.s)
    assert_array_equal(sel.rcr, [0, 1])
    assert_array_equal(sel.detectors, data.detectors[[2, 0]])
    assert_array_equal(sel.pixels, data.pixels[[1]])
    assert_array_equal(sel.energies["channel"], [1, 3, 4])


def test_getitem_elut_not_selected_on_detectors(data):
    # the ELUT factor has never been selected over detectors (see _REDUCTIONS)
    sel = data[:, [0, 1], :, [0, 1]]
    assert_array_equal(sel.elut_cor_fac, data.elut_cor_fac[..., [0, 1]])


def test_getitem_boolean_mask_and_int(data):
    mask = np.array([True, False, True, True])
    sel = data[mask]
    assert sel.counts.shape == (3, N_DET, N_PIX, N_ENERGY)
    assert_array_equal(sel.rcr, [0, 1, 1])

    single = data[:, :, :, 2]
    assert single.counts.shape == (N_TIME, N_DET, N_PIX, 1)
    assert_array_equal(single.energies["channel"], [2])

    last = data[-1]
    assert last.counts.shape == (1, N_DET, N_PIX, N_ENERGY)
    assert_array_equal(last.counts, data.counts[-1:])


def test_getitem_none_fields(data):
    plain = _CountData(**{**vars(data), "livefrac": None, "livefrac_err": None, "elut_cor_fac": None})
    sel = plain[[1], [0], [0], [0]]
    assert sel.livefrac is None
    assert sel.livefrac_err is None
    assert sel.elut_cor_fac is None
    assert sel.counts.shape == (1, 1, 1, 1)


def test_getitem_returns_new_object(data):
    sel = data[[0]]
    assert sel is not data
    assert data.counts.shape == (N_TIME, N_DET, N_PIX, N_ENERGY)


def test_combine_energy(data):
    out = data.combine(ENERGY, [[0, 1], [2, 4]])
    assert out.counts.shape == (N_TIME, N_DET, N_PIX, 2)
    assert_allclose(out.counts[..., 1], data.counts[..., 2:].sum(axis=-1))
    assert_allclose(out.counts_err[..., 0], np.sqrt((data.counts_err[..., :2] ** 2).sum(axis=-1)))
    # livetime uncertainty is summed linearly over energies
    assert_allclose(out.livefrac_err[..., 1], data.livefrac_err[..., 2:].sum(axis=-1))
    assert_allclose(out.elut_cor_fac[..., 0], data.elut_cor_fac[..., :2].mean(axis=-1))
    assert_array_equal(out.livefrac, data.livefrac)
    assert out.energies.colnames == ["e_low", "e_high"]
    assert_array_equal(out.energies["e_low"], [4, 6] * u.keV)
    assert_array_equal(out.energies["e_high"], [6, 15] * u.keV)


def test_combine_detector(data):
    out = data.combine(DETECTOR, [[0, 2]])
    assert_allclose(out.counts[:, 0], data.counts.sum(axis=1))
    assert_allclose(out.counts_err[:, 0], np.sqrt((data.counts_err**2).sum(axis=1)))
    assert_allclose(out.livefrac[:, 0], data.livefrac.mean(axis=1))
    # livetime uncertainty is combined as an RMS over detectors
    assert_allclose(out.livefrac_err[:, 0], np.sqrt((data.livefrac_err**2).mean(axis=1)))
    # the ELUT factor is not combined over detectors
    assert_array_equal(out.elut_cor_fac, data.elut_cor_fac)
    assert_array_equal(out.detectors, data.detectors.any(axis=0, keepdims=True))


def test_combine_pixel(data):
    out = data.combine(PIXEL, [[0, 1]])
    assert_allclose(out.counts[:, :, 0], data.counts.sum(axis=2))
    assert_allclose(out.livefrac_err[:, :, 0], np.sqrt((data.livefrac_err**2).sum(axis=2)))
    assert_array_equal(out.livefrac, data.livefrac)
    assert_array_equal(out.pixels, data.pixels.any(axis=0, keepdims=True))


def test_combine_time(data):
    out = data.combine(TIME, [[0, 1], [2, 3]])
    assert_allclose(out.counts[0], data.counts[:2].sum(axis=0))
    assert_allclose(out.livefrac_err[1], np.sqrt((data.livefrac_err[2:] ** 2).sum(axis=0)))
    # livetime weighted by duration
    assert_allclose(out.livefrac[1], (data.livefrac[2] * 6 + data.livefrac[3] * 2) / 8)
    # the bins span 11:59:59-12:00:03 and 12:00:03-12:00:11, centred and timed from their edges
    assert_allclose(out.durations.to_value(u.s), [4, 8])
    assert_allclose((out.times - Time("2024-03-10T12:00:00")).to_value(u.s), [1, 7], atol=1e-6)
    assert_array_equal(out.rcr, [0, 1])
    assert out.elut_cor_fac.shape == data.elut_cor_fac.shape


def test_combine_matches_numpy_on_one_bin_ranges(data):
    out = data.combine(ENERGY, [[1, 1], [3, 3]])
    sel = data[:, :, :, [1, 3]]
    assert_allclose(out.counts, sel.counts)
    assert_allclose(out.counts_err, sel.counts_err)
    assert_allclose(out.livefrac_err, sel.livefrac_err)


def test_combine_range_clipped_to_axis(data):
    # a range running past the end of an axis takes what is there, as numpy slicing does
    out = data.combine(PIXEL, [[0, 11]])
    assert_allclose(out.counts[:, :, 0], data.counts.sum(axis=2))
