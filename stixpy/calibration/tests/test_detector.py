from datetime import datetime

import numpy as np
import pytest
from numpy.testing import assert_equal

import astropy.units as u
from astropy.table import QTable

from stixpy.calibration.detector import get_sci_channels, tailing_matrix, tailing_matrix_corrected


def test_get_sci_channels():
    sci_channels = get_sci_channels(datetime.now())
    assert isinstance(sci_channels, QTable)
    assert len(sci_channels) == 33
    assert_equal(sci_channels["Elower"][1], 4.0 * u.keV)
    assert_equal(sci_channels["Eupper"][1], 5.0 * u.keV)

    sci_channels = get_sci_channels(datetime(2023, 1, 30, 12))
    assert isinstance(sci_channels, QTable)
    assert len(sci_channels) == 33
    assert_equal(sci_channels["Elower"][1], 4.0 * u.keV)
    assert_equal(sci_channels["Eupper"][1], 4.45 * u.keV)

    with pytest.raises(ValueError, match=r"No Science Energy.*"):
        get_sci_channels(datetime(2018, 1, 1))


@pytest.fixture
def cdte_xsec():
    # Smooth power law + constant stand-in for the CdTe attenuation coefficient (1/cm)
    energy = np.geomspace(1, 1000, 200) * u.keV
    return energy, (5e3 * (energy / (10 * u.keV)) ** -2.7 + 0.3) / u.cm


@pytest.mark.parametrize("n_bins", [2, 10, 59])
def test_tailing_matrix_shape(cdte_xsec, n_bins):
    edges = np.linspace(4, 150, n_bins + 1) * u.keV
    tm = tailing_matrix(edges, *cdte_xsec, n_layers=100)
    assert tm.shape == (n_bins, n_bins)


def test_tailing_matrix_conserves_or_loses_charge(cdte_xsec):
    edges = np.linspace(4, 150, 60) * u.keV
    tm = tailing_matrix(edges, *cdte_xsec)
    assert np.all(tm >= 0)
    # Columns are source bins: counts can only be lost or moved, never created
    assert np.all(tm.sum(axis=0) <= 1 + 1e-12)
    # Most counts stay in the source bin or the bins just below it
    assert np.all(tm.sum(axis=0)[1:-1] > 0.8)


def test_tailing_matrix_only_moves_down_in_energy(cdte_xsec):
    edges = np.geomspace(4, 150, 40) * u.keV
    tm = tailing_matrix(edges, *cdte_xsec)
    # tm[dest, src] with dest > src would move counts up in energy
    assert_equal(np.tril(tm, k=-1), 0)
    assert np.any(np.triu(tm, k=1) > 0)


def test_tailing_matrix_weak_trapping_is_near_diagonal(cdte_xsec):
    edges = np.linspace(4, 150, 60) * u.keV
    strong = tailing_matrix(edges, *cdte_xsec)
    weak = tailing_matrix(edges, *cdte_xsec, trap_length_h=1 * u.km, trap_length_e=1 * u.km, r0=0)
    assert np.all(np.diag(weak)[:-1] > 0.9)
    assert np.all(np.diag(weak)[:-1] >= np.diag(strong)[:-1])


def test_tailing_matrix_units(cdte_xsec):
    energy, xsec = cdte_xsec
    edges = np.linspace(4, 150, 30) * u.keV
    expected = tailing_matrix(edges, energy, xsec, n_layers=200)
    # Equivalent units give the same matrix
    other = tailing_matrix(
        edges.to(u.eV), energy.to(u.MeV), xsec.to(1 / u.mm), depth=0.1 * u.cm, trap_length_h=3600 * u.um, n_layers=200
    )
    np.testing.assert_allclose(other, expected)
    with pytest.raises(u.UnitsError):
        tailing_matrix(edges.value * u.cm, energy, xsec)


@pytest.mark.parametrize("n_bins", [2, 10, 59])
def test_tailing_matrix_corrected_shape(cdte_xsec, n_bins):
    edges = np.linspace(4, 150, n_bins + 1) * u.keV
    tm = tailing_matrix_corrected(edges, *cdte_xsec, n_layers=100)
    assert tm.shape == (n_bins, n_bins)


@pytest.mark.parametrize(
    "edges",
    [
        np.arange(4, 151, 1.0) * u.keV,
        np.geomspace(4, 150, 60) * u.keV,
        [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 32, 40, 50, 63, 76, 100, 150] * u.keV,
    ],
)
def test_tailing_matrix_corrected_conserves_counts(cdte_xsec, edges):
    tm = tailing_matrix_corrected(edges, *cdte_xsec)
    assert np.all(tm >= 0)
    assert_equal(np.tril(tm, k=-1), 0)
    col_sums = tm.sum(axis=0)
    assert np.all(col_sums <= 1 + 1e-12)
    # Counts are only lost below the lowest edge; the damage layer collects at least 1 - r0 = 0.2,
    # so bins starting above 5 * 4 keV lose nothing, including the highest bin
    np.testing.assert_allclose(col_sums[edges[:-1] >= 20 * u.keV], 1, rtol=1e-10)


def test_tailing_matrix_corrected_weak_trapping_is_identity(cdte_xsec):
    edges = np.geomspace(4, 150, 40) * u.keV
    tm = tailing_matrix_corrected(edges, *cdte_xsec, trap_length_h=1 * u.km, trap_length_e=1 * u.km, r0=0)
    np.testing.assert_allclose(tm, np.eye(39), atol=1e-4)


def test_tailing_matrix_corrected_close_to_idl(cdte_xsec):
    # On a uniform grid the IDL discretisation errors are small, apart from the empty highest bin
    edges = np.arange(4, 151, 1.0) * u.keV
    idl = tailing_matrix(edges, *cdte_xsec)
    corrected = tailing_matrix_corrected(edges, *cdte_xsec)
    assert np.abs(corrected - idl)[:, :-1].max() < 0.05
    assert_equal(idl[:, -1], 0)
    assert corrected[-1, -1] > 0
