from pathlib import Path

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from stixpy.calibration.grid import CFL_BKG_0, FINEST_ALL_0, get_grid_transmission
from stixpy.coordinates.frames import STIXImaging

GRID_DIR = Path(__file__).parent.parent.parent / "config" / "data" / "grid"
ON_AXIS = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=STIXImaging)
IMAGING = np.setdiff1d(np.arange(32), CFL_BKG_0)


@pytest.fixture
def intercepts():
    return np.asarray(Table.read(GRID_DIR / "stix_subcoll_transmission_10_15keV.csv", format="ascii.csv")["intercept"])


def test_grid_transmission_shape():
    transmission = get_grid_transmission([10, 20, 30] * u.keV, [0, 5, 10, 20], ON_AXIS)
    assert transmission.shape == (3, 4)


def test_grid_transmission_requires_quantity():
    with pytest.raises(TypeError):
        get_grid_transmission([10, 20], IMAGING, ON_AXIS)


def test_grid_transmission_on_axis_matches_calibration(intercepts):
    # The calibration was measured at 10-15 keV, where the grids are close to opaque, so the model should match the
    # measured on-axis transmission to within a few percent
    transmission = get_grid_transmission(12.5 * u.keV, IMAGING, ON_AXIS)[0]
    assert np.allclose(transmission, intercepts[IMAGING], rtol=0.05)


def test_grid_transmission_finest_grids(intercepts):
    transmission = get_grid_transmission([12.5, 100] * u.keV, FINEST_ALL_0, ON_AXIS)
    assert np.all(np.isfinite(transmission))
    assert np.all((transmission > 0) & (transmission < 1))
    # No shadowing term for the finest grids so they match the calibration closely at low energy
    assert np.allclose(transmission[0], intercepts[FINEST_ALL_0], rtol=1e-3)
    assert np.all(transmission[1] > transmission[0])


def test_grid_transmission_increases_with_energy():
    # Not monotonic due to the tungsten absorption edges (e.g. K-edge at 69.5 keV) so compare well separated energies
    energies = [15, 30, 60, 200, 1000] * u.keV
    transmission = get_grid_transmission(energies, IMAGING, ON_AXIS)
    assert np.all(np.diff(transmission[[0, 1, 2]], axis=0) > 0)
    assert np.all(np.diff(transmission[[2, 3, 4]], axis=0) > 0)
    assert np.all(transmission[-1] > 0.9)
    assert np.all(transmission <= 1)


def test_grid_transmission_cfl_bkg():
    nominal = Table.read(GRID_DIR / "nom_bkg_grid_transmission.txt", format="ascii.no_header", comment="[;~]")["col1"]
    transmission = get_grid_transmission([10, 50, 100] * u.keV, CFL_BKG_0, ON_AXIS)
    assert np.all(transmission == np.asarray(nominal[CFL_BKG_0]))


def test_grid_transmission_flare_location_inputs():
    on_axis_none = get_grid_transmission(20 * u.keV, IMAGING)
    on_axis_coord = get_grid_transmission(20 * u.keV, IMAGING, ON_AXIS)
    on_axis_quantity = get_grid_transmission(20 * u.keV, IMAGING, [0, 0] * u.arcsec)
    assert np.allclose(on_axis_none, on_axis_coord)
    assert np.allclose(on_axis_none, on_axis_quantity)

    off_axis_coord = get_grid_transmission(
        20 * u.keV, IMAGING, SkyCoord(300 * u.arcsec, 500 * u.arcsec, frame=STIXImaging)
    )
    off_axis_quantity = get_grid_transmission(20 * u.keV, IMAGING, [5, 500 / 60] * u.arcmin)
    assert np.allclose(off_axis_coord, off_axis_quantity)
    assert not np.allclose(off_axis_coord, on_axis_coord)


def test_grid_transmission_bad_flare_location():
    with pytest.raises(ValueError, match="STIXImaging"):
        get_grid_transmission(20 * u.keV, IMAGING, SkyCoord(0 * u.deg, 0 * u.deg, frame="icrs"))
    with pytest.raises(ValueError, match="shape"):
        get_grid_transmission(20 * u.keV, IMAGING, [0, 0, 0] * u.arcsec)


def test_grid_transmission_non_positive():
    # sc 13 (1b) has a small intercept (0.0384), a positive slope (0.0093 / deg) and is oriented at ~90 deg, so a
    # large enough negative Ty gives a negative transmission
    with pytest.raises(ValueError, match="<= 0"):
        get_grid_transmission(20 * u.keV, [12], [0, -5] * u.deg)
