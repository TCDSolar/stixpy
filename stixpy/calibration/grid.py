"""
Grid Calibration
"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

__all__ = ["get_grid_transmission", "_calculate_grid_transmission"]

from stixpy.coordinates.frames import STIXImaging


def get_grid_transmission(flare_location: STIXImaging):
    r"""
    Return the grid transmission for the 32 sub-collimators corrected for internal shadowing.

    Read the grid tables and calculate the grid transmission for 32 sub-collimators.
    For the finest grids only transmission without correction for internal shadowing

    Parameters
    ----------
    flare_location :
        Location of the flare
    """
    column_names = ["sc", "p", "o", "phase", "slit", "grad", "rms", "thick", "bwidth", "bpitch"]

    root = Path(__file__).parent.parent
    grid_info = Path(root, *["config", "data", "grid"])
    front = Table.read(grid_info / "grid_param_front.txt", format="ascii", names=column_names)
    rear = Table.read(grid_info / "grid_param_rear.txt", format="ascii", names=column_names)

    transmission_front = _calculate_grid_transmission(front, flare_location)
    transmission_rear = _calculate_grid_transmission(rear, flare_location)
    total_transmission = transmission_front * transmission_rear

    # The finest grids are made from multiple layers for the moment remove these and set 1
    final_transmission = np.ones(32)
    sc = front["sc"]
    finest_scs = [11, 12, 13, 17, 18, 19]  # 1a, 2a, 1b, 2c, 1c, 2b
    idx = np.argwhere(np.isin(sc, finest_scs, invert=True)).ravel()
    final_transmission[sc[idx] - 1] = total_transmission[idx]

    return final_transmission


def _calculate_grid_transmission(grid_params, flare_location):
    r"""
    Calculate grid transmission accounting for internal shadowing.

    Parameters
    ----------
    grid_params
        Grid parameter tables
    flare_location
        Position of flare in stix imaging frame

    Returns
    -------
    Transmission through all grids defined by the grid parameters
    """
    orient = grid_params["o"] * u.deg  # As viewed from the detector side (data recorded from front)
    pitch = grid_params["p"]
    slit = grid_params["slit"]
    thick = grid_params["thick"]
    flare_dist = np.abs(flare_location.Tx * np.cos(orient)) + flare_location.Ty * np.sin(orient)
    shadow_width = thick * np.tan(flare_dist)
    transmission = (slit - shadow_width) / pitch
    return transmission
