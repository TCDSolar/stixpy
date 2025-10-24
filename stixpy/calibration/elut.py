from pathlib import Path

import numpy as np

from stixpy.calibration.detector import get_sci_channels
from stixpy.io.readers import read_elut, read_elut_index

__all__ = ["get_elut", "get_elut_correction"]

def get_elut(date):
    r"""
    Get the energy lookup table (ELUT) for the given date

    Combines the ELUT with the science energy channels for the same date.

    Parameters
    ----------
    date `astropy.time.Time`
        Date to look up the ELUT.

    Returns
    -------

    """
    root = Path(__file__).parent.parent
    elut_index_file = Path(root, *["config", "data", "elut", "elut_index.csv"])

    elut_index = read_elut_index(elut_index_file)
    elut_info = elut_index.at(date)
    if len(elut_info) == 0:
        raise ValueError(f"No ELUT for for date {date}")
    elif len(elut_info) > 1:
        raise ValueError(f"Multiple ELUTs for for date {date}")
    start_date, end_date, elut_file = list(elut_info)[0]
    sci_channels = get_sci_channels(date)
    elut_table = read_elut(elut_file, sci_channels)


    return elut_table

def get_elut_correction(e_ind, pixel_data):
    r"""
    Get ELUT correction factors

    Only correct the low and high energy edges as internal bins are contiguous.

    Parameters
    ----------
    e_ind : `np.ndarray`
        Energy channel indices
    pixel_data : `~stixpy.products.sources.CompressedPixelData`
        Pixel data

    Returns
    -------

    """

    energy_mask = pixel_data.energy_masks.energy_mask.astype(bool)
    elut = get_elut(pixel_data.time_range.center)
    ebin_edges_low = np.zeros((32, 12, 32), dtype=float)
    ebin_edges_low[..., 1:] = elut.e_actual
    ebin_edges_low = ebin_edges_low[..., energy_mask]
    ebin_edges_high = np.zeros((32, 12, 32), dtype=float)
    ebin_edges_high[..., 0:-1] = elut.e_actual
    ebin_edges_high[..., -1] = np.nan
    ebin_edges_high = ebin_edges_high[..., energy_mask]
    ebin_widths = ebin_edges_high - ebin_edges_low
    ebin_sci_edges_low = elut.e[..., 0:-1].value
    ebin_sci_edges_low = ebin_sci_edges_low[..., energy_mask]
    ebin_sci_edges_high = elut.e[..., 1:].value
    ebin_sci_edges_high = ebin_sci_edges_high[..., energy_mask]
    e_cor_low = (ebin_edges_high[..., e_ind[0]] - ebin_sci_edges_low[..., e_ind[0]]) / ebin_widths[..., e_ind[0]]
    e_cor_high = (ebin_sci_edges_high[..., e_ind[-1]] - ebin_edges_low[..., e_ind[-1]]) / ebin_widths[..., e_ind[-1]]

    numbers = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31
    ])

    bins =  ebin_sci_edges_high - ebin_sci_edges_low
    
    bins_actual = ebin_widths[numbers, :8, :].mean(axis=1).mean(axis=0)

    cor = bins / bins_actual
    
    return e_cor_high, e_cor_low, cor
