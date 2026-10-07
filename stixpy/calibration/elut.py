from pathlib import Path
from typing import NamedTuple

import numpy as np

from stixpy.calibration.detector import get_sci_channels
from stixpy.io.readers import read_elut, read_elut_index

__all__ = ["ELUTCorrection", "get_elut", "get_elut_correction"]


class ELUTCorrection(NamedTuple):
    r"""
    Energy lookup table (ELUT) correction factors and bin widths.

    Returned by `get_elut_correction`. The fields can be accessed by name or unpacked in order.
    """

    e_cor_high: np.ndarray
    """Fraction of the actual width of the highest selected energy bin that falls inside the science bin, shape
    ``(32, 12)`` (detector, pixel)."""
    e_cor_low: np.ndarray
    """Fraction of the actual width of the lowest selected energy bin that falls inside the science bin, shape
    ``(32, 12)`` (detector, pixel)."""
    bins: np.ndarray
    """Science energy bin widths in keV, broadcast to shape ``(1, 32, 12, n_energy)``."""
    bins_actual: np.ndarray
    """Calibrated (actual) energy bin widths in keV for each detector and pixel, shape ``(1, 32, 12, n_energy)``."""


def get_elut(date):
    r"""
    Get the energy lookup table (ELUT) for the given date.

    Combines the ELUT with the science energy channels for the same date.

    Parameters
    ----------
    date : `datetime.datetime` or `astropy.time.Time`
        Date to look up the ELUT.

    Returns
    -------
    ELUT
        Object with the ELUT data as attributes. These include ``file`` (the ELUT file name), ``offset`` and
        ``gain`` with shape ``(32, 12)`` (detector, pixel), ``e_actual``, the calibrated energy edges in keV with
        shape ``(32, 12, 31)``, and ``e``, ``e_sci_low``, ``e_sci_high`` and ``e_width``, the science energy channel
        edges and widths for the date.

    Raises
    ------
    ValueError
        If there is no ELUT, or more than one ELUT, for the given date.
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
    Get ELUT correction factors.

    Only correct the low and high energy edges as internal bins are contiguous.

    Parameters
    ----------
    e_ind : `numpy.ndarray`
        Indices of the selected energy channels, after the energy mask has been applied.
    pixel_data : `~stixpy.product.sources.science.ScienceData`
        Pixel or spectrogram data. Must have ``energy_masks`` and ``time_range`` attributes.

    Returns
    -------
    `ELUTCorrection`
        Named tuple ``(e_cor_high, e_cor_low, bins, bins_actual)``. ``e_cor_high`` and ``e_cor_low`` are the
        fractions of the highest and lowest selected energy bins, shape ``(32, 12)`` (detector, pixel), that fall
        inside the science energy range. ``bins`` and ``bins_actual`` are the science and the calibrated energy bin
        widths in keV, shape ``(1, 32, 12, n_energy)``. ``bins / bins_actual`` gives a per-bin correction factor.
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

    bins = ebin_sci_edges_high - ebin_sci_edges_low
    n_energy = bins.shape[-1]
    bins = np.broadcast_to(bins.reshape(1, 1, 1, n_energy), (1, 32, 12, n_energy))
    bins_actual = ebin_widths[np.newaxis, ...]

    return ELUTCorrection(e_cor_high, e_cor_low, bins, bins_actual)
