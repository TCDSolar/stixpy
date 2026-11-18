"""
Checks and helpers for `~stixpy.product.sources.science.subtract_background`.
"""

from dataclasses import replace

import numpy as np

import astropy.units as u

from stixpy.calibration.elut import get_elut
from stixpy.product.sources._selection import _on_indices

__all__ = []


def _bkg_check(data, bkg):
    """
    Check that a background product can be subtracted from selected science data.

    Parameters
    ----------
    data : SelectedData
        Science data from `~stixpy.product.sources.science.ScienceData.get_data`.
    bkg : ScienceData
        Background product.

    Raises
    ------
    TypeError
        If `bkg` is not a `ScienceData` product.
    ValueError
        If `data` holds summed pixels or is not livetime corrected; if the background is not pixel data with one time
        bin and one detector and pixel mask; if it was recorded with a different ELUT; if a science energy bin is not
        in the background; or if a detector or pixel used by the science data is off in the background.
    """
    from stixpy.product.sources.science import ScienceData  # here, as science imports this module

    if not isinstance(bkg, ScienceData):
        raise TypeError(f"bkg must be a STIX science product (ScienceData), got {type(bkg).__name__}.")

    selection = data.selection
    if selection.summed_pixels:
        raise ValueError(
            "Background subtraction is not supported for summed pixel data, as the summed pixels can not be "
            "matched to the background pixels."
        )
    if not selection.livetime_correction:
        raise ValueError(
            "Background subtraction needs the livetime correction, as the science and background counts are "
            "livetime corrected before the background is subtracted. Use get_data with livetime_correction=True."
        )

    if bkg.data["counts"].ndim != 4 or np.ndim(bkg.pixel_masks.masks) != 2:
        raise ValueError(f"The background must be pixel data (sci-xray-rpd or sci-xray-cpd), got {type(bkg).__name__}.")
    if len(bkg.data["time"]) != 1:
        raise ValueError(
            f"The background must have a single time bin, but it has {len(bkg.data['time'])}. Use a STIX "
            "background (BKG) file, which integrates the background over one long time bin."
        )
    if len(bkg.pixel_masks.masks) != 1:
        raise ValueError("The background pixel mask changes within the file.")

    # as STIX-GSW's stx_convert_pixel_data, which also refuses a background recorded under another ELUT
    if selection.elut_file is not None:
        bkg_elut = get_elut(bkg.time_range.center).file
        if bkg_elut != selection.elut_file:
            raise ValueError(
                f"The background was recorded with the ELUT {bkg_elut}, but the science data with "
                f"{selection.elut_file}. Use a background recorded under the same ELUT, closer in time to the "
                "science data."
            )

    # every science energy bin must start and end on a background bin edge
    bkg_low, bkg_high = bkg.energies["e_low"].to_value(u.keV), bkg.energies["e_high"].to_value(u.keV)
    missing = [
        f"{low:g}-{high:g}"
        for low, high in zip(data.energies["e_low"].to_value(u.keV), data.energies["e_high"].to_value(u.keV))
        if not (np.isclose(bkg_low, low).any() and np.isclose(bkg_high, high, equal_nan=True).any())
    ]
    if missing:
        raise ValueError(
            f"The science energy bins {missing} keV are not in the background, so it can not be subtracted. Select "
            "only energies the background has, with energy_indices."
        )

    if selection.ssw_compatible and selection.spectrogram:
        return

    detectors, pixels = _used(selection)
    missing_dets = np.setdiff1d(detectors, _on_indices(bkg.detector_masks.masks))
    missing_pix = np.setdiff1d(pixels, _on_indices(bkg.pixel_masks.masks))
    if missing_dets.size or missing_pix.size:
        used = "were summed on board into the spectrogram" if selection.spectrogram else "are selected"
        raise ValueError(
            f"Detectors {missing_dets.tolist()} and pixels {missing_pix.tolist()} {used} but are off in the "
            "background, so they can not be background subtracted. Detectors on in the background: "
            f"{_on_indices(bkg.detector_masks.masks).tolist()}, pixels on in the background: "
            f"{_on_indices(bkg.pixel_masks.masks).tolist()}."
        )


def _used(selection):
    """The detectors and pixels the selected science data hold."""
    return np.flatnonzero(selection.detectors.any(axis=0)), np.flatnonzero(selection.pixels)


def _bkg_selection(selection):
    """
    The selection to apply to the background, which is pixel data: for spectrogram science data, the detectors and
    pixels summed on board, summed, without the ELUT correction, which is not applied to spectrograms.
    """
    if not selection.spectrogram:
        return selection
    detectors, pixels = (None, None) if selection.ssw_compatible else _used(selection)
    return replace(
        selection,
        detector_indices=detectors,
        pixel_indices=pixels,
        checked_detectors=detectors,
        checked_pixels=pixels,
        sum_detectors=True,
        elut_correction=False,
        spectrogram=False,
    )


def _restrict(data, detectors, pixels):
    """
    Keep only `detectors` and `pixels` of loaded pixel data, so that ``[start, end]`` ranges and full axes take only
    what the science data hold, even where more detectors or pixels are on in the background.

    Parameters
    ----------
    data : _CountData
        Pixel data from `~stixpy.product.sources._countdata._load`, on the full detector and pixel grid.
    detectors, pixels : numpy.ndarray
        Detector and pixel numbers to keep.

    Returns
    -------
    _CountData
    """
    keep = np.zeros((1, 32, 12, 1), dtype=bool)
    keep[0, detectors[:, None], pixels, 0] = True
    return replace(
        data,
        counts=np.where(keep, data.counts, 0 * data.counts.unit),
        counts_err=np.where(keep, data.counts_err, 0 * data.counts_err.unit),
        detectors=data.detectors & np.isin(np.arange(32), detectors),
        pixels=data.pixels & np.isin(np.arange(12), pixels),
    )
