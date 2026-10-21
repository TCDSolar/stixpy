"""
Background subtraction for `~stixpy.product.sources.science.ScienceData.get_data`.
"""

from dataclasses import replace

import numpy as np

import astropy.units as u

from stixpy.product.sources._corrections import _add_livetime, _livetime_uncertainty
from stixpy.product.sources._countdata import _load, _trigger_correlated
from stixpy.product.sources._selection import DETECTOR, PIXEL, _flat_indices, _on_indices

__all__ = []


def _bkg_check(product, bkg, detector_indices, pixel_indices, livetime_correction):
    """
    Check that a background product can be subtracted from a science product.

    Parameters
    ----------
    product : ScienceData
        Science product, pixel data or a spectrogram.
    bkg : ScienceData
        Background product.
    detector_indices : numpy.ndarray or None
        Science detector selection from `_indices_check`, flat or [start, end]
        pairs. None for spectrogram products.
    pixel_indices : numpy.ndarray or None
        Science pixel selection from `_indices_check`, flat or [start, end]
        pairs. None for spectrogram products.
    livetime_correction : bool
        The ``livetime_correction`` given to ``get_data``.

    Raises
    ------
    TypeError
        If `bkg` is not a `ScienceData` product.
    ValueError
        If `livetime_correction` is False; if the science product holds summed
        pixels; if the background is not pixel data with one time bin and one
        detector and pixel mask; if a science energy bin is not in the
        background; or if a detector or pixel used by the science data is off in
        the background.
    """
    from stixpy.product.sources.science import ScienceData  # here, as science imports this module

    if not isinstance(bkg, ScienceData):
        raise TypeError(f"bkg must be a STIX science product (ScienceData), got {type(bkg).__name__}.")

    if not livetime_correction:
        raise ValueError(
            "Background subtraction needs the livetime correction, as the science and background counts are "
            "livetime corrected before the background is subtracted. Leave livetime_correction=True with bkg."
        )

    is_spectrogram = product.data["counts"].ndim < 4
    if not is_spectrogram and np.ndim(product.pixel_masks.masks) != 2:
        raise ValueError(
            "Background subtraction is not supported for summed pixel data, as the summed pixels can not be "
            "matched to the background pixels."
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

    # every science energy bin must be in the background, with the same edges
    sci_low, sci_high = product.energies["e_low"].to_value(u.keV), product.energies["e_high"].to_value(u.keV)
    bkg_low, bkg_high = bkg.energies["e_low"].to_value(u.keV), bkg.energies["e_high"].to_value(u.keV)
    missing = []
    for low, high in zip(sci_low, sci_high):
        match = np.flatnonzero(np.isclose(bkg_low, low))
        if match.size != 1 or not np.isclose(bkg_high[match[0]], high, equal_nan=True):
            missing.append(f"{low:g}-{high:g}")
    if missing:
        raise ValueError(
            f"The science energy bins {missing} keV are not in the background, so it can not be subtracted. Use "
            "a background with the same energy binning as the science data."
        )

    # every detector and pixel the science data use must be on in the background
    if is_spectrogram:
        detectors = _on_indices(product.detector_masks.masks)
        pixels = _on_indices(product.pixel_masks.masks)
        used = "were summed on board into the spectrogram"
    else:
        detectors = np.intersect1d(_flat_indices(detector_indices), _on_indices(product.detector_masks.masks))
        pixels = np.intersect1d(_flat_indices(pixel_indices), _on_indices(product.pixel_masks.masks))
        used = "are selected"
    missing_dets = np.setdiff1d(detectors, _on_indices(bkg.detector_masks.masks))
    missing_pix = np.setdiff1d(pixels, _on_indices(bkg.pixel_masks.masks))
    if missing_dets.size or missing_pix.size:
        raise ValueError(
            f"Detectors {missing_dets.tolist()} and pixels {missing_pix.tolist()} {used} but are off in the "
            "background, so they can not be background subtracted. Detectors on in the background: "
            f"{_on_indices(bkg.detector_masks.masks).tolist()}, pixels on in the background: "
            f"{_on_indices(bkg.pixel_masks.masks).tolist()}."
        )


def _load_background(bkg, product, data):
    """
    Load a background product for subtraction from the science data.

    The background is loaded as the science data are (see `_load`), cut down to the science energy bins, and given
    its livetime, with the uncertainty computed on ELUT corrected counts if the science data are ELUT corrected.
    For a spectrogram it is also cut down to the detectors and pixels summed on board that are on in both products.

    Parameters
    ----------
    bkg : ScienceData
        Background product, already checked with `_bkg_check`.
    product : ScienceData
        Science product.
    data : _CountData
        The science data, from `_load`, with any ELUT correction.

    Returns
    -------
    _CountData
        The background, without ELUT correction.
    """
    bkg_data = _load(bkg, data.error_model)
    _, _, matched = np.intersect1d(data.energies["e_low"], bkg_data.energies["e_low"], return_indices=True)
    bkg_data = bkg_data[:, :, :, matched]

    pixels, detectors = _bkg_indices_check(product, bkg)
    bkg_data = _add_livetime(bkg_data, bkg, pixels, elut_cor_fac=data.elut_cor_fac)
    if data.counts.shape[DETECTOR] == 1:
        bkg_data = bkg_data[:, detectors, pixels]
    return bkg_data


def _subtract_background(data, bkg_data, detector_groups, pixel_indices):
    """
    Subtract a background from the science data, as STIX-GSW's ``stx_convert_pixel_data`` and
    ``stx_convert_spectrogram`` do.

    Both are livetime corrected, and the background is ELUT corrected with the science factor. The background is
    converted to a count rate over its time bin, scaled to the duration of each science time bin and subtracted.
    Uncertainties are combined in quadrature, with the livetime uncertainty of each folded in by
    `_livetime_uncertainty`. With the "correlated" error model the science livetime uncertainty and the scaled
    background uncertainty are kept apart instead, in `livefrac_err` and `bkg_err`, as the background uncertainty is
    the same measurement in every time bin; the livetime uncertainties are summed linearly over pixels, and over the
    detectors on one trigger counter.

    The result is then scaled back from livetime-corrected counts:

    - spectrogram products: by the science livetime fraction;
    - pixel data: by the mean livetime fraction of each group of detectors, as `_apply_livetime` does without a
      background. Detectors and pixels that are not selected are set to zero, and the pixels are summed.

    Parameters
    ----------
    data : _CountData
        Science data from `_load`, with any ELUT correction and the livetime (see `_add_livetime`).
    bkg_data : _CountData
        Background from `_load_background`.
    detector_groups : list of numpy.ndarray
        Detectors to subtract from, in groups that share one livetime (see `_livetime_groups`). Not used for
        spectrogram products.
    pixel_indices : numpy.ndarray or None
        Pixels to subtract from, flat or [start, end] pairs. Not used for spectrogram products.

    Returns
    -------
    _CountData
        Livetime corrected, with one pixel position. With the "legacy" error model there is no livetime
        uncertainty, as it is in `counts_err`.
    """
    elut_cor_fac = data.elut_cor_fac
    is_spectrogram = data.counts.shape[DETECTOR] == 1
    legacy = data.error_model == "legacy"

    bkg_err = bkg_data.counts_err
    livefrac_err = None
    if is_spectrogram:
        if elut_cor_fac is not None:
            bkg_err = bkg_err * elut_cor_fac
        # the counts were summed on board over the detectors and pixels, so sum the background over them too
        if legacy:
            bkg_err = _livetime_uncertainty(bkg_err, bkg_data.livefrac_err, bkg_data.livefrac)
            bkg_err = np.sqrt(np.nansum(bkg_err**2, axis=(DETECTOR, PIXEL), keepdims=True))
        else:
            bkg_stat = np.sqrt(np.nansum((bkg_err / bkg_data.livefrac) ** 2, axis=(DETECTOR, PIXEL), keepdims=True))
            bkg_livetime = np.nansum(bkg_data.livefrac_err, axis=PIXEL, keepdims=True)
            bkg_livetime = _trigger_correlated(bkg_livetime, DETECTOR, 0, bkg_livetime.shape[DETECTOR] - 1, bkg_data)
            bkg_err = np.sqrt(bkg_stat**2 + bkg_livetime**2)
        bkg_counts = np.nansum(bkg_data.counts, axis=(DETECTOR, PIXEL), keepdims=True)
        bkg_livefrac = np.nanmean(bkg_data.livefrac, axis=DETECTOR, keepdims=True)

        livefrac = np.nanmean(data.livefrac, axis=DETECTOR, keepdims=True)
        if legacy:
            livefrac_err = np.nanmean(data.livefrac_err, axis=(DETECTOR, PIXEL), keepdims=True)
            counts_err = _livetime_uncertainty(data.counts_err, livefrac_err, livefrac)
            livefrac_err = None
        else:
            counts_err = data.counts_err / livefrac
            livefrac_err = data.livefrac_err
    else:
        pixels_on = np.flatnonzero(data.pixels.any(axis=0))
        pixels = _flat_indices(pixel_indices).astype(int)
        if np.ndim(pixel_indices) == 2:
            pixels = np.intersect1d(pixels, pixels_on)
        detectors = np.concatenate(detector_groups)

        bkg_err = bkg_err[:, :, pixels, :]
        if elut_cor_fac is not None:
            bkg_err = bkg_err * elut_cor_fac

        bkg_counts = bkg_data.counts
        bkg_livefrac = bkg_data.livefrac
        livefrac = data.livefrac
        if legacy:
            bkg_err = _livetime_uncertainty(bkg_err, _pixels(bkg_data.livefrac_err, pixels), bkg_data.livefrac)
            counts_err = _livetime_uncertainty(
                data.counts_err[:, :, pixels, :], _pixels(data.livefrac_err, pixels), livefrac
            )
        else:
            # one background measurement, so its statistical and livetime parts are added together here
            bkg_stat = np.sqrt(np.nansum((bkg_err / bkg_livefrac) ** 2, axis=PIXEL, keepdims=True))
            bkg_livetime = np.nansum(bkg_data.livefrac_err[:, :, pixels, :], axis=PIXEL, keepdims=True)
            bkg_err = np.sqrt(bkg_stat**2 + bkg_livetime**2)
            counts_err = np.sqrt(
                np.nansum((data.counts_err[:, :, pixels, :] / livefrac) ** 2, axis=PIXEL, keepdims=True)
            )
            livefrac_err = np.nansum(data.livefrac_err[:, :, pixels, :], axis=PIXEL, keepdims=True)

    t_norm = data.durations.to(u.s).reshape(-1, 1, 1, 1)
    t_norm_bkg = bkg_data.durations.to(u.s).mean()

    counts_lvtcorr = data.counts / livefrac
    counts_lvtcorr_bkg = bkg_counts / bkg_livefrac
    if elut_cor_fac is not None:
        counts_lvtcorr_bkg = counts_lvtcorr_bkg * elut_cor_fac

    counts = counts_lvtcorr - t_norm * (counts_lvtcorr_bkg / t_norm_bkg)
    bkg_err = (bkg_err / t_norm_bkg) * t_norm
    if legacy:
        counts_err, bkg_err = np.sqrt(counts_err**2 + bkg_err**2), None

    if is_spectrogram:
        return replace(
            data,
            counts=counts * livefrac,
            counts_err=counts_err * livefrac,
            livefrac_err=None if livefrac_err is None else livefrac_err * livefrac,
            bkg_err=None if bkg_err is None else bkg_err * livefrac,
        )

    # scale each group of detectors back by its mean livetime fraction, as `_apply_livetime` does without a
    # background; detectors in no group keep their own
    eff_livefrac = np.array(livefrac, dtype=float)
    for group in detector_groups:
        eff_livefrac[:, group] = np.nanmean(livefrac[:, group, :, :], axis=DETECTOR, keepdims=True)

    # the background is only subtracted where science data are selected, so the other detectors and pixels stay
    # zero, as they are without a background
    selected = np.zeros((1, 32, 12, 1), dtype=bool)
    selected[0, detectors[:, None], pixels, 0] = True
    counts = np.where(selected, counts * eff_livefrac, 0 * counts.unit)
    detectors_selected = selected.any(axis=PIXEL, keepdims=True)

    def scaled(err):
        return None if err is None else np.where(detectors_selected, err * eff_livefrac, 0 * err.unit)

    subtracted = replace(
        data,
        counts=counts,
        counts_err=scaled(counts_err),
        livefrac=eff_livefrac,
        livefrac_err=scaled(livefrac_err),
        bkg_err=scaled(bkg_err),
        pixels=data.pixels[np.unique(pixels)].any(axis=0, keepdims=True),
    )
    if np.ndim(pixel_indices) == 2:
        subtracted = subtracted.combine(PIXEL, pixel_indices)  # sum range by range, as the pixel selection does
    return replace(
        subtracted,
        counts=np.nansum(subtracted.counts, axis=PIXEL, keepdims=True),
        pixels=subtracted.pixels.any(axis=0, keepdims=True),
    )


def _pixels(values, pixels):
    """`values` at `pixels` on the pixel axis, unless that axis has length 1."""
    return values if values.shape[PIXEL] == 1 else values[:, :, pixels, :]


def _clip_negative(data, axes):
    """
    Set the counts to zero wherever their total over `axes` is negative.

    Parameters
    ----------
    data : _CountData
        Background subtracted data.
    axes : tuple of int
        Axes pooled into one spectrum: ``(DETECTOR, PIXEL)`` or ``(PIXEL,)``.

    Returns
    -------
    _CountData
    """
    # STIX-GSW's stx_convert_pixel_data does not clip; kept for now, to be revisited with the other error fixes
    total = np.nansum(data.counts, axis=axes, keepdims=True)
    return replace(data, counts=np.where(total < 0, 0, data.counts.value) * data.counts.unit)


def _bkg_indices_check(product, bkg):
    """
    Find the detectors and pixels present in both the science and background
    products.

    Parameters
    ----------
    product : ScienceData
        Science product.
    bkg : ScienceData
        Background product.

    Returns
    -------
    pixel_indices : list of int
        Pixels in both products, in the order they appear in `product`.
    detector_indices : list of int
        Detectors in both products, in the order they appear in `product`.
    """

    pixel_indices_full = _on_indices(product.pixel_masks.masks)
    pixel_indices_full_bkg = _on_indices(bkg.pixel_masks.masks)
    pixel_indices = [d for i, d in enumerate(pixel_indices_full) if d in pixel_indices_full_bkg]

    detector_indices_full = _on_indices(product.detector_masks.masks)
    detector_indices_full_bkg = _on_indices(bkg.detector_masks.masks)
    detector_indices = [d for i, d in enumerate(detector_indices_full) if d in detector_indices_full_bkg]

    return pixel_indices, detector_indices
