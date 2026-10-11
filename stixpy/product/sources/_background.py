"""
Background subtraction for `~stixpy.product.sources.science.ScienceData.get_data`.
"""

import numpy as np

import astropy.units as u

from stixpy.product.sources._corrections import _indices_expand_ranges, _livetime_uncertainty
from stixpy.product.sources._selection import _on_indices

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
        detectors = np.intersect1d(
            np.asarray(_indices_expand_ranges(detector_indices, nest=False))
            if np.ndim(detector_indices) == 2
            else np.ravel(detector_indices),
            _on_indices(product.detector_masks.masks),
        )
        pixels = np.intersect1d(
            np.asarray(_indices_expand_ranges(pixel_indices, nest=False))
            if np.ndim(pixel_indices) == 2
            else np.ravel(pixel_indices),
            _on_indices(product.pixel_masks.masks),
        )
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


def _bkg_sub(
    product,
    bkg,
    counts,
    counts_err,
    counts_bkg,
    counts_err_bkg,
    detector_indices,
    pixel_indices,
    detector_indices_bkg,
    pixel_indices_bkg,
    energy_indices_bkg,
    livefrac,
    livefrac_error,
    livefrac_bkg,
    livefrac_error_bkg,
    elut_cor_fac,
    rcr,
):
    """
    Subtract a background product from the science product.

    The background is first cut down to the detectors, pixels and energy bins
    it shares with the science product. Both products are livetime corrected,
    and ELUT corrected if a factor is given. The background is converted to a
    count rate using its mean integration time, scaled to the duration of each
    science time bin, and subtracted. Uncertainties are combined in quadrature,
    with the livetime uncertainty folded in by `_livetime_uncertainty`.

    The result is then scaled back from livetime-corrected counts:

    - spectrogram products: by the science livetime fraction;
    - pixel data: by the mean livetime fraction of the detectors combined into
      each output spectrum (all selected detectors for flat indices, each range
      for [start, end] pairs), as `_apply_livetime` does without a background.
      Detectors and pixels that are not selected are set to zero.

    The 0 keV bottom bin and the open top bin are removed from the counts,
    uncertainties, `e_norm`, `livefrac_error` and ELUT factor, but not from the
    energy table, so that `_data_select` can trim the table and shift the energy
    indices in one place.

    Parameters
    ----------
    product : ScienceData
        Science product.
    bkg : ScienceData
        Background product.
    counts : astropy.units.Quantity
        The science counts, shape (time, detector, pixel, energy), with pixel
        data on the full detector and pixel grid (see ``_full_layout``) and
        spectrogram counts reshaped to (time, 1, 1, energy).
    counts_err : astropy.units.Quantity
        The compression error of `counts`, same shape.
    counts_bkg : astropy.units.Quantity
        The background counts on the full detector and pixel grid.
    counts_err_bkg : astropy.units.Quantity
        The compression error of `counts_bkg`, same shape.
    detector_indices : numpy.ndarray
        Science detector selection, flat or [start, end] pairs.
    pixel_indices : numpy.ndarray
        Science pixel selection, flat or [start, end] pairs.
    detector_indices_bkg : list of int
        Background detectors shared with the science product, from
        `_bkg_indices_check`.
    pixel_indices_bkg : list of int
        Background pixels shared with the science product, from
        `_bkg_indices_check`.
    energy_indices_bkg : numpy.ndarray
        Background energy bins matching the science energy bins, from
        `_energies_bkg_sub`.
    livefrac : numpy.ndarray
        Science livetime fraction from `_livefrac`.
    livefrac_error : astropy.units.Quantity
        Science livetime uncertainty from `_livefrac`.
    livefrac_bkg : numpy.ndarray
        Background livetime fraction from `_livefrac`.
    livefrac_error_bkg : astropy.units.Quantity
        Background livetime uncertainty from `_livefrac`.
    elut_cor_fac : numpy.ndarray or None
        ELUT correction factor, applied to both products. None for no ELUT
        correction.
    rcr : array_like
        RCR state of each time bin, passed through unchanged.

    Returns
    -------
    tuple
        ``(counts, counts_var, t_norm, e_norm, livefrac, livefrac_error,
        elut_cor_fac, times, energies, rcr)``, in the form `_data_select`
        accepts with ``bkg=True``. `counts_var` holds the 1-sigma uncertainty
        with the pixel axis already summed, and `energies` is the full,
        untrimmed table.

    Notes
    -----
    The background must already have been checked against the science product
    with `_bkg_check`.
    """

    e_norm = product.dE
    is_spectrogram = product.data["counts"].ndim < 4

    counts_var = (counts_err.value**2) * u.ct
    counts_var_bkg = (counts_err_bkg.value**2) * u.ct

    counts_var_bkg = np.sqrt(counts_bkg + counts_var_bkg)

    # Both products are on the full 32 x 12 grid (_full_layout), so detector and pixel
    # numbers index the background directly: keep the full grid, so its shape matches the
    # science counts, and only select the matching energy bins.
    counts_bkg = counts_bkg[:, :, :, energy_indices_bkg]
    counts_var_bkg = counts_var_bkg[:, :, :, energy_indices_bkg]

    if elut_cor_fac is None:
        livefrac_error_bkg = livefrac_error_bkg[:, :, :, energy_indices_bkg]

    if not is_spectrogram:
        pix = np.asarray(pixel_indices)
        if pix.ndim == 2:
            pix = np.asarray(_indices_expand_ranges(pix, nest=False))
        pix = np.asarray(pix, dtype=int)

        detectors_on = _on_indices(product.detector_masks.masks)
        dets = np.asarray(detector_indices)
        if dets.ndim == 2:
            # [start, end] ranges may span detectors that are off, which add nothing; each range shares one livetime
            detector_groups = [np.intersect1d(np.arange(dl, dh + 1), detectors_on) for dl, dh in dets]
            dets = np.concatenate(detector_groups)
        else:
            dets = dets.ravel()
            detector_groups = [dets]
        if np.asarray(pixel_indices).ndim == 2:
            pix = np.intersect1d(pix, _on_indices(product.pixel_masks.masks))

        # the background is only subtracted where science data are selected, so the other detectors and pixels
        # stay zero, as they are without a background
        selected = np.zeros((1, 32, 12, 1), dtype=bool)
        selected[0, dets[:, None], pix, 0] = True

        counts_var_bkg = counts_var_bkg[:, :, pix, :]
        if livefrac_error_bkg.shape[2] != 1:
            livefrac_error_bkg = livefrac_error_bkg[:, :, pix, :]
    else:
        # spectrogram: the counts were summed onboard over the detectors and pixels in its
        # masks, so sum the background over the same ones (those also on in the background)
        counts_bkg = counts_bkg[:, detector_indices_bkg][:, :, pixel_indices_bkg]
        counts_var_bkg = counts_var_bkg[:, detector_indices_bkg][:, :, pixel_indices_bkg]
        livefrac_error_bkg = livefrac_error_bkg[:, detector_indices_bkg][:, :, pixel_indices_bkg]
        livefrac_bkg = livefrac_bkg[:, detector_indices_bkg]

    if elut_cor_fac is not None:
        counts_var_bkg = counts_var_bkg * elut_cor_fac

    counts_var_bkg = _livetime_uncertainty(counts_var_bkg, livefrac_error_bkg, livefrac_bkg)

    if is_spectrogram:
        livefrac = np.nanmean(livefrac, axis=1, keepdims=True)
        livefrac_error = np.nanmean(livefrac_error, axis=(1, 2), keepdims=True)

        counts_bkg = np.nansum(counts_bkg, axis=(1, 2), keepdims=True)
        counts_var_bkg = np.sqrt(np.nansum(counts_var_bkg**2, axis=(1, 2), keepdims=True))

        livefrac_bkg = np.nanmean(livefrac_bkg, axis=1, keepdims=True)
        livefrac_error_bkg = np.sqrt(np.nansum(livefrac_error_bkg**2, axis=(1, 2), keepdims=True))

    counts_var = np.sqrt(counts + counts_var)

    t_norm = product.data["timedel"]
    times = product.times
    energies = product.energies

    if not is_spectrogram:
        counts_var = counts_var[:, :, pix, :]

        if livefrac_error.shape[2] != 1:
            livefrac_error = livefrac_error[:, :, pix, :]

    if elut_cor_fac is not None:
        counts_var = counts_var * elut_cor_fac

    counts_var = _livetime_uncertainty(counts_var, livefrac_error, livefrac)

    t_norm_bkg = bkg.data["timedel"]
    t_norm = t_norm.to(u.s)
    t_norm_bkg = t_norm_bkg.to(u.s)

    if elut_cor_fac is not None:
        counts_lvtcorr = (counts * elut_cor_fac) / livefrac
    else:
        counts_lvtcorr = (counts) / livefrac

    if elut_cor_fac is not None:
        counts_lvtcorr_bkg = (counts_bkg / livefrac_bkg) * elut_cor_fac
    else:
        counts_lvtcorr_bkg = counts_bkg / livefrac_bkg

    count_rate_lvtcorr_bkg = counts_lvtcorr_bkg / t_norm_bkg.mean()
    count_lvtcorr_scaled_bkg = t_norm.reshape(len(t_norm), 1, 1, 1) * count_rate_lvtcorr_bkg

    counts_var_lvtcorr = counts_var
    counts_var_lvtcorr_bkg = counts_var_bkg

    counts_var_lvtcorr_scaled_bkg = (counts_var_lvtcorr_bkg / t_norm_bkg.mean()) * t_norm.reshape(len(t_norm), 1, 1, 1)

    spec_in_corr = counts_lvtcorr - count_lvtcorr_scaled_bkg

    spec_in_err = np.sqrt((counts_var_lvtcorr**2) + (counts_var_lvtcorr_scaled_bkg**2))

    if energies["e_low"][0].value == 0:
        spec_in_corr = spec_in_corr[..., 1:]
        spec_in_err = spec_in_err[..., 1:]
        e_norm = e_norm[1:]
        livefrac_error = livefrac_error[..., 1:]
        if elut_cor_fac is not None:
            elut_cor_fac = elut_cor_fac[..., 1:]

    if np.isnan(energies["e_high"][-1].value):
        spec_in_corr = spec_in_corr[..., :-1]
        spec_in_err = spec_in_err[..., :-1]
        e_norm = e_norm[:-1]
        livefrac_error = livefrac_error[..., :-1]
        if elut_cor_fac is not None:
            elut_cor_fac = elut_cor_fac[..., :-1]

    if is_spectrogram:
        spec_in_final = spec_in_corr * livefrac
        spec_in_err_final = spec_in_err * livefrac

        counts = spec_in_final
        counts_var = spec_in_err_final

    else:
        # scale each group of detectors back by its mean livetime fraction, as `_apply_livetime` does without a
        # background
        eff_livefrac = np.array(livefrac, dtype=float)  # detectors in no group keep their own
        for group in detector_groups:
            eff_livefrac[:, group] = np.nanmean(livefrac[:, group, :, :], axis=1, keepdims=True)

        counts = np.where(selected, spec_in_corr * eff_livefrac, 0 * spec_in_corr.unit)
        counts_var = np.where(selected.any(axis=2, keepdims=True), spec_in_err * eff_livefrac, 0 * spec_in_err.unit)

        livefrac = np.broadcast_to(eff_livefrac, counts.shape)

    return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr


def _energies_bkg_sub(product, bkg):
    """
    Find the background energy bins that match the science energy bins.

    Bins are matched on their lower edge, `e_low`.

    Parameters
    ----------
    product : ScienceData
        Science product.
    bkg : ScienceData
        Background product.

    Returns
    -------
    numpy.ndarray
        Indices into `bkg.energies` of the bins whose lower edge also appears
        in `product.energies`, in increasing energy order.
    """
    _, _, indices_sub = np.intersect1d(product.energies["e_low"], bkg.energies["e_low"], return_indices=True)

    return indices_sub


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
