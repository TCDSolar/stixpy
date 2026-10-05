"""
Livetime and energy look-up table (ELUT) corrections, and the data selection that applies them, for
`~stixpy.product.sources.science.ScienceData.get_data`.
"""

import numpy as np

import astropy.units as u
from astropy.table import QTable
from astropy.time import Time

from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.config.instrument import STIX_INSTRUMENT
from stixpy.product.sources._rcr import _handle_nested_pairs, _rcr_shift, _rcr_signal, _rcr_warning
from stixpy.product.sources._selection import _on_indices

__all__ = []


def _indices_expand_ranges(pairs, nest=True):
    """
    Expand [start, end] pairs into the indices they cover, inclusive.

    Parameters
    ----------
    pairs : array_like
        [start, end] pairs, e.g. ``[[1, 3], [7, 8]]``.
    nest : bool, optional
        If True (default), return one array per pair:
        ``[array([1, 2, 3]), array([7, 8])]``. If False, return one flat list:
        ``[1, 2, 3, 7, 8]``.

    Returns
    -------
    list
        A list of arrays if `nest` is True, otherwise a flat list of indices.
    """
    result = []
    for pair in pairs:
        if nest:
            result.append(np.arange(pair[0], pair[1] + 1, 1))
        else:
            result.extend(np.arange(pair[0], pair[1] + 1, 1))
    return result


def _livetime_uncertainty(counts_var, livefrac_error, livefrac):
    """
    Combine the count uncertainty with the livetime uncertainty.

    If `livefrac_error` is None, `counts_var` is returned unchanged. Otherwise
    both are summed in quadrature over the pixel axis (axis 2), and then
    combined as

        sqrt((counts_var / livefrac)**2 + livefrac_error**2)

    Parameters
    ----------
    counts_var : astropy.units.Quantity
        Count uncertainty (1-sigma, not variance), shape
        (time, detector, pixel, energy).
    livefrac_error : astropy.units.Quantity or None
        Livetime uncertainty in counts, as returned by `_livefrac`, with the
        same shape as `counts_var`.
    livefrac : numpy.ndarray
        Livetime fraction, broadcastable against `counts_var`.

    Returns
    -------
    astropy.units.Quantity
        The combined uncertainty in counts, with the pixel axis reduced to
        length 1, or `counts_var` unchanged if `livefrac_error` is None.
    """

    if livefrac_error is not None:
        counts_var = np.sqrt(np.nansum(counts_var**2, axis=2, keepdims=True))
        livefrac_error = np.sqrt(np.nansum(livefrac_error**2, axis=2, keepdims=True))

        counts_var_lvtcorr = np.sqrt(((counts_var / livefrac) ** 2).value + livefrac_error.value**2)

        return counts_var_lvtcorr * u.ct

    else:
        return counts_var


def _apply_livetime(counts, counts_var, livefrac, groups):
    """
    Apply the livetime correction so that each group of detectors shares one
    effective livetime.

    Within each group, every detector's counts are divided by its own livetime
    fraction and multiplied by the group's mean livetime fraction for that time
    bin. The uncertainties are multiplied by the same group mean, and the
    livetime fraction of every detector in the group is replaced by it.
    Detectors that are in no group are left unchanged.

    Parameters
    ----------
    counts : astropy.units.Quantity or numpy.ndarray
        Counts, shape (time, detector, pixel, energy).
    counts_var : astropy.units.Quantity or numpy.ndarray
        Count uncertainty, with the same detector axis as `counts`.
    livefrac : numpy.ndarray
        Livetime fraction, shape (time, detector, 1, 1).
    groups : list of array_like
        Detector indices of each group, e.g. ``[[0, 1], [2, 3]]``.

    Returns
    -------
    counts : astropy.units.Quantity or numpy.ndarray
        Livetime-corrected counts.
    counts_var : astropy.units.Quantity or numpy.ndarray
        Uncertainties scaled by each group's mean livetime fraction.
    livefrac : numpy.ndarray
        Livetime fraction, with each group set to its mean.
    """
    counts_corr = counts / livefrac
    counts_out = counts.astype(float).copy()
    counts_var_out = counts_var.astype(float).copy()
    new_livefrac = livefrac.astype(float).copy()
    for g in groups:
        g = np.atleast_1d(np.asarray(g))
        eff_lt = np.nanmean(livefrac[:, g, :, :], axis=1, keepdims=True)  # scalar per time bin
        counts_out[:, g, :, :] = counts_corr[:, g, :, :] * eff_lt
        counts_var_out[:, g, :, :] = counts_var[:, g, :, :] * eff_lt
        new_livefrac[:, g, :, :] = np.broadcast_to(eff_lt, new_livefrac[:, g, :, :].shape)
    return counts_out, counts_var_out, new_livefrac


def _livefrac(product, counts, elut_cor_fac, pixel_indices):
    """
    Compute the livetime fraction and its uncertainty from the trigger counts.

    For pixel data the triggers are mapped onto detectors with
    `STIX_INSTRUMENT.subcol_adc_mapping`, giving one livetime fraction per
    detector and time bin. For spectrogram products the trigger total is
    divided by 16, giving one value per time bin.

    The trigger uncertainty is ``sqrt(triggers_comp_err**2 + triggers)``. The
    livetime fraction is also evaluated at the triggers minus and plus that
    uncertainty (floored, to match IDL), and the livetime uncertainty is half
    the resulting spread in livetime-corrected counts.

    For pixel data that uncertainty is computed on the counts summed over the
    selected pixels, then shared out over those pixels in proportion to the
    square root of each pixel's share of the counts, so that a quadrature sum
    over pixels returns the total. Pixels that are not selected get zero.

    Parameters
    ----------
    product : ScienceData
        Product to compute the livetime for.
    counts : astropy.units.Quantity
        The product's counts, shape (time, detector, pixel, energy), with pixel
        data on the full detector and pixel grid (see ``_full_layout``) and
        spectrogram counts reshaped to (time, 1, 1, energy).
    elut_cor_fac : numpy.ndarray or None
        ELUT correction factor applied to the counts before the uncertainty is
        computed. None for no ELUT correction.
    pixel_indices : numpy.ndarray or None
        Pixels to compute the uncertainty over, flat or [start, end] pairs, or
        None for all pixels. Ignored for spectrogram products.

    Returns
    -------
    livefrac : numpy.ndarray
        Livetime fraction, shape (time, detector, 1, 1) for pixel data or
        (time, 1, 1, 1) for spectrogram products.
    livefrac_error : astropy.units.Quantity
        Livetime uncertainty in counts, with the same shape as `counts`.
    """

    trigger_to_detector = STIX_INSTRUMENT.subcol_adc_mapping
    shape = product.data["counts"].shape

    if len(shape) < 4:
        trig_raw = product.data["triggers"] / 16
        trig_err = np.sqrt(product.data["triggers_comp_err"] ** 2 + product.data["triggers"]) / 16

        triggers = np.floor(trig_raw)
        triggers_lower = np.floor(np.maximum(trig_raw - trig_err, 0))
        triggers_upper = np.floor(trig_raw + trig_err)

        timedel = product.data["timedel"].to("s")

        livefrac, _, _ = get_livetime_fraction(triggers / timedel)
        livefrac_lower, _, _ = get_livetime_fraction(triggers_lower / timedel)
        livefrac_upper, _, _ = get_livetime_fraction(triggers_upper / timedel)

        livefrac = livefrac.reshape(livefrac.shape + (1, 1, 1))
        livefrac_lower = livefrac_lower.reshape(livefrac_lower.shape + (1, 1, 1))
        livefrac_upper = livefrac_upper.reshape(livefrac_upper.shape + (1, 1, 1))

    else:
        triggers = product.data["triggers"][:, trigger_to_detector].astype(float)[...]

        triggers_error = product.data["triggers_comp_err"][:, trigger_to_detector].astype(float)[...]

        triggers_error = np.sqrt(triggers_error**2 + triggers)

        triggers_lower = np.floor(
            np.maximum(triggers - triggers_error, 0)
        )  # This brings in line with IDL precision, if removed then the ratio at livefrac of ~0.5 goes to 0.0002, rather than ~1e-7.
        triggers_upper = np.floor(
            triggers + triggers_error
        )  # This brings in line with IDL precision, if removed then the ratio at livefrac of ~0.5 goes to 0.0002, rather than ~1e-7.

        livefrac, _, _ = get_livetime_fraction(triggers / product.data["timedel"].to("s").reshape(-1, 1))
        livefrac_lower, _, _ = get_livetime_fraction(triggers_lower / product.data["timedel"].to("s").reshape(-1, 1))
        livefrac_upper, _, _ = get_livetime_fraction(triggers_upper / product.data["timedel"].to("s").reshape(-1, 1))

        livefrac = livefrac.reshape(livefrac.shape + (1, 1))
        livefrac_lower = livefrac_lower.reshape(livefrac_lower.shape + (1, 1))
        livefrac_upper = livefrac_upper.reshape(livefrac_upper.shape + (1, 1))

    if elut_cor_fac is not None:
        counts = counts * elut_cor_fac

    # ---- resolve the pixel summation group ------------------------------

    if len(shape) < 4:
        livefrac_error = ((counts / livefrac_lower) - (counts / livefrac_upper)) / 2

    else:
        n_pix = counts.shape[2]

        if pixel_indices is None:
            pix = np.arange(n_pix)
        else:
            pix = np.asarray(pixel_indices)
            if pix.ndim == 2:
                pix = np.asarray(_indices_expand_ranges(pix, nest=False))
            else:
                pix = pix.ravel()

        pix = np.asarray(pix, dtype=int)

        # ---- distribute the correlated livetime error over pixels -----------
        # The livetime error is a systematic shared by every pixel of a detector, so
        # its true total is the linear term evaluated on the pixel-summed counts.
        # Spreading it as e_tot * sqrt(w_p), with w_p = C_p / C_tot summing to 1,
        # means a downstream quadrature sum over the pixel axis returns e_tot exactly.
        c_g = counts[:, :, pix, :]
        c_tot = np.nansum(c_g, axis=2, keepdims=True)
        e_tot = (c_tot / livefrac_lower - c_tot / livefrac_upper) / 2

        c_fin = np.where(np.isfinite(c_g), np.abs(c_g), 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = c_fin / np.nansum(c_fin, axis=2, keepdims=True)
        w = np.where(np.isfinite(w), w, 1 / pix.size)

        livefrac_error = np.zeros(counts.shape) * counts.unit
        livefrac_error[:, :, pix, :] = e_tot * np.sqrt(w)

    return livefrac, livefrac_error


def _normalize_elut_by_group_detector_mean(bins, bins_actual, index_groups):
    """
    Compute the ELUT correction factor with `bins_actual` averaged within each
    detector group.

    Within each group, `bins_actual` is replaced by its mean over the group's
    detectors, so every detector in the group gets the same denominator.
    Detectors in no group keep their own value.

    Parameters
    ----------
    bins : numpy.ndarray
        From `get_elut_correction`, 4-D with detectors on axis 1.
    bins_actual : numpy.ndarray
        From `get_elut_correction`, same shape as `bins`.
    index_groups : list of array_like
        Detector indices of each group, e.g. ``[[0, 1, 2], [5, 6]]``.

    Returns
    -------
    numpy.ndarray
        ``bins / bins_actual``, with `bins_actual` averaged within each group.
    """
    bins = np.asarray(bins)
    bins_actual = np.asarray(bins_actual)

    bins_actual_mod = bins_actual.copy()

    for group in index_groups:
        idx = np.array(group)

        # mean over just the group's indices along `axis`, keepdims for broadcasting
        group_mean = bins_actual[:, idx, :, :].mean(axis=1, keepdims=True)

        # shape to broadcast the mean into (same as the group's slice shape)
        broadcast_shape = list(bins_actual.shape)
        broadcast_shape[1] = len(idx)

        bins_actual_mod[:, idx, :, :] = np.broadcast_to(group_mean, tuple(broadcast_shape))

    return bins / bins_actual_mod


def _elut_correction_sort(bins, bins_actual, detector_sum, pixel_indices, detector_indices, spec_file):
    """
    Compute the ELUT correction factor for the selected detectors and pixels.

    `bins` and `bins_actual` are averaged over the selected pixels, and the
    factor is ``bins / bins_actual``. Detectors are handled as follows:

    - spectrogram products (`spec_file` True): averaged over the detectors
      given, giving one factor;
    - `detector_sum` True: averaged over the selected detectors, giving one
      factor;
    - flat detector indices, not summed: one factor per detector, for every
      detector rather than only the selected ones;
    - [start, end] detector ranges, not summed: `bins_actual` averaged within
      each range (see `_normalize_elut_by_group_detector_mean`).

    Parameters
    ----------
    bins : numpy.ndarray
        From `get_elut_correction`, 4-D with detectors on axis 1 and pixels on
        axis 2.
    bins_actual : numpy.ndarray
        From `get_elut_correction`, same shape as `bins`.
    detector_sum : bool
        Whether detectors are summed into one spectrum.
    pixel_indices : numpy.ndarray
        Selected pixels, flat or [start, end] pairs.
    detector_indices : numpy.ndarray
        Selected detectors, flat or [start, end] pairs.
    spec_file : bool
        True for spectrogram products.

    Returns
    -------
    numpy.ndarray
        The ELUT correction factor, broadcastable against the counts.
    """

    pixel_indices = np.asarray(pixel_indices)
    detector_indices = np.asarray(detector_indices)

    if spec_file:
        bins = np.nanmean(bins[:, :, pixel_indices, :], axis=2, keepdims=True)
        bins_actual = np.nanmean(bins_actual[:, :, pixel_indices, :], axis=2, keepdims=True)

        bins = np.nanmean(bins[:, detector_indices, :, :], axis=1, keepdims=True)
        bins_actual = np.nanmean(bins_actual[:, detector_indices, :, :], axis=1, keepdims=True)

        elut_cor_fac = bins / bins_actual

    else:
        if pixel_indices.ndim == 2:
            pixel_indices = _indices_expand_ranges(pixel_indices, nest=False)

        bins = np.nanmean(bins[:, :, pixel_indices, :], axis=2, keepdims=True)
        bins_actual = np.nanmean(bins_actual[:, :, pixel_indices, :], axis=2, keepdims=True)

        if detector_sum:
            if detector_indices.ndim == 2:
                detector_indices = _indices_expand_ranges(detector_indices, nest=False)

            bins = np.nanmean(bins[:, detector_indices, :, :], axis=1, keepdims=True)

            bins_actual = np.nanmean(bins_actual[:, detector_indices, :, :], axis=1, keepdims=True)

            elut_cor_fac = bins / bins_actual

        elif detector_indices.ndim == 1:
            elut_cor_fac = bins / bins_actual

        elif detector_indices.ndim == 2:
            detector_indices = _indices_expand_ranges(detector_indices, nest=True)

            elut_cor_fac = _normalize_elut_by_group_detector_mean(bins, bins_actual, detector_indices)

    return elut_cor_fac


def _data_select(
    product,
    counts,
    counts_err,
    detector_indices,
    pixel_indices,
    energy_indices,
    time_indices,
    livefrac,
    livefrac_error,
    elut_cor_fac,
    sum_all_times,
    n_detectors=None,
):
    """
    Apply the requested detector, pixel, energy and time selection to the data.

    The count uncertainty starts as ``sqrt(counts + compression_error**2)``.

    The 0 keV bottom energy bin and the open top bin (upper edge NaN) are
    removed first. Energy indices number the rows of the product's energy
    table, so they are shifted down by one when the bottom bin is removed and
    then clipped into the remaining range.

    On every axis, flat indices keep those bins and [start, end] pairs sum each
    inclusive range into one bin (counts summed, uncertainties in quadrature,
    livetime values averaged). The steps run in this order:

    1. ELUT correction.
    2. Pixel selection.
    3. Energy selection; for ranges, `e_norm` and the energy table are rebuilt.
    4. Livetime correction by detector group (see `_apply_livetime`), then a sum
       over pixels. Only when a livetime fraction is available. With no
       detector selection all detectors that are on form one group, a flat
       selection forms one group, and each [start, end] range is its own group.
    5. Detector selection.
    6. RCR alignment (see `_rcr_shift`).
    7. Time selection, and optionally a sum over all time bins.

    Parameters
    ----------
    product : ScienceData
        The science product.
    counts : astropy.units.Quantity
        The product's counts, shape (time, detector, pixel, energy), with pixel
        data on the full detector and pixel grid (see ``_full_layout``) and
        spectrogram counts reshaped to (time, 1, 1, energy).
    counts_err : astropy.units.Quantity
        The compression error of `counts`, same shape.
    detector_indices : numpy.ndarray or None
        Flat detector indices or [start, end] pairs. Detector labels such as
        "top24" must already have been resolved by `_indices_check`. Ignored
        for spectrogram products.
    pixel_indices : numpy.ndarray or None
        Flat pixel indices or [start, end] pairs. Ignored for spectrogram
        products.
    energy_indices : list, numpy.ndarray or None
        Flat energy indices or [start, end] pairs, numbering the rows of the
        product's energy table.
    time_indices : list, numpy.ndarray or None
        Flat time indices or [start, end] pairs, as returned by
        `_time_indices_format`.
    livefrac : numpy.ndarray or None
        Livetime fraction from `_livefrac`.
    livefrac_error : astropy.units.Quantity or None
        Livetime uncertainty from `_livefrac`.
    elut_cor_fac : numpy.ndarray or None
        ELUT correction factor from `_elut_correction_sort`.
    sum_all_times : bool
        If True, the resulting time bins are summed into one.
    n_detectors : int, optional
        Number of detectors summed on board, for spectrogram products. Used to
        scale the RCR jump signal.

    Returns
    -------
    tuple
        ``(counts, counts_var, t_norm, e_norm, livefrac, livefrac_error,
        elut_cor_fac, times, energies, rcr)`` after the selection. `counts_var`
        holds the 1-sigma uncertainty, not the variance.
    """
    rcr = np.asarray(product.data["rcr"])
    e_norm = product.dE

    if product.data["counts"].ndim < 4:
        detector_indices = None
        pixel_indices = None

    counts_var = np.sqrt(counts.value + counts_err.value**2) * u.ct

    t_norm = product.data["timedel"]
    times = product.times
    energies = product.energies

    if energies["e_low"][0].value == 0:
        counts = counts[..., 1:]
        counts_var = counts_var[..., 1:]
        energies = energies[1:]
        e_norm = e_norm[1:]
        if energy_indices is not None:
            energy_indices = np.asarray(energy_indices) - 1
        if elut_cor_fac is not None:
            elut_cor_fac = elut_cor_fac[..., 1:]
        if livefrac is not None:
            livefrac_error = livefrac_error[..., 1:]

    if np.isnan(energies["e_high"][-1].value):
        counts = counts[..., :-1]
        counts_var = counts_var[..., :-1]
        energies = energies[:-1]
        e_norm = e_norm[:-1]
        if elut_cor_fac is not None:
            elut_cor_fac = elut_cor_fac[..., :-1]
        if livefrac is not None:
            livefrac_error = livefrac_error[..., :-1]

    if elut_cor_fac is not None:
        counts = counts * elut_cor_fac
        counts_var = counts_var * elut_cor_fac

    if energy_indices is not None:
        energy_indices = np.clip(energy_indices, 0, len(energies) - 1)

    if pixel_indices is not None:
        pixel_indices = np.asarray(pixel_indices)
        if pixel_indices.ndim == 1:
            pixel_mask = np.full(12, False)
            pixel_mask[pixel_indices] = True
            num_pixels = counts.shape[2]
            counts = counts[..., pixel_mask[:num_pixels], :]
            counts_var = counts_var[..., pixel_mask[:num_pixels], :]
            if livefrac is not None and livefrac.shape[2] != 1:
                livefrac = livefrac[:, :, pixel_mask[:num_pixels], :]
            if livefrac_error is not None and livefrac_error.shape[2] != 1:
                livefrac_error = livefrac_error[:, :, pixel_mask[:num_pixels], :]

        if pixel_indices.ndim == 2:
            counts = np.concatenate(
                [np.sum(counts[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices], axis=2
            )
            counts_var = np.concatenate(
                [
                    np.sqrt(np.sum(counts_var[..., pl : ph + 1, :] ** 2, axis=2, keepdims=True))
                    for pl, ph in pixel_indices
                ],
                axis=2,
            )

            if livefrac is not None and livefrac.shape[2] != 1:
                livefrac = np.concatenate(
                    [np.mean(livefrac[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices],
                    axis=2,
                )

            if livefrac_error is not None:
                livefrac_error = np.concatenate(
                    [
                        np.sqrt(np.sum(livefrac_error[..., pl : ph + 1, :] ** 2, axis=2, keepdims=True))
                        for pl, ph in pixel_indices
                    ],
                    axis=2,
                )

    counts_rcr = np.nansum(counts, axis=-1, keepdims=True)

    if energy_indices is not None:
        energy_indices = np.asarray(energy_indices)
        if energy_indices.ndim == 1:
            energy_mask = np.full(counts.shape[-1], False)
            energy_mask[energy_indices] = True
            counts = counts[..., energy_mask]
            counts_var = counts_var[..., energy_mask]
            e_norm = e_norm[energy_mask]
            energies = energies[energy_mask]

            if elut_cor_fac is not None:
                elut_cor_fac = elut_cor_fac[..., energy_mask]

            if livefrac_error is not None:
                livefrac_error = livefrac_error[..., energy_mask]

        if energy_indices.ndim == 2:
            counts = np.concatenate(
                [np.sum(counts[..., el : eh + 1], axis=-1, keepdims=True) for el, eh in energy_indices], axis=-1
            )

            counts_var = np.concatenate(
                [
                    np.sqrt(np.sum(counts_var[..., el : eh + 1] ** 2, axis=-1, keepdims=True))
                    for el, eh in energy_indices
                ],
                axis=-1,
            )

            e_norm = np.hstack([(energies["e_high"][eh] - energies["e_low"][el]) for el, eh in energy_indices])

            if elut_cor_fac is not None:
                elut_cor_fac = np.concatenate(
                    [np.mean(elut_cor_fac[..., el : eh + 1], axis=-1, keepdims=True) for el, eh in energy_indices],
                    axis=-1,
                )

            if livefrac_error is not None:
                # scales with the counts and shares one livetime across all energies,
                # so the bins in a band are fully correlated: add them linearly
                livefrac_error = np.concatenate(
                    [np.sum(livefrac_error[..., el : eh + 1], axis=-1, keepdims=True) for el, eh in energy_indices],
                    axis=-1,
                )

            energies = np.atleast_2d(
                [(energies["e_low"][el].value, energies["e_high"][eh].value) for el, eh in energy_indices]
            )
            energies = QTable(energies * u.keV, names=["e_low", "e_high"])

    if livefrac is not None and detector_indices is None:
        n_det = counts.shape[1]
        # every detector of a spectrogram, or every detector that is on for pixel data on the full grid
        groups = [np.arange(n_det)] if n_det == 1 else [_on_indices(product.detector_masks.masks)]

        counts_rcr = _apply_livetime(counts_rcr, counts_rcr, livefrac, groups)[0]
        counts_rcr = np.nansum(counts_rcr, axis=2, keepdims=True)
        counts_var = _livetime_uncertainty(counts_var, livefrac_error, livefrac)
        counts, counts_var, livefrac = _apply_livetime(counts, counts_var, livefrac, groups)
        counts = np.nansum(counts, axis=2, keepdims=True)

    if detector_indices is not None:
        detector_indices = np.asarray(detector_indices)  # "top24" must already be resolved to indices upstream

        # ---- livetime -------------------------------------------------------
        if livefrac is not None:
            if detector_indices.ndim == 1:
                groups = [detector_indices]  # all selected -> one spectrum
            else:  # ndim == 2 : each (dl, dh) range -> one output spectrum, of the detectors that are on
                detectors_on = _on_indices(product.detector_masks.masks)
                groups = [np.intersect1d(np.arange(dl, dh + 1), detectors_on) for dl, dh in detector_indices]

            counts_rcr = _apply_livetime(counts_rcr, counts_rcr, livefrac, groups)[0]
            counts_rcr = np.nansum(counts_rcr, axis=2, keepdims=True)
            counts_var = _livetime_uncertainty(counts_var, livefrac_error, livefrac)
            counts, counts_var, livefrac = _apply_livetime(counts, counts_var, livefrac, groups)
            counts = np.nansum(counts, axis=2, keepdims=True)

        # ---- detector selection / summing -----------------------------------
        selected = (
            np.concatenate([np.arange(dl, dh + 1) for dl, dh in detector_indices])
            if detector_indices.ndim == 2
            else detector_indices
        )
        n_det_rcr = np.count_nonzero(np.nansum(np.abs(counts_rcr[:, selected]), axis=(0, 2, 3)))

        if detector_indices.ndim == 1:
            detector_mask = np.full(32, False)
            detector_mask[detector_indices] = True
            counts = counts[:, detector_mask, ...]
            counts_rcr = counts_rcr[:, detector_mask, ...]
            counts_var = counts_var[:, detector_mask, ...]
            if livefrac is not None:
                livefrac = livefrac[:, detector_mask, :, :]
            if livefrac_error is not None:
                livefrac_error = livefrac_error[:, detector_mask, :, :]

        if detector_indices.ndim == 2:
            counts = np.hstack(
                [np.sum(counts[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices]
            )

            counts_rcr = np.hstack(
                [np.sum(counts_rcr[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices]
            )

            counts_var = np.concatenate(
                [
                    np.sqrt(np.sum(counts_var[:, dl : dh + 1, ...] ** 2, axis=1, keepdims=True))
                    for dl, dh in detector_indices
                ],
                axis=1,
            )
            if livefrac is not None:
                livefrac = np.concatenate(
                    [np.mean(livefrac[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices],
                    axis=1,
                )
            if livefrac_error is not None:
                livefrac_error = np.concatenate(
                    [
                        np.sqrt(np.mean(livefrac_error[:, dl : dh + 1, ...] ** 2, axis=1, keepdims=True))
                        for dl, dh in detector_indices
                    ],
                    axis=1,
                )

    if detector_indices is None:
        # no detector selection: a spectrogram (summed on board over n_detectors)
        # or pixel data with every detector
        if n_detectors is not None:
            n_det_rcr = n_detectors
        else:
            n_det_rcr = np.count_nonzero(np.nansum(np.abs(counts_rcr), axis=(0, 2, 3)))
    rcr = _rcr_shift(rcr, _rcr_signal(counts_rcr, n_det_rcr))
    if time_indices is not None:
        if np.asarray(time_indices).ndim == 1:
            _rcr_warning(list(time_indices), rcr)
        else:
            _handle_nested_pairs(time_indices, rcr)

    if time_indices is not None:
        time_indices = np.asarray(time_indices)
        if time_indices.ndim == 1:
            time_mask = np.full(times.shape, False)
            time_mask[time_indices] = True
            counts = counts[time_mask, ...]
            counts_var = counts_var[time_mask, ...]
            t_norm = t_norm[time_mask]
            rcr = rcr[time_mask]
            if livefrac is not None:
                livefrac = livefrac[time_mask, ...]
            if livefrac_error is not None:
                livefrac_error = livefrac_error[time_mask, ...]
            times = times[time_mask]

        if time_indices.ndim == 2:
            new_times = []
            dt = []
            for tl, th in time_indices:
                ts = times[tl] - t_norm[tl] * 0.5
                te = times[th] + t_norm[th] * 0.5
                td = te - ts
                tc = ts + (td * 0.5)
                dt.append(td.to("s"))
                new_times.append(tc)

            dt = np.hstack(dt)
            times = Time(new_times)

            counts = np.vstack([np.sum(counts[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices])
            rcr = np.vstack([np.mean(rcr[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices])

            if livefrac is not None:
                dur = t_norm.to_value(u.s).reshape((-1,) + (1,) * (livefrac.ndim - 1))
                livefrac = np.vstack(
                    [
                        np.sum(livefrac[tl : th + 1, ...] * dur[tl : th + 1], axis=0, keepdims=True)
                        / np.sum(dur[tl : th + 1])
                        for tl, th in time_indices
                    ]
                )

            if livefrac_error is not None:
                livefrac_error = np.vstack(
                    [
                        np.sqrt(np.sum(livefrac_error[tl : th + 1, ...] ** 2, axis=0, keepdims=True))
                        for tl, th in time_indices
                    ]
                )

            counts_var = np.vstack(
                [np.sqrt(np.sum(counts_var[tl : th + 1, ...] ** 2, axis=0, keepdims=True)) for tl, th in time_indices]
            )
            t_norm = dt

    if sum_all_times and np.shape(counts)[0] > 1:
        rcr_unique = np.unique(rcr)
        if rcr_unique.size != 1:
            raise ValueError(
                "Cannot sum all times as the RCR state changes between the selected time ranges "
                f"(RCR states {rcr_unique.astype(int).tolist()}). "
                "Select time ranges in a single RCR state."
            )

        rcr = rcr_unique
        # one bin from the start of the first selected bin to the end of the last
        start = times[0] - 0.5 * t_norm[0]
        end = times[-1] + 0.5 * t_norm[-1]
        times = Time([start + 0.5 * (end - start)])

        # weight by each bin's duration, so counts / (t_norm * livefrac) divides by the total live time
        if livefrac is not None:
            w = (t_norm / np.sum(t_norm)).to_value(u.one).reshape((-1,) + (1,) * (livefrac.ndim - 1))
            livefrac = np.sum(livefrac * w, axis=0, keepdims=True)
        if livefrac_error is not None:
            livefrac_error = np.sqrt(np.mean(livefrac_error**2, axis=0, keepdims=True))

        counts = np.sum(counts, axis=0, keepdims=True)
        counts_var = np.sqrt(np.sum(counts_var**2, axis=0, keepdims=True))
        t_norm = np.sum(t_norm, keepdims=True)

    return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr
