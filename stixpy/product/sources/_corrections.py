"""
Livetime and energy look-up table (ELUT) corrections for `~stixpy.product.sources.science.ScienceData.get_data`.
"""

from dataclasses import replace

import numpy as np

import astropy.units as u

from stixpy.calibration.elut import get_elut_correction
from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.config.instrument import STIX_INSTRUMENT
from stixpy.product.sources._selection import DETECTOR, PIXEL, _flat_indices, _kept_energy_bins, _on_indices

__all__ = []


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


def _livefrac(product, counts, pixel_indices, per_pixel=False, *, ssw_compatible=False):
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
    With `per_pixel`, it is instead the uncertainty of each pixel's own counts,
    for a linear sum over pixels.

    Parameters
    ----------
    product : ScienceData
        Product to compute the livetime for.
    counts : astropy.units.Quantity
        The product's counts, shape (time, detector, pixel, energy), with pixel
        data on the full detector and pixel grid (see ``_full_layout``) and
        spectrogram counts reshaped to (time, 1, 1, energy).
    pixel_indices : numpy.ndarray or None
        Pixels to compute the uncertainty over, flat or [start, end] pairs, or
        None for all pixels. Ignored for spectrogram products.
    per_pixel : bool, optional
        Give each pixel the uncertainty of its own counts (the "correlated"
        error model) rather than spreading the total (the "legacy" one).

    Returns
    -------
    livefrac : numpy.ndarray
        Livetime fraction, shape (time, detector, 1, 1) for pixel data or
        (time, 1, 1, 1) for spectrogram products.
    livefrac_error : astropy.units.Quantity
        Livetime uncertainty in counts, with the same shape as `counts` after
        any energy selection.
    """

    trigger_to_detector = STIX_INSTRUMENT.subcol_adc_mapping
    shape = product.data["counts"].shape

    if len(shape) < 4:
        trig_raw = product.data["triggers"] / 16
        trig_err = np.sqrt(product.data["triggers_comp_err"] ** 2 + product.data["triggers"]) / 16

        timedel = product.data["timedel"].to(u.s)
        triggers = np.floor(trig_raw) if ssw_compatible else trig_raw
        rate = triggers / timedel
        livefrac, _, _ = get_livetime_fraction(rate)
        lower_triggers = np.maximum(trig_raw - trig_err, 0)
        upper_triggers = trig_raw + trig_err
        if ssw_compatible:
            lower_triggers = np.floor(lower_triggers)
            upper_triggers = np.floor(upper_triggers)
        lower, _, _ = get_livetime_fraction(lower_triggers / timedel)
        upper, _, _ = get_livetime_fraction(upper_triggers / timedel)
        inverse_livetime_error = (1 / upper - 1 / lower) / 2

        livefrac = livefrac.reshape(livefrac.shape + (1, 1, 1))

    else:
        triggers = product.data["triggers"][:, trigger_to_detector].astype(float)[...]

        triggers_error = product.data["triggers_comp_err"][:, trigger_to_detector].astype(float)[...]

        triggers_error = np.sqrt(triggers_error**2 + triggers)

        triggers_lower = np.maximum(triggers - triggers_error, 0)
        triggers_upper = triggers + triggers_error
        if ssw_compatible:
            triggers_lower = np.floor(triggers_lower)
            triggers_upper = np.floor(triggers_upper)

        livefrac, _, _ = get_livetime_fraction(triggers / product.data["timedel"].to("s").reshape(-1, 1))
        livefrac_lower, _, _ = get_livetime_fraction(triggers_lower / product.data["timedel"].to("s").reshape(-1, 1))
        livefrac_upper, _, _ = get_livetime_fraction(triggers_upper / product.data["timedel"].to("s").reshape(-1, 1))

        livefrac = livefrac.reshape(livefrac.shape + (1, 1))
        livefrac_lower = livefrac_lower.reshape(livefrac_lower.shape + (1, 1))
        livefrac_upper = livefrac_upper.reshape(livefrac_upper.shape + (1, 1))

    # ---- resolve the pixel summation group ------------------------------

    if len(shape) < 4:
        livefrac_error = counts * inverse_livetime_error[:, None, None, None]
    elif per_pixel:
        livefrac_error = (counts / livefrac_upper - counts / livefrac_lower) / 2

    else:
        n_pix = counts.shape[2]

        if pixel_indices is None:
            pix = np.arange(n_pix)
        else:
            pix = np.asarray(pixel_indices)
            if pix.ndim == 2:
                pix = np.asarray(_flat_indices(pix))
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
        e_tot = (c_tot / livefrac_upper - c_tot / livefrac_lower) / 2

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


def _elut_correction_sort(bins, bins_actual, detector_sum, pixel_indices, detector_indices):
    """
    Compute the ELUT correction factor for the selected detectors and pixels.

    `bins` and `bins_actual` are averaged over the selected pixels, and the
    factor is ``bins / bins_actual``. Detectors are handled as follows:

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

    Returns
    -------
    numpy.ndarray
        The ELUT correction factor, broadcastable against the counts.
    """

    # np.mean rather than np.nanmean: the calibrated width of the open top bin is NaN for every detector and
    # pixel, so it stays NaN (that bin is never returned) without nanmean's "Mean of empty slice" warning
    pixel_indices = _flat_indices(np.asarray(pixel_indices))
    detector_indices = np.asarray(detector_indices)

    bins = np.mean(bins[:, :, pixel_indices, :], axis=2, keepdims=True)
    bins_actual = np.mean(bins_actual[:, :, pixel_indices, :], axis=2, keepdims=True)

    if detector_sum:
        detector_indices = _flat_indices(detector_indices)
        bins = np.mean(bins[:, detector_indices, :, :], axis=1, keepdims=True)
        bins_actual = np.mean(bins_actual[:, detector_indices, :, :], axis=1, keepdims=True)
        return bins / bins_actual
    if detector_indices.ndim == 2:
        groups = [np.arange(start, end + 1) for start, end in detector_indices]
        return _normalize_elut_by_group_detector_mean(bins, bins_actual, groups)
    return bins / bins_actual


def _elut_factor(product, detector_indices, pixel_indices, detector_sum):
    """
    The ELUT correction factor of the energy bins `~stixpy.product.sources._countdata._load` keeps, for pixel data.

    Averaged over the selected detectors and pixels that are on, as `_elut_correction_sort` sets out. Spectrograms are
    not ELUT corrected: averaged over the detectors and pixels summed on board, the calibrated edges are within 0.01 keV
    of the science edges, and STIX-GSW does not correct them either.

    Parameters
    ----------
    product : ScienceData
        The pixel data product.
    detector_indices, pixel_indices : numpy.ndarray
        The selection from `_indices_check`, flat or [start, end] pairs.
    detector_sum : bool
        Whether the detectors are summed into one spectrum.

    Returns
    -------
    numpy.ndarray
        The factor, broadcastable against the counts.
    """
    _, _, bins, bins_actual = get_elut_correction(np.array(product.energies["channel"]), product)

    # the selected detectors and pixels that are on, as [start, end] ranges may span ones that are off
    detectors = (
        np.intersect1d(_flat_indices(detector_indices), _on_indices(product.detector_masks.masks))
        if detector_sum
        else detector_indices
    )
    pixels = np.intersect1d(_flat_indices(pixel_indices), _on_indices(product.pixel_masks.masks))

    factor = _elut_correction_sort(bins, bins_actual, detector_sum, pixels, detectors)
    return factor[..., _kept_energy_bins(product.energies)]


def _apply_elut(data, elut_cor_fac):
    """
    Apply the ELUT correction factor to the counts and their uncertainty.

    Parameters
    ----------
    data : _CountData
        Data from `_load`.
    elut_cor_fac : numpy.ndarray
        From `_elut_factor`.

    Returns
    -------
    _CountData
    """
    return replace(
        data, counts=data.counts * elut_cor_fac, counts_err=data.counts_err * elut_cor_fac, elut_cor_fac=elut_cor_fac
    )


def _add_livetime(data, product, pixel_indices, *, ssw_compatible=False):
    """
    Add the livetime fraction and its uncertainty (see `_livefrac`) to the data, without correcting the counts.

    Parameters
    ----------
    data : _CountData
        Data from `_load`, with or without the ELUT correction.
    product : ScienceData
        The product `data` was loaded from, for its triggers.
    pixel_indices : numpy.ndarray or None
        Pixels the livetime uncertainty is spread over, flat or [start, end] pairs.

    Returns
    -------
    _CountData
    """
    livefrac, livefrac_err = _livefrac(
        product,
        data.counts,
        pixel_indices,
        per_pixel=data.error_model == "correlated",
        ssw_compatible=ssw_compatible,
    )
    return replace(data, livefrac=livefrac, livefrac_err=livefrac_err)


def _livetime_groups(data, detector_indices):
    """
    The groups of detectors that share one livetime, the mean livetime fraction of the group.

    Parameters
    ----------
    data : _CountData
        Data from `_load`, on the full detector axis.
    detector_indices : numpy.ndarray or None
        The detector selection, flat or [start, end] pairs.

    Returns
    -------
    list of numpy.ndarray
        For a spectrogram, its one detector position. Otherwise every detector that is on with no selection, each
        [start, end] range's detectors that are on, and the flat detectors together.
    """
    if data.counts.shape[DETECTOR] == 1:
        return [np.arange(1)]
    detectors_on = np.flatnonzero(data.detectors.any(axis=0))
    if detector_indices is None:
        return [detectors_on]
    detector_indices = np.asarray(detector_indices)
    if detector_indices.ndim == 2:
        return [np.intersect1d(np.arange(start, end + 1), detectors_on) for start, end in detector_indices]
    return [detector_indices]


def _adc_covariance(errors, detectors):
    """Return each detector's contribution to the off-diagonal ADC variance."""
    adc = STIX_INSTRUMENT.subcol_adc_mapping[detectors]
    covariance = np.zeros_like(errors**2)
    for channel in np.unique(adc):
        members = adc == channel
        if np.count_nonzero(members) > 1:
            shared = errors[:, members]
            covariance[:, members] = shared * (shared.sum(axis=1, keepdims=True) - shared)
    return covariance


def _correct_livetime(data, groups, *, shared_adc=False):
    """
    Apply the livetime correction by detector group, and sum the pixels.

    The livetime uncertainty is added to the count uncertainty (see `_livetime_uncertainty`), each group of
    detectors is scaled to its mean livetime (see `_apply_livetime`), and the pixels are summed.

    With the "correlated" error model the livetime uncertainty is kept apart instead: the statistical uncertainty is
    summed over pixels in quadrature and the livetime uncertainty linearly, as the pixels of a detector share its
    livetime, and both are scaled like the counts.

    Parameters
    ----------
    data : _CountData
        Data with a livetime fraction and uncertainty (see `_add_livetime`).
    groups : list of array_like
        Detector groups, from `_livetime_groups`.

    Returns
    -------
    _CountData
        With one pixel position. With the "legacy" error model `livefrac_err` is kept as it was.
    """
    changes = {}
    if data.error_model == "legacy":
        counts_err = _livetime_uncertainty(data.counts_err, data.livefrac_err, data.livefrac)
        if shared_adc and data.livefrac_err is not None:
            err = np.sqrt(np.nansum(data.livefrac_err**2, axis=PIXEL, keepdims=True))
            variance = counts_err**2
            for group in groups:
                variance[:, group] += _adc_covariance(err[:, group], group)
            counts_err = np.sqrt(variance)
    else:
        counts_err = np.sqrt(np.nansum((data.counts_err / data.livefrac) ** 2, axis=PIXEL, keepdims=True))
        livefrac_err = np.nansum(data.livefrac_err, axis=PIXEL, keepdims=True)
        changes["livefrac_err"] = _apply_livetime(livefrac_err, livefrac_err, data.livefrac, groups)[1]
    counts, counts_err, livefrac = _apply_livetime(data.counts, counts_err, data.livefrac, groups)
    return replace(
        data,
        counts=np.nansum(counts, axis=PIXEL, keepdims=True),
        counts_err=counts_err,
        livefrac=livefrac,
        pixels=data.pixels.any(axis=0, keepdims=True),
        **changes,
    )
