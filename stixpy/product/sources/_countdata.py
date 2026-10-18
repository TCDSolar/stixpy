"""
The count data passed between the steps of `~stixpy.product.sources.science.ScienceData.get_data`, and the steps
that load and select it.
"""

from dataclasses import dataclass, replace

import numpy as np

import astropy.units as u
from astropy.table import QTable
from astropy.time import Time

from stixpy.product.sources._corrections import _correct_livetime
from stixpy.product.sources._rcr import _handle_nested_pairs, _rcr_warning, _shift_rcr
from stixpy.product.sources._selection import (
    DETECTOR,
    ENERGY,
    PIXEL,
    TIME,
    _flat_indices,
    _kept_energy_bins,
    _on_indices,
)

__all__ = []


def _range(values, axis, start, end):
    """``values[start:end + 1]`` along `axis`."""
    return values[(slice(None),) * axis + (slice(start, end + 1),)]


def _sum(values, axis, start, end, data):
    return np.sum(_range(values, axis, start, end), axis=axis, keepdims=True)


def _quadrature(values, axis, start, end, data):
    return np.sqrt(np.sum(_range(values, axis, start, end) ** 2, axis=axis, keepdims=True))


def _rms(values, axis, start, end, data):
    return np.sqrt(np.mean(_range(values, axis, start, end) ** 2, axis=axis, keepdims=True))


def _mean(values, axis, start, end, data):
    return np.mean(_range(values, axis, start, end), axis=axis, keepdims=True)


def _duration_weighted_mean(values, axis, start, end, data):
    durations = data.durations.to_value(u.s).reshape((-1,) + (1,) * (values.ndim - 1))
    weighted = _range(values * durations, axis, start, end)
    return np.sum(weighted, axis=axis, keepdims=True) / np.sum(durations[start : end + 1])


# How each array field is combined over a [start, end] range of each axis. None leaves the field as it is on that
# axis: it is neither indexed nor combined.
_REDUCTIONS = {
    # time, detector, pixel, energy
    "counts": (_sum, _sum, _sum, _sum),
    "counts_err": (_quadrature, _quadrature, _quadrature, _quadrature),
    "livefrac": (_duration_weighted_mean, _mean, _mean, _mean),
    # The livetime uncertainty is reduced differently on each axis, as it was before `_CountData`: in quadrature
    # over pixels and times, linearly over energies, and as an RMS over detectors (and over all times with
    # ``sum_all_times``, see `_sum_all_times`). Kept as it is for now, to be made consistent with the other error
    # fixes.
    "livefrac_err": (_quadrature, _rms, _quadrature, _sum),
    # The ELUT factor has never been selected or combined over detectors, so with ``sunkit_spex_detector_sum=False``
    # it keeps every detector. Kept as it is for now.
    "elut_cor_fac": (_mean, None, _mean, _mean),
}


@dataclass(frozen=True, eq=False)
class _CountData:
    """
    Counts and everything that is selected and combined with them, on the (time, detector, pixel, energy) axes.

    The array fields are broadcastable against `counts`: an axis of length 1 is shared by every position on that
    axis, and is left alone by indexing and `combine`. `combine` reduces each field as `_REDUCTIONS` sets out.
    """

    counts: u.Quantity
    """Counts, shape (time, detector, pixel, energy)."""
    counts_err: u.Quantity
    """1-sigma uncertainty of `counts`."""
    livefrac: np.ndarray | None
    """Livetime fraction, e.g. shape (time, detector, 1, 1), or None without the livetime correction."""
    livefrac_err: u.Quantity | None
    """Livetime uncertainty in counts, or None."""
    elut_cor_fac: np.ndarray | None
    """ELUT correction factor applied to the counts, or None without the ELUT correction."""
    times: Time
    """Centre of each time bin."""
    durations: u.Quantity
    """Duration of each time bin."""
    energies: QTable
    """Energy table, one row per energy bin, with at least ``e_low`` and ``e_high`` columns."""
    detectors: np.ndarray
    """Shape (detector, 32), True for the detectors that are on and held by each position on the detector axis."""
    pixels: np.ndarray
    """Shape (pixel, 12), True for the pixels that are on and held by each position on the pixel axis."""
    rcr: np.ndarray
    """RCR state of each time bin."""

    def _arrays(self):
        return {name: getattr(self, name) for name in _REDUCTIONS}

    def _applies(self, name, values, axis):
        """Whether field `name` is indexed and combined on `axis`."""
        return (
            values is not None
            and _REDUCTIONS[name][axis] is not None
            and (name == "counts" or values.shape[axis] != 1 or self.counts.shape[axis] == 1)
        )

    def __getitem__(self, item):
        """
        Index the (time, detector, pixel, energy) axes, one after the other, as numpy does each axis.

        An integer keeps its axis, as ``[i]`` would. Every array field is indexed (see `_applies`), and so are the
        times, durations and RCR states, the detector and pixel tables and the energy table.
        """
        if not isinstance(item, tuple):
            item = (item,)
        data = self
        for axis, index in enumerate(item):
            if isinstance(index, slice) and index == slice(None):
                continue
            if isinstance(index, (int, np.integer)):
                index = [index]
            data = data._index_axis(axis, index)
        return data

    def _index_axis(self, axis, index):
        where = (slice(None),) * axis + (index,)
        changes = {name: values[where] for name, values in self._arrays().items() if self._applies(name, values, axis)}
        if axis == TIME:
            changes.update(times=self.times[index], durations=self.durations[index], rcr=np.asarray(self.rcr)[index])
        elif axis == DETECTOR:
            changes["detectors"] = self.detectors[index]
        elif axis == PIXEL:
            changes["pixels"] = self.pixels[index]
        else:
            changes["energies"] = self.energies[index]
        return replace(self, **changes)

    def combine(self, axis, ranges):
        """
        Combine each inclusive [start, end] range of `axis` into one bin.

        The array fields are reduced as `_REDUCTIONS` sets out. A time range is centred between the start of its
        first bin and the end of its last, its duration is that span, and it takes the RCR state of its first bin.
        An energy range runs from the lower edge of its first bin to the upper edge of its last, and the energy table
        is rebuilt with only ``e_low`` and ``e_high``. A detector or pixel range holds every detector or pixel of the
        bins in it.

        Parameters
        ----------
        axis : int
            `TIME`, `DETECTOR`, `PIXEL` or `ENERGY`.
        ranges : array_like
            [start, end] index pairs, inclusive at both ends.

        Returns
        -------
        _CountData
        """
        ranges = [(int(start), int(end)) for start, end in ranges]
        changes = {
            name: np.concatenate(
                [_REDUCTIONS[name][axis](values, axis, start, end, self) for start, end in ranges], axis=axis
            )
            for name, values in self._arrays().items()
            if self._applies(name, values, axis)
        }

        if axis == TIME:
            times, durations = [], []
            for start, end in ranges:
                bin_start = self.times[start] - self.durations[start] * 0.5
                span = self.times[end] + self.durations[end] * 0.5 - bin_start
                times.append(bin_start + span * 0.5)
                durations.append(span.to(u.s))
            rcr = np.asarray(self.rcr)
            changes.update(
                times=Time(times), durations=np.hstack(durations), rcr=np.array([rcr[start] for start, _ in ranges])
            )
        elif axis == DETECTOR:
            changes["detectors"] = np.stack([self.detectors[start : end + 1].any(axis=0) for start, end in ranges])
        elif axis == PIXEL:
            changes["pixels"] = np.stack([self.pixels[start : end + 1].any(axis=0) for start, end in ranges])
        else:
            edges = [(self.energies["e_low"][start].value, self.energies["e_high"][end].value) for start, end in ranges]
            changes["energies"] = QTable(np.atleast_2d(edges) * u.keV, names=["e_low", "e_high"])
        return replace(self, **changes)


def _full_layout(values, detector_masks, pixel_masks, name="values"):
    """
    Place a (time, detector, pixel, energy) data column on the full 32 x 12 grid.

    L1 pixel data files store only the enabled detectors and pixels, so position
    n on those axes is not necessarily detector or pixel n. As in IDL's
    ``stx_read_pixel_data_fits_file``, the stored values are placed at their
    detector and pixel numbers, with zeros for detectors and pixels that are off.
    After this every detector and pixel index is a real detector or pixel number.

    Parameters
    ----------
    values : astropy.units.Quantity or numpy.ndarray
        Data column, e.g. ``data['counts']`` or ``data['counts_comp_err']``.
    detector_masks : DetectorMasks or None
        Detector masks of the product.
    pixel_masks : PixelMasks or None
        Pixel masks of the product.
    name : str, optional
        Name of the column, used in error messages.

    Returns
    -------
    astropy.units.Quantity or numpy.ndarray
        The column with shape (time, 32, 12, energy). Returned unchanged if it
        already has that shape, is not 4-dimensional, has no masks, or holds summed pixel sets.

    Raises
    ------
    ValueError
        If the detector mask changes within the file, or the stored shape does
        not match the masks.
    """
    if (
        values.ndim != 4
        or values.shape[1:3] == (32, 12)
        or detector_masks is None
        or pixel_masks is None
        or pixel_masks.masks.ndim != 2
    ):
        return values

    if len(detector_masks.masks) > 1:
        raise ValueError(
            f"The detector mask changes within the file, so {name!r} can not be placed on the full detector grid."
        )
    dets = np.flatnonzero(detector_masks.masks[0])
    pixs = np.flatnonzero(pixel_masks.masks.any(axis=0))  # pixels used at any time, as stored by STIXcore

    if values.shape[1] == 32:
        values = values[:, dets]
    if values.shape[2] == 12:
        values = values[:, :, pixs]
    if values.shape[1:3] != (dets.size, pixs.size):
        raise ValueError(
            f"{name!r} has shape {values.shape}, which does not match the masks "
            f"({dets.size} detectors, {pixs.size} pixels)."
        )

    full = np.zeros((values.shape[0], 32, 12, values.shape[3]), dtype=values.dtype)
    if isinstance(values, u.Quantity):
        full = full << values.unit
    full[:, dets[:, None], pixs, :] = values
    return full


def _on(masks, n_total):
    """Boolean mask of the detectors or pixels that are on in any of `masks`, or all of them without masks."""
    on = np.zeros(n_total, dtype=bool)
    if masks is None:
        on[:] = True
    else:
        on[_on_indices(masks.masks)] = True
    return on


def _load(product):
    """
    Load the counts of a science product, without any correction.

    Pixel data are placed on the full detector and pixel grid (see `_full_layout`), and spectrogram counts are
    reshaped to (time, 1, 1, energy). The uncertainty is the statistical one, ``sqrt(counts + compression_error**2)``.
    The 0 keV bottom energy bin and the open top bin are removed (see `_kept_energy_bins`).

    Parameters
    ----------
    product : ScienceData
        Pixel data or spectrogram product.

    Returns
    -------
    _CountData
        With no livetime or ELUT correction.
    """
    detector_masks = getattr(product, "detector_masks", None)
    pixel_masks = getattr(product, "pixel_masks", None)
    counts = _full_layout(product.data["counts"], detector_masks, pixel_masks, name="counts")
    err_name = "counts_comp_err" if "counts_comp_err" in product.data.colnames else "counts_comp_comp_err"
    compression_err = _full_layout(product.data[err_name], detector_masks, pixel_masks, name=err_name)

    if counts.ndim < 4:
        # a spectrogram, summed on board over the detectors and pixels in its masks
        counts = counts.reshape(counts.shape[0], 1, 1, counts.shape[-1])
        compression_err = compression_err.reshape(counts.shape)
        detectors = _on(detector_masks, 32)[np.newaxis]
        pixels = _on(pixel_masks, 12)[np.newaxis]
    else:
        detectors = np.eye(counts.shape[1], 32, dtype=bool) & _on(detector_masks, 32)
        if pixel_masks is not None and np.ndim(pixel_masks.masks) == 3:
            pixels = np.asarray(pixel_masks.masks, dtype=bool).any(axis=0)  # the pixels summed on board into each
        else:
            pixels = np.eye(counts.shape[2], 12, dtype=bool) & _on(pixel_masks, 12)

    data = _CountData(
        counts=counts,
        counts_err=np.sqrt(counts.value + compression_err.value**2) * u.ct,
        livefrac=None,
        livefrac_err=None,
        elut_cor_fac=None,
        times=product.times,
        durations=product.data["timedel"],
        energies=product.energies,
        detectors=detectors,
        pixels=pixels,
        rcr=np.asarray(product.data["rcr"]),
    )
    return data[:, :, :, _kept_energy_bins(product.energies)]


def _select_axis(data, axis, indices):
    """
    Select on one axis: flat indices keep those bins, in order and once each, and [start, end] pairs combine each
    inclusive range into one bin (see `_CountData.combine`). None keeps the axis as it is.
    """
    if indices is None:
        return data
    indices = np.asarray(indices)
    if indices.ndim == 2:
        return data.combine(axis, indices)
    keep = np.isin(np.arange(data.counts.shape[axis]), indices)
    return data[(slice(None),) * axis + (keep,)]


def _rcr_track(data):
    """
    The counts summed over every energy bin, which the RCR state changes are aligned to (see `_shift_rcr`).
    """
    summed = data.combine(ENERGY, [[0, data.counts.shape[ENERGY] - 1]])
    # nansum, as the RCR signal has always skipped NaN counts
    return replace(summed, counts=np.nansum(data.counts, axis=ENERGY, keepdims=True))


def _select(data, detector_indices, pixel_indices, energy_indices, *, livetime_groups=None, n_detectors=None):
    """
    Select the detectors, pixels and energies, apply the livetime correction, and align the RCR states.

    The steps run in this order, each on one axis (see `_select_axis`):

    1. Pixel selection.
    2. Energy selection. The RCR states are aligned to the counts summed over every energy, so a second copy of
       the data, summed over energy, goes through the steps that follow too.
    3. Livetime correction by detector group, and a sum over pixels (see `_correct_livetime`), if
       `livetime_groups` is given.
    4. Detector selection.
    5. RCR alignment (see `_shift_rcr`).

    Parameters
    ----------
    data : _CountData
        Data from `_load`, with any ELUT correction, livetime or background subtraction applied.
    detector_indices, pixel_indices : numpy.ndarray or None
        Flat indices or [start, end] pairs, numbering the detector and pixel axes, or None to keep the axis.
    energy_indices : numpy.ndarray or None
        Flat indices or [start, end] pairs of the energy bins of `data` (see `_energy_positions`), or None.
    livetime_groups : list of array_like, optional
        Detector groups that share one livetime, for `_correct_livetime`. None to skip the livetime correction,
        when it is not wanted or has already been applied.
    n_detectors : int, optional
        Number of detectors the RCR signal is scaled to. By default, the number of selected detectors with counts.

    Returns
    -------
    _CountData
        The selected data, with the RCR states aligned.
    """
    data = _select_axis(data, PIXEL, pixel_indices)
    track = _rcr_track(data)
    data = _select_axis(data, ENERGY, energy_indices)

    if livetime_groups is not None:
        data = _correct_livetime(data, livetime_groups)
        track = _correct_livetime(track, livetime_groups)

    if n_detectors is None:
        selected = slice(None) if detector_indices is None else _flat_indices(detector_indices)
        n_detectors = np.count_nonzero(np.nansum(np.abs(track.counts[:, selected]), axis=(0, 2, 3)))

    data = _select_axis(data, DETECTOR, detector_indices)
    track = _select_axis(track, DETECTOR, detector_indices)
    return replace(data, rcr=_shift_rcr(track, n_detectors))


def _select_times(data, time_indices, sum_all_times):
    """
    Select the time bins, and optionally sum them into one.

    Parameters
    ----------
    data : _CountData
        Data with the RCR states aligned (see `_select`).
    time_indices : list, numpy.ndarray or None
        Flat time indices or [start, end] pairs, as returned by `_time_indices_format`, or None for every bin.
    sum_all_times : bool
        If True, sum the selected time bins into one (see `_sum_all_times`).

    Returns
    -------
    _CountData

    Raises
    ------
    ValueError
        If the RCR state changes within a time range, or between the bins summed by `sum_all_times`.

    Warns
    -----
    UserWarning
        If the selected time bins or ranges are in different RCR states.
    """
    if time_indices is not None:
        if np.ndim(time_indices) == 1:
            _rcr_warning(list(time_indices), data.rcr)
        else:
            _handle_nested_pairs(time_indices, data.rcr)
        data = _select_axis(data, TIME, time_indices)

    if sum_all_times and data.counts.shape[TIME] > 1:
        data = _sum_all_times(data)
    return data


def _sum_all_times(data):
    """
    Sum every time bin into one, from the start of the first bin to the end of the last.

    Raises
    ------
    ValueError
        If the RCR state changes between the bins.
    """
    rcr = np.unique(data.rcr)
    if rcr.size != 1:
        raise ValueError(
            "Cannot sum all times as the RCR state changes between the selected time ranges "
            f"(RCR states {rcr.astype(int).tolist()}). Select time ranges in a single RCR state."
        )

    summed = data.combine(TIME, [[0, data.counts.shape[TIME] - 1]])
    # Unlike a time range, which takes its span as its duration, adds the livetime uncertainty in quadrature and
    # weights the livetime by duration, summing all times has always summed the durations, so gaps between the bins
    # are left out, taken an RMS of the livetime uncertainty, and weighted the livetime by each bin's share of the
    # summed duration. Kept as it is for now, to be made consistent with the other error fixes.
    changes = {"durations": np.sum(data.durations, keepdims=True), "rcr": rcr}
    if data.livefrac is not None:
        weights = (data.durations / np.sum(data.durations)).to_value(u.one)
        weights = weights.reshape((-1,) + (1,) * (data.livefrac.ndim - 1))
        changes["livefrac"] = np.sum(data.livefrac * weights, axis=TIME, keepdims=True)
    if data.livefrac_err is not None:
        changes["livefrac_err"] = np.sqrt(np.mean(data.livefrac_err**2, axis=TIME, keepdims=True))
    return replace(summed, **changes)
