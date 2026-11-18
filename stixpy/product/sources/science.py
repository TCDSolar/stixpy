from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import ConciseDateFormatter, DateFormatter, HourLocator
from matplotlib.widgets import Slider

import astropy.units as u
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.utils.masked import Masked
from astropy.visualization import quantity_support

from sunpy.time.timerange import TimeRange
from sunpy.util import deprecated

from stixpy.calibration.elut import get_elut
from stixpy.io.readers import read_subc_params
from stixpy.product.product import L1Product
from stixpy.product.sources._background import _bkg_check, _bkg_selection, _restrict, _used
from stixpy.product.sources._corrections import _add_livetime, _apply_elut, _elut_factor, _livetime_groups
from stixpy.product.sources._countdata import ERROR_MODELS, _load, _select, _select_times
from stixpy.product.sources._selection import (
    DETECTOR,
    PIXEL,
    _energy_indices_format,
    _energy_positions,
    _indices_check,
    _on_indices,
    _time_indices_format,
)
from stixpy.visualisation.plotters import PixelPlotter

__all__ = [
    "ScienceData",
    "SelectedData",
    "DataSelection",
    "subtract_background",
    "RawPixelData",
    "CompressedPixelData",
    "SummedCompressedPixelData",
    "Visibility",
    "Spectrogram",
    "TimesSeriesPlotMixin",
    "SpectrogramPlotMixin",
    "PixelPlotMixin",
    "PPrintMixin",
    "IndexMasks",
    "DetectorMasks",
    "PixelMasks",
    "EnergyEdgeMasks",
]


quantity_support()


SubCollimatorConfig = read_subc_params(
    Path(__file__).parent.parent.parent / "config" / "data" / "detector" / "stx_subc_params.csv"
)


def _selected_active(
    product, detector_indices, pixel_indices, time_indices, sum_all_times, livetime_correction, sum_detectors=False
):
    """Track which detector and pixel positions contribute to each selected time bin."""
    if product.data["counts"].ndim < 4 or product.pixel_masks.masks.ndim != 2:
        return None

    n_times = len(product.times)
    detectors_on = np.zeros((n_times, 32), dtype=bool)
    pixels_on = np.zeros((n_times, 12), dtype=bool)
    for mask, rows in zip(product.detector_masks.masks, product.detector_masks.indices):
        detectors_on[rows] = mask
    for mask, rows in zip(product.pixel_masks.masks, product.pixel_masks.indices):
        pixels_on[rows] = mask
    active = detectors_on[:, :, np.newaxis] & pixels_on[:, np.newaxis, :]

    if pixel_indices is not None:
        pixels = np.asarray(pixel_indices)
        if pixels.ndim == 1:
            active = active[:, :, pixels]
        else:
            active = np.concatenate(
                [np.any(active[:, :, lo : hi + 1], axis=2, keepdims=True) for lo, hi in pixels], axis=2
            )
    if livetime_correction:
        active = np.any(active, axis=2, keepdims=True)

    if detector_indices is not None:
        detectors = np.asarray(detector_indices)
        if detectors.ndim == 1:
            active = active[:, detectors]
        else:
            active = np.concatenate(
                [np.any(active[:, lo : hi + 1], axis=1, keepdims=True) for lo, hi in detectors], axis=1
            )
    if sum_detectors:
        active = np.any(active, axis=1, keepdims=True)

    if time_indices is not None:
        times = np.asarray(time_indices)
        if times.ndim == 1:
            active = active[times]
        else:
            active = np.concatenate([np.any(active[lo : hi + 1], axis=0, keepdims=True) for lo, hi in times], axis=0)
    if sum_all_times and active.shape[0] > 1:
        active = np.any(active, axis=0, keepdims=True)
    return active


@dataclass(frozen=True, eq=False)
class DataSelection:
    """
    The selection `ScienceData.get_data` made, resolved against the product, kept in `SelectedData.selection`.

    `subtract_background` applies it to the background product, and `stixpy.spectroscopy.to_spectrum` builds the
    response matrix from it, so both use exactly the detectors, pixels and energies of the counts.
    """

    detector_indices: np.ndarray | None
    """Detector selection as applied: flat indices or [start, end] pairs, or None to keep every detector position."""
    pixel_indices: np.ndarray | None
    """Pixel selection as applied, like `detector_indices`."""
    checked_detectors: np.ndarray | None
    """The selected detectors that are on, flat or [start, end] pairs. None for a spectrogram."""
    checked_pixels: np.ndarray | None
    """The selected pixels that are on, flat or [start, end] pairs. None for a spectrogram."""
    detectors: np.ndarray
    """Shape (detector, 32): True for the detectors summed into each position on the detector axis of the counts."""
    pixels: np.ndarray
    """Shape (12,): True for the pixels the counts hold."""
    sum_detectors: bool
    """Whether the detectors were summed into one."""
    livetime_correction: bool
    """Whether the livetime correction was applied."""
    elut_correction: bool
    """Whether the ELUT correction was applied (never for a spectrogram)."""
    ssw_compatible: bool
    """Use SSW conventions for spectrogram livetime and legacy detector uncertainties."""
    error_model: str
    """The error model, "legacy" or "correlated"."""
    spectrogram: bool
    """Whether the product is a spectrogram, summed on board over `detectors` and `pixels`."""
    summed_pixels: bool
    """Whether the product holds pixels summed on board (sci-xray-scpd)."""
    elut_file: str | None
    """The ELUT file of the ELUT correction, or None without it."""
    distance: u.Quantity | None
    """Distance of Solar Orbiter from the Sun, or None if the product has no ``DSUN_OBS``."""
    time_range: TimeRange
    """Time range of the product."""


class SelectedData(NamedTuple):
    r"""
    The data selected by `ScienceData.get_data`.

    Returned uncertainties are per bin. Summing them in quadrature outside this API ignores covariance between
    bins that share a trigger counter.

    Access the fields by name, e.g. ``data.counts``. Fields may be added in later versions, so code that unpacks
    every field by position may break.
    """

    counts: u.Quantity
    """Counts, shape ``(time, detector, pixel, energy)``, normalised as requested by ``vtype``."""
    counts_err: u.Quantity
    """1-sigma uncertainty of `counts`, same shape and unit, including the livetime uncertainty when corrected."""
    t_norm: u.Quantity
    """Duration of each time bin in seconds, shape ``(time, 1, 1, 1)``."""
    e_norm: u.Quantity
    """Width of each energy bin in keV, shape ``(1, 1, 1, energy)``."""
    livefrac: np.ndarray | None
    """Livetime fraction, shape ``(time, detector, 1, 1)``, or None without the livetime correction."""
    livefrac_error: u.Quantity | None
    """Livetime uncertainty in counts, already included in `counts_err`, or None without the livetime correction.
    With ``error_model="legacy"`` it is not normalised by ``vtype``."""
    elut_cor_fac: np.ndarray | None
    """ELUT correction factor applied to the counts, broadcastable against them, or None without the ELUT
    correction."""
    times: Time
    """Centre of each time bin."""
    energies: QTable
    """Energy table with ``e_low`` and ``e_high`` columns, one row per energy bin."""
    rcr: np.ndarray
    """RCR (attenuator) state of each time bin, aligned to the jumps in the counts as STIX-GSW does."""
    vtype: str
    """Normalisation of `counts`: 'c', 'cr' or 'dcr' (see `ScienceData.get_data`)."""
    selection: DataSelection
    """The selection the data were made with."""

    ssw_spectrogram_variance: np.ndarray | None = None
    """Coefficients of the extra SSW spectrogram livetime variance in count space."""

    background_exposure_rss: u.Quantity | None = None
    """Effective exposure for SSW-style background error propagation across time."""

    @classmethod
    def from_count_data(cls, data, vtype, selection):
        """
        Normalise selected count data as `vtype` asks.

        Parameters
        ----------
        data : _CountData
            The selected data.
        vtype : {'c', 'cr', 'dcr'}
            Counts, count rate in ct/s, or differential count rate in ct/(s keV). With a livetime fraction, the
            rates are also divided by it.
        selection : DataSelection
            The selection the data were made with.

        Returns
        -------
        SelectedData

        Raises
        ------
        ValueError
            If `vtype` is not 'c', 'cr' or 'dcr'.
        """
        e_norm = (data.energies["e_high"] - data.energies["e_low"])[np.newaxis, np.newaxis, np.newaxis, :]
        t_norm = data.durations[:, np.newaxis, np.newaxis, np.newaxis].to(u.s)

        livefrac = data.livefrac
        if livefrac is not None:
            livefrac = np.nanmean(livefrac, axis=PIXEL, keepdims=True)

        if vtype == "c":
            norm = 1
        elif vtype == "cr":
            norm = 1 / t_norm if livefrac is None else 1 / (t_norm * livefrac)
        elif vtype == "dcr":
            norm = 1 / (e_norm * t_norm) if livefrac is None else 1 / (e_norm * t_norm * livefrac)
        else:
            raise ValueError("vtype must be one of 'c', 'cr', 'dcr'.")

        counts_err = data.total_err() * norm
        if data.ssw_spectrogram_variance is not None:
            extra = np.sqrt(data.ssw_spectrogram_variance[..., 0]) * u.ct * norm
            counts_err = np.hypot(counts_err, extra)

        return cls(
            counts=data.counts * norm,
            background_exposure_rss=data.background_exposure_rss,
            counts_err=counts_err,
            ssw_spectrogram_variance=data.ssw_spectrogram_variance,
            t_norm=t_norm,
            e_norm=e_norm,
            livefrac=livefrac,
            livefrac_error=data.livefrac_err
            if data.error_model == "legacy" or data.livefrac_err is None
            else data.livefrac_err * norm,
            elut_cor_fac=data.elut_cor_fac,
            times=data.times,
            energies=data.energies,
            rcr=data.rcr,
            vtype=vtype,
            selection=selection,
        )


class PPrintMixin:
    """
    Provides pretty printing for index masks.
    """

    @staticmethod
    def _pprint_indices(indices):
        groups = np.split(np.r_[: len(indices)], np.where(np.diff(indices) != 1)[0] + 1)
        out = ""
        for group in groups:
            if group.size < 3:
                out += f"{indices[group]}"
            else:
                out += f"[{indices[group[0]]}...{indices[group[-1]]}]"

        return out


class IndexMasks(PPrintMixin):
    """
    Index mask class to store masked indices.

    Attributes
    ----------
    masks : `numpy.ndarray`
        The mask arrays
    indices : `numpy.ndarray`
        The indices the mask/s applies to

    """

    def __init__(self, mask_array):
        masks = np.unique(mask_array, axis=0)
        indices = [np.argwhere(np.all(mask_array == mask, axis=1)).reshape(-1) for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f"{self.__class__.__name__}\n"
        for m, i in zip(self.masks, self.indices):
            text += (
                f"    {self._pprint_indices(i)}: [{','.join(np.where(m, np.arange(m.size), np.full(m.size, '_')))}]\n"
            )
        return text


class DetectorMasks(IndexMasks):
    """
    Detector Index Masks
    """

    pass


class EnergyEdgeMasks(IndexMasks):
    """
    Energy Edges Mask
    """

    @property
    def energy_mask(self):
        """
        Return mask of energy channels from mask of energy edges.

        Returns
        -------
        `np.array`
        """
        energy_bin_mask = (self.masks & np.roll(self.masks, 1))[0, 1:]
        indices = np.where(energy_bin_mask == 1)
        energy_bin_mask[indices[0][0] : indices[0][-1] + 1] = 1
        return energy_bin_mask


class PixelMasks(PPrintMixin):
    """
    Pixel Index Masks
    """

    def __init__(self, pixel_masks):
        masks = np.unique(pixel_masks, axis=0)
        indices = []
        if masks.ndim == 2:
            indices = [np.argwhere(np.all(pixel_masks == mask, axis=1)).reshape(-1) for mask in masks]
        elif masks.ndim == 3:
            indices = [np.argwhere(np.all(pixel_masks == mask, axis=(1, 2))).reshape(-1) for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f"{self.__class__.__name__}\n"
        for m, i in zip(self.masks, self.indices):
            text += f"    {self._pprint_indices(i)}: [{str(np.where(m.shape[0], m, np.full(m.shape, '_')))}]\n"
        return text


class SpectrogramPlotMixin:
    """
    Spectrogram plot mixin providing spectrogram plotting for pixel data.
    """

    def plot_spectrogram(
        self,
        axes=None,
        vtype="dcr",
        time_indices=None,
        energy_indices=None,
        detector_indices="all",
        pixel_indices="all",
        **plot_kwargs,
    ):
        """
        Plot a spectrogram for the selected time and energies.

        The data are not corrected for livetime or the ELUT.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
            Axes to plot on. A new figure is created if None.
        vtype : {'c', 'cr', 'dcr'}, optional
            Normalisation of the plotted values: counts ('c'), count rate in ct/s
            ('cr'), or differential count rate in ct/(s keV) ('dcr', default).
        time_indices : list, numpy.ndarray, str or astropy.time.Time, optional
            Flat indices keep those time bins and [start, end] pairs sum each range,
            e.g. ``[0, 2, 5]`` or ``[[0, 2], [3, 5]]``. Times are also accepted; see
            `ScienceData.get_data`.
        pixel_indices : list or numpy.ndarray, optional
            A single pixel, e.g. ``[4]``, or one [start, end] range to sum, e.g.
            ``[[0, 11]]``. Default "all" sums every pixel. Can not be used with a
            spectrogram product.
        detector_indices : list or numpy.ndarray, optional
            A single detector, e.g. ``[5]``, or one [start, end] range to sum, e.g.
            ``[[0, 31]]``. Default "all" sums every detector. Can not be used with a
            spectrogram product.
        energy_indices : list, numpy.ndarray or astropy.units.Quantity, optional
            Flat indices keep those energy bins (rows of the energy table) and
            [start, end] pairs sum each range. Energies can also be given in keV,
            e.g. ``[[6, 10], [25, 100]] * u.keV``; each range takes the bins whose
            centres lie inside it.
        **plot_kwargs
            Passed to :meth:`~matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        ValueError
            If more than one detector or pixel (or more than one range of them) is
            selected, or if detector or pixel indices are given for a spectrogram
            product.

        Notes
        -----
        The units of the plotted data are determined by the `vtype` parameter:

        - 'c': counts
        - 'cr': counts per second
        - 'dcr': counts per second per keV
        """

        if axes is None:
            fig, axes = plt.subplots()

        counts_shape = self.data["counts"].shape
        if len(counts_shape) != 4:
            # if spectrogram can't do anything with pixel or detector indices
            if detector_indices != "all" or pixel_indices != "all":
                raise ValueError("Detector and or pixel indices have can not be used with spectrogram")

            pid = None
            did = None
        else:
            if isinstance(detector_indices, str) and detector_indices == "all":
                did = None  # every detector that is switched on, summed below
            else:
                det_idx_arr = np.array(detector_indices)
                if det_idx_arr.ndim == 1 and det_idx_arr.size != 1:
                    raise ValueError(
                        "Spectrogram plots can only show data from a single "
                        "detector or summed over a number of detectors"
                    )
                elif det_idx_arr.ndim == 2 and det_idx_arr.shape[0] != 1:
                    raise ValueError("Spectrogram plots can only one sum detector or summed over a number of detectors")
                did = detector_indices

            if isinstance(pixel_indices, str) and pixel_indices == "all":
                pid = None  # every pixel that is switched on, summed below
            else:
                pix_idx_arr = np.array(pixel_indices)
                if pix_idx_arr.ndim == 1 and pix_idx_arr.size != 1:
                    raise ValueError(
                        "Spectrogram plots can only show data from a single "
                        "detector or summed over a number of detectors"
                    )
                elif pix_idx_arr.ndim == 2 and pix_idx_arr.shape[0] != 1:
                    raise ValueError("Spectrogram plots can only one sum detector or summed over a number of detectors")
                pid = pixel_indices

        data = self.get_data(
            vtype=vtype,
            detector_indices=did,
            pixel_indices=pid,
            time_indices=time_indices,
            energy_indices=energy_indices,
            livetime_correction=False,
            elut_correction=False,
        )
        times, energies = data.times, data.energies
        timedeltas = data.t_norm.to(u.s).reshape(-1)

        # get_data has already removed the 0 keV bottom and open top energy bins
        e_edges = np.hstack([energies["e_low"], energies["e_high"][-1]]).value
        t_edges = Time(np.concatenate([times - timedeltas / 2, times[-1] + timedeltas[-1:] / 2]))

        counts = np.nansum(data.counts, axis=(1, 2), keepdims=True)  # one summed spectrogram

        pcolor_kwargs = {"norm": LogNorm(), "shading": "flat"}
        pcolor_kwargs.update(plot_kwargs)
        im = axes.pcolormesh(t_edges.datetime, e_edges, counts[:, 0, 0, :].T.value, **pcolor_kwargs)

        # axes.colorbar(im).set_label(format(counts.unit))
        axes.xaxis_date()
        # axes.set_yticks(range(y_lims[0], y_lims[1] + 1))
        # axes.set_yticklabels(labels)
        minor_loc = HourLocator()
        axes.xaxis.set_minor_locator(minor_loc)
        axes.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
        # fig.autofmt_xdate()
        # fig.tight_layout()
        for i in plt.get_fignums():
            if axes in plt.figure(i).axes:
                plt.sca(axes)
                plt.sci(im)

        return im


class TimesSeriesPlotMixin:
    """
    TimesSeries plot mixin providing timeseries plotting for pixel data.
    """

    def plot_timeseries(
        self,
        vtype="dcr",
        time_indices=None,
        energy_indices=None,
        detector_indices="all",
        pixel_indices="all",
        axes=None,
        error_bar=False,
        **plot_kwarg,
    ):
        """
        Plot a times series of the selected times and energies.

        Parameters
        ----------
        vtype : str
           Type of value to return control the default normalisation:
               * 'c' - count [c]
               * 'cr' - count rate [c/s]
               * 'dcr' - differential count rate [c/(s keV)]
        time_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
        detector_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `detector_indices=[0, 2, 5]` would return only the first, third and
            sixth detectors while `detector_indices=[[0, 2],[3, 5]]` would sum the data between.
        pixel_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `pixel_indices=[0, 2, 5]` would return only the first, third and
            sixth pixels while `pixel_indices=[[0, 2],[3, 5]]` would sum the data between.
        axes : optional `matplotlib.axes`
            The matplotlib axes on which to plot the time series.
        error_bar : optional `bool`
            Add error bars to plot.
        **plot_kwargs : `dict`
            Any additional arguments are passed to :meth:`~matplotlib.axes.Axes.plot`.

        Returns
        -------
        `matplotlib.axes`

        """
        if axes is None:
            fig, axes = plt.subplots()

        # "all" is every detector / pixel that is switched on, summed after get_data
        sum_detectors = isinstance(detector_indices, str) and detector_indices == "all"
        sum_pixels = isinstance(pixel_indices, str) and pixel_indices == "all"
        if sum_detectors:
            detector_indices = None
        if sum_pixels:
            pixel_indices = None

        data = self.get_data(
            vtype=vtype,
            detector_indices=detector_indices,
            pixel_indices=pixel_indices,
            time_indices=time_indices,
            energy_indices=energy_indices,
            livetime_correction=False,
            elut_correction=False,
        )
        counts, errors, times, energies = data.counts, data.counts_err, data.times, data.energies

        sum_axes = tuple(axis for axis, summed in ((1, sum_detectors), (2, sum_pixels)) if summed)
        if sum_axes:
            counts = np.sum(counts, axis=sum_axes, keepdims=True)
            errors = np.sqrt(np.sum(errors**2, axis=sum_axes, keepdims=True))

        labels = [f"{el.value} - {eh.value} keV" for el, eh in energies["e_low", "e_high"]]

        n_time, n_det, n_pix, n_energy = counts.shape

        for did, pid, eid in product(range(n_det), range(n_pix), range(n_energy)):
            if error_bar:
                lines = axes.errorbar(
                    times.to_datetime(),
                    counts[:, did, pid, eid],
                    yerr=errors[:, did, pid, eid],
                    label=labels[eid],
                    **plot_kwarg,
                )
            else:
                lines = axes.plot(times.to_datetime(), counts[:, did, pid, eid], label=labels[eid], **plot_kwarg)

        axes.set_yscale("log")
        axes.xaxis.set_major_formatter(ConciseDateFormatter(axes.xaxis.get_major_locator()))

        return lines


class PixelPlotMixin:
    """
    Pixel plot mixin providing pixel plotting for pixel data.
    """

    def plot_pixels(self, *, kind="pixel", time_indices=None, energy_indices=None, fig=None, cmap=None, **kwargs):
        pixel_plotter = PixelPlotter(self, time_indices=time_indices, energy_indices=energy_indices)
        pixel_plotter.plot(kind=kind, fig=fig, cmap=cmap, **kwargs)
        return pixel_plotter


class ScienceData(L1Product):
    """
    Basic science data class
    """

    def __init__(self, *, meta, control, data, energies, idb_versions=None):
        """

        Parameters
        ----------
        meta : `astropy.fits.Header`
            Fits header
        control : `astropy.table.QTable`
            Fits file control extension
        data :` astropy.table.QTable`
            Fits file data extension
        energies : `astropy.table.QTable`
            Fits file energy extension
        """
        super().__init__(meta=meta, control=control, data=data, energies=energies, idb_versions=idb_versions)

        self.count_type = "rate"
        if "detector_masks" in self.data.colnames:
            self.detector_masks = DetectorMasks(self.data["detector_masks"])
        if "pixel_masks" in self.data.colnames:
            self.pixel_masks = PixelMasks(self.data["pixel_masks"])
        if "energy_bin_edge_mask" in self.control.colnames:
            self.energy_masks = EnergyEdgeMasks(self.control["energy_bin_edge_mask"])
            self.dE = energies["e_high"] - energies["e_low"]

    @property
    def time_range(self):
        """
        A `sunpy.time.TimeRange` for the data.
        """
        return TimeRange(
            self.data["time"][0] - self.data["timedel"][0] / 2, self.data["time"][-1] + self.data["timedel"][-1] / 2
        )

    @property
    def pixels(self):
        """
        A `stixpy.science.PixelMasks` object representing the pixels contained in the data
        """
        return self.pixel_masks

    @property
    def detectors(self):
        """
        A `stixpy.science.DetectorMasks` object representing the detectors contained in the data.
        """
        return self.detector_masks

    @property
    def energies(self):
        """
        A `astropy.table.Table` object representing the energies contained in the data.
        """
        return self._energies

    @property
    def times(self):
        """
        An `astropy.time.Time` array representing the center of the observed time bins.
        """
        return self.data["time"]

    @property
    @deprecated(name="duration", since="0.2", message="Use `durations` instead", warning_type=DeprecationWarning)
    def duration(self):
        """
        An `astropy.units.Quantity` array giving the duration or integration time
        """
        return self.data["timedel"]

    @property
    def durations(self):
        """
        An `astropy.units.Quantity` array giving the duration or integration time
        """
        return self.data["timedel"]

    @property
    def rcr_raw(self):
        """
        The rcr state
        """
        return self.data["rcr"]

    @property
    def rcr_shifted(self):
        """
        The RCR state of each time bin, aligned to the counts as STIX-GSW does, for the
        settings last given to `align_rcr` (see `rcr_settings`).

        Raises
        ------
        ValueError
            If `align_rcr` has not been called yet.
        """
        if getattr(self, "_rcr_shifted", None) is None:
            raise ValueError(
                "rcr_shifted depends on the detector, pixel, livetime and ELUT "
                "settings. Call align_rcr(...) with the settings you use first."
            )
        return self._rcr_shifted

    @property
    def rcr_settings(self):
        """
        The settings `rcr_shifted` was aligned with, as given to `align_rcr`, or None.
        """
        return getattr(self, "_rcr_settings", None)

    def align_rcr(self, *, detector_indices=None, pixel_indices=None, livetime_correction=True, elut_correction=True):
        """
        Align the RCR state changes to the jumps in the counts as STIX-GSW does, for
        these settings, and keep the result as `rcr_shifted`.

        Parameters
        ----------
        detector_indices, pixel_indices, livetime_correction, elut_correction
            As in `get_data`. Use the same values as for the data you work with.

        Returns
        -------
        numpy.ndarray
            The aligned RCR state of each time bin, also kept as `rcr_shifted`.
        """
        settings = {
            "detector_indices": detector_indices,
            "pixel_indices": pixel_indices,
            "livetime_correction": livetime_correction,
            "elut_correction": elut_correction,
        }
        self._rcr_shifted = np.asarray(self.get_data(vtype="c", **settings).rcr)
        self._rcr_settings = settings
        return self._rcr_shifted

    def get_data(
        self,
        *,
        vtype="dcr",
        time_indices=None,
        energy_indices=None,
        detector_indices=None,
        pixel_indices=None,
        sum_all_times=False,
        sum_detectors=False,
        livetime_correction=True,
        elut_correction=True,
        error_model="legacy",
        ssw_compatible=False,
    ):
        """
        Return the selected data, with optional livetime and ELUT corrections.

        Use `subtract_background` to subtract a background from the result, and
        `stixpy.spectroscopy.to_spectrum` to turn it into `sunkit_spex` spectra.

        Parameters
        ----------
        vtype : {'c', 'cr', 'dcr'}, optional
            Normalisation of the returned counts and uncertainties: counts ('c'),
            count rate in ct/s ('cr'), or differential count rate in ct/(s keV)
            ('dcr', default). With `livetime_correction`, the rates are also divided
            by the livetime fraction.
        time_indices : list, numpy.ndarray or astropy.time.Time, optional
            Flat integer indices keep those time bins, and [start, end] integer
            pairs sum each range. Flat strings or `~astropy.time.Time` select
            the bin containing each instant; [start, end] pairs sum complete bins
            within each range. Indices and times outside
            the file raise a `ValueError`.
        energy_indices : list, numpy.ndarray or astropy.units.Quantity, optional
            Flat integer indices keep those energy bins, and [start, end] pairs sum
            each range. Indices number the rows of the product's energy table
            (`energies`), which only holds the bins in the energy mask. Flat
            Quantity values select the bins containing them; [start, end] ranges,
            such as ``[[4, 10], [15, 28]] * u.keV``, take bins whose centres lie
            inside them.
        detector_indices : list, numpy.ndarray or str, optional
            Flat indices keep those detectors, and [start, end] pairs sum each
            range. Two labels are accepted: "top24", the 24 imaging detectors used
            for spectroscopy in STIX-GSW, and "bkg", the background detector
            (index 9) with its small-aperture pixels [2, 5]. None (default) uses
            every detector in the product. Ignored for spectrogram products.
        pixel_indices : list or numpy.ndarray, optional
            Flat indices keep those pixels, and [start, end] pairs sum each range.
            None (default) uses every pixel in the product, or [2, 5] with
            ``detector_indices="bkg"``. Ignored for spectrogram products.
        sum_all_times : bool, optional
            If True, sum the selected time bins into one. Default False.
        sum_detectors : bool, optional
            If True, sum the selected detectors (or detector ranges) into one, as
            for a spectrum of several detectors. This also averages the ELUT factor
            over all of them. Default False.
        livetime_correction : bool, optional
            Apply the livetime correction. Default True.
        elut_correction : bool, optional
            Apply the ELUT correction to pixel data. Default True. Spectrograms are
            never ELUT corrected, see the Notes.
        error_model : {'legacy', 'correlated'}, optional
            How the uncertainties are combined. 'legacy' (default) combines them as
            stixpy has so far. 'correlated' adds each part as its correlations
            need: the statistical uncertainty in quadrature everywhere, and the
            livetime uncertainty linearly over the pixels and energies of a
            detector and over the two detectors on one trigger counter, and in
            quadrature between trigger counters and time bins. It also gives a time
            range the summed duration of its bins rather than its span. See the
            Notes.


        ssw_compatible : bool, optional
            Use analytic trigger uncertainty propagation and per-pixel quadrature for
            pixel data, and truncate the nominal trigger count per counter for
            spectrograms. Requires ``error_model="legacy"``. Default False.

        Returns
        -------
        SelectedData
            Named tuple with the selected ``counts`` and ``counts_err``
            (normalised according to `vtype`), ``t_norm``, ``e_norm``,
            ``livefrac``, ``livefrac_error``, ``elut_cor_fac``, ``times``,
            ``energies``, ``rcr``, ``vtype`` and the ``selection`` made. Inactive
            detector and pixel positions are masked in both quantities.
            Access the fields by name, as more may be added.

        Raises
        ------
        ValueError
            If a selection is outside the file or the product's energy table, if a
            detector label is not recognised, if ``detector_indices="bkg"`` is given
            with pixels other than [2, 5], if a time range spans an RCR change, if
            `vtype` is not 'c', 'cr' or 'dcr', or if `error_model` is not 'legacy'
            or 'correlated'.

        Notes
        -----
        The 0 keV bottom energy bin and the open top energy bin are never returned.

        With `livetime_correction`, the livetime is corrected as in STIX-GSW: the
        counts of each detector are scaled to the mean livetime fraction of the
        detectors they are combined with (all selected detectors for flat indices,
        each range for [start, end] pairs, every detector that is on for no
        selection), the pixels are summed, so the pixel axis has length 1, and the
        rates (`vtype` 'cr' and 'dcr') are divided by that livetime. With
        ``vtype='c'`` the counts are not divided by the livetime. The livetime
        uncertainty is added to ``counts_err``.

        With `elut_correction`, each energy bin of pixel data is scaled by the
        ratio of its science width to its calibrated (ELUT) width, averaged over
        the selected detectors and pixels that are on (and over all selected
        detectors with `sum_detectors`). Spectrograms are not corrected, as STIX-GSW
        does not: averaged over the detectors and pixels summed on board, the
        calibrated edges are within 0.01 keV of the science edges.

        With ``error_model='legacy'`` the livetime uncertainty is added to
        ``counts_err`` before the detectors and time bins are combined, and then
        summed in quadrature like the statistical uncertainty. This treats it as
        independent between detectors and time bins, though two detectors share
        each trigger counter. Time ranges take their span as their duration, and
        the ``livefrac_error`` returned is combined as an RMS over detectors.

        With ``error_model='correlated'`` the livetime uncertainty is kept apart
        until the end, and added to ``counts_err`` in quadrature only after every
        combination. ``livefrac_error`` then holds the combined livetime
        uncertainty, normalised like the counts.
        """
        if ssw_compatible and error_model == "correlated":
            raise ValueError('ssw_compatible=True requires error_model="legacy".')
        if error_model not in ERROR_MODELS:
            raise ValueError(f"error_model must be one of {ERROR_MODELS}, not {error_model!r}.")

        energy_indices = _energy_indices_format(energy_indices, self.energies)
        if time_indices is not None:
            time_indices = _time_indices_format(time_indices, self.times, self.data["timedel"])

        # _indices_check fills in every detector and pixel that is on when none are given, but get_data has always
        # kept the full axes then, so only take its result for selections that were made
        bkg_label = isinstance(detector_indices, str) and detector_indices.lower() == "bkg"
        checked_detectors, checked_pixels, energy_indices = _indices_check(
            self, detector_indices, pixel_indices, energy_indices
        )
        if detector_indices is not None or sum_detectors:
            detector_indices = checked_detectors  # a sum takes only the detectors that are on
            if np.ndim(detector_indices) == 1:
                detector_indices = list(dict.fromkeys(detector_indices))
        if pixel_indices is not None or bkg_label:
            pixel_indices = checked_pixels

        is_spectrogram = self.data["counts"].ndim < 4
        elut_correction = elut_correction and not is_spectrogram
        selection = DataSelection(
            detector_indices=None if is_spectrogram else detector_indices,
            pixel_indices=None if is_spectrogram else pixel_indices,
            checked_detectors=checked_detectors,
            checked_pixels=checked_pixels,
            detectors=None,
            pixels=None,
            sum_detectors=sum_detectors,
            livetime_correction=livetime_correction,
            elut_correction=elut_correction,
            error_model=error_model,
            ssw_compatible=ssw_compatible,
            spectrogram=is_spectrogram,
            summed_pixels=not is_spectrogram and np.ndim(self.pixel_masks.masks) != 2,
            elut_file=get_elut(self.time_range.center).file if elut_correction else None,
            distance=self.meta["DSUN_OBS"] * u.m if "DSUN_OBS" in self.meta else None,
            time_range=self.time_range,
        )

        data = _selected_counts(self, selection, _energy_positions(energy_indices, self.energies))
        if ssw_compatible and data.livefrac is not None:
            exposure = data.durations[:, None, None, None].to(u.s) * np.nanmean(
                data.livefrac, axis=PIXEL, keepdims=True
            )
            data = replace(data, background_exposure_rss=exposure)
            if selection.spectrogram:
                # IDL adds a second livetime term evaluated on the residual.
                # Keep its quadratic coefficients through the time sum so
                # subtract_background can evaluate it at the background rate.
                counts = data.counts.to_value(u.ct)
                extra = (data.livefrac_err * data.livefrac).to_value(u.ct)
                relative = np.divide(extra, counts, out=np.zeros_like(extra), where=counts != 0)
                slope = exposure.to_value(u.s) * relative
                coefficients = np.stack((extra**2, extra * slope, slope**2), axis=-1)
                data = replace(data, ssw_spectrogram_variance=coefficients)
        data = _select_times(data, time_indices, sum_all_times)
        selection = replace(selection, detectors=data.detectors, pixels=data.pixels.any(axis=0))
        selected = SelectedData.from_count_data(data, vtype, selection)
        active = _selected_active(
            self, detector_indices, pixel_indices, time_indices, sum_all_times, livetime_correction, sum_detectors
        )
        if active is not None and not np.all(active):
            mask = np.broadcast_to(~active[..., np.newaxis], selected.counts.shape)
            selected = selected._replace(
                counts=Masked(selected.counts, mask=mask),
                counts_err=Masked(selected.counts_err, mask=mask),
            )
        return selected

    def concatenate(self, others):
        """
        Concatenate two or more science products.

        Parameters
        ----------
        others: `list` [`stixpy.science.ScienceData`]
            The other/s science products to concatenate

        Returns
        -------
        `stixpy.science.ScienceData`
            The concatenated science products
        """
        others = others if isinstance(others, list) else [others]
        if all(isinstance(o, type(self)) for o in others):
            control = self.control[:]
            data = self.data[:]
            for other in others:
                self_control_ind_max = data["control_index"].max() + 1
                other.control["index"] = other.control["index"] + self_control_ind_max
                other.data["control_index"] = other.data["control_index"] + self_control_ind_max

                try:
                    [
                        (table.meta.pop("DATASUM"), table.meta.pop("CHECKSUM"))
                        for table in [control, other.control, data, other.data]
                    ]
                except KeyError:
                    pass

                control = vstack([control, other.control])
                data = vstack([data, other.data])

            return type(self)(
                meta=self.meta, control=control, data=data, energies=self.energies, idb_version=self.idb_versions
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"{self.time_range}"
            f"    {self.detector_masks}\n"
            f"    {self.pixel_masks}\n"
            f"    {self.energy_masks}"
        )


def _selected_counts(product, selection, energy_positions, restrict=False):
    """
    Load a product and apply a selection and its corrections, before any time selection.

    Parameters
    ----------
    product : ScienceData
        The product to load.
    selection : DataSelection
        The selection; its `~DataSelection.detectors` and `~DataSelection.pixels` are only used with `restrict`.
    energy_positions : numpy.ndarray or None
        Energy selection as positions in the kept energy bins (see `_energy_positions`).
    restrict : bool, optional
        Keep only the detectors and pixels of the selection, for a background that may have more of them on.

    Returns
    -------
    _CountData
    """
    data = _load(product, selection.error_model)
    if restrict:
        data = _restrict(data, *_used(selection))
    if selection.elut_correction:
        factor = _elut_factor(product, selection.checked_detectors, selection.checked_pixels, selection.sum_detectors)
        data = _apply_elut(data, factor)

    livetime_groups = None
    if selection.livetime_correction:
        data = _add_livetime(data, product, selection.checked_pixels, ssw_compatible=selection.ssw_compatible)
        livetime_groups = _livetime_groups(data, selection.detector_indices)

    n_det_spectrogram = None
    if product.data["counts"].ndim < 4:
        n_det_spectrogram = _on_indices(product.detector_masks.masks).size
    data = _select(
        data,
        selection.detector_indices,
        selection.pixel_indices,
        energy_positions,
        livetime_groups=livetime_groups,
        n_detectors=n_det_spectrogram,
        preserve_detector_order=True,
        shared_adc=not selection.ssw_compatible
        and (selection.sum_detectors or np.ndim(selection.detector_indices) == 2),
    )
    if selection.sum_detectors and data.counts.shape[DETECTOR] > 1:
        data = data.combine(DETECTOR, [[0, data.counts.shape[DETECTOR] - 1]])
    return data


def subtract_background(data, background):
    """
    Subtract a background from data selected with `ScienceData.get_data`.

    The selection of `data` is applied to the background product, so that both hold the same detectors, pixels and
    energy bins, with the same corrections. As in STIX-GSW's ``stx_convert_pixel_data`` and
    ``stx_convert_spectrogram``, the livetime corrected background count rate is subtracted from the livetime
    corrected science count rate, so the background is scaled to the duration of each time bin. The uncertainties
    are added in quadrature, and negative results are kept.

    For a spectrogram, the background is summed over the detectors and pixels summed on board into the spectrogram.

    Parameters
    ----------
    data : SelectedData
        Science data from `ScienceData.get_data`, with the livetime correction.
    background : ScienceData
        Background product, e.g. a STIX background (BKG) file: pixel data (sci-xray-rpd or sci-xray-cpd) with a
        single time bin, recorded with the same ELUT as the science data, with every detector and pixel of `data`
        switched on and every energy bin of `data`.

    Returns
    -------
    SelectedData
        `data` with the background subtracted from ``counts`` and its uncertainty added to ``counts_err``.

    Raises
    ------
    TypeError
        If `background` is not a `ScienceData` product.
    ValueError
        If `data` is not livetime corrected or holds pixels summed on board, or if the background does not match
        `data` (see `background`).

    Examples
    --------
    >>> from stixpy.product import Product
    >>> from stixpy.product.sources.science import subtract_background
    >>> cpd = Product("http://dataarchive.stix.i4ds.net/fits/L1/2024/03/10/SCI/"
    ...               "solo_L1_stix-sci-xray-cpd_20240310T115906-20240310T121540_V02_2403109216-57290.fits")  # doctest: +SKIP
    >>> bkg = Product("http://dataarchive.stix.i4ds.net/fits/L1/2024/03/08/SCI/"
    ...               "solo_L1_stix-sci-xray-cpd_20240308T193915-20240308T203235_V02_2403087339-57240.fits")  # doctest: +SKIP
    >>> data = subtract_background(cpd.get_data(detector_indices="top24", vtype="cr"), bkg)  # doctest: +SKIP
    """
    _bkg_check(data, background)

    selection = _bkg_selection(data.selection)
    use_all_background = data.selection.ssw_compatible and data.selection.spectrogram
    edges = np.stack([data.energies["e_low"].to_value(u.keV), data.energies["e_high"].to_value(u.keV)], axis=1)
    energies = _energy_indices_format(edges * u.keV, background.energies)
    bkg = _selected_counts(
        background, selection, _energy_positions(energies, background.energies), restrict=not use_all_background
    )
    bkg = SelectedData.from_count_data(_select_times(bkg, None, True), "cr", selection)
    if bkg.counts.shape[1:] != data.counts.shape[1:]:
        raise ValueError(f"The background has shape {bkg.counts.shape[1:]} but the data {data.counts.shape[1:]}.")

    # the background as a livetime corrected count rate, normalised as the data are
    scale = {"c": data.t_norm * data.livefrac, "cr": 1, "dcr": 1 / data.e_norm}[data.vtype]
    error_scale = scale
    if data.selection.ssw_compatible and data.background_exposure_rss is not None:
        error_scale = scale * (data.background_exposure_rss / (data.t_norm * data.livefrac)).to(u.one)
    variance = data.counts_err**2 + (error_scale * bkg.counts_err) ** 2
    if data.ssw_spectrogram_variance is not None:
        _, cross, quadratic = np.moveaxis(data.ssw_spectrogram_variance, -1, 0)
        rate = bkg.counts.to_value(u.ct / u.s)
        norm = scale / (data.t_norm * data.livefrac)
        variance += (quadratic * rate**2 - 2 * cross * rate) * (u.ct * norm) ** 2
    counts_err = np.sqrt(variance)
    return data._replace(counts=data.counts - scale * bkg.counts, counts_err=counts_err)


class RawPixelData(ScienceData, PixelPlotMixin, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Uncompressed or raw count data from selected pixels, detectors and energies.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.product import Product
    >>> raw_pd = Product("http://dataarchive.stix.i4ds.net/fits/L1/2020/05/05/SCI/"
    ...                  "solo_L1_stix-sci-xray-rpd_20200505T235959-20200506T000019_V02_0087031808-50882.fits")  # doctest: +REMOTE_DATA
    >>> raw_pd  # doctest: +REMOTE_DATA
    RawPixelData   <sunpy.time.timerange.TimeRange object at ...
        Start: 2020-05-05 23:59:59
        End:   2020-05-06 00:00:19
        Center:2020-05-06 00:00:09
    Duration:0.00023148148148144365 days or
               0.005555555555554648 hours or
               0.33333333333327886 minutes or
               19.99999999999673 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0...4]: [['1' '1' '1' '1' '1' '1' '1' '1' '1' '1' '1' '1']]
    <BLANKLINE>
        EnergyEdgeMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    <BLANKLINE>
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 20) and level == "L1":
            return True


class CompressedPixelData(ScienceData, PixelPlotMixin, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Compressed count data from selected pixels, detectors and energies.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.product import Product
    >>> compressed_pd = Product("http://dataarchive.stix.i4ds.net/fits/L1/2020/05/05/SCI/"
    ...                         "solo_L1_stix-sci-xray-cpd_20200505T235959-20200506T000019_V02_0087031809-50883.fits")  # doctest: +REMOTE_DATA
    >>> compressed_pd  # doctest: +REMOTE_DATA
    CompressedPixelData   <sunpy.time.timerange.TimeRange object at ...
        Start: 2020-05-05 23:59:59
        End:   2020-05-06 00:00:19
        Center:2020-05-06 00:00:09
        Duration:0.00023148148148144365 days or
               0.005555555555554648 hours or
               0.33333333333327886 minutes or
               19.99999999999673 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0...4]: [['1' '1' '1' '1' '1' '1' '1' '1' '1' '1' '1' '1']]
    <BLANKLINE>
        EnergyEdgeMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 21) and level == "L1":
            return True


class SummedCompressedPixelData(ScienceData, PixelPlotMixin, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Compressed and Summed count data from selected pixels, detectors and energies.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.product import Product
    >>> summed_pd = Product("http://dataarchive.stix.i4ds.net/fits/L1/2020/05/05/SCI/"
    ...                     "solo_L1_stix-sci-xray-scpd_20200505T235959-20200506T000019_V02_0087031810-50884.fits")  # doctest: +REMOTE_DATA
    >>> summed_pd  # doctest: +REMOTE_DATA
    SummedCompressedPixelData   <sunpy.time.timerange.TimeRange object at ...
        Start: 2020-05-05 23:59:59
        End:   2020-05-06 00:00:19
        Center:2020-05-06 00:00:09
        Duration:0.00023148148148144365 days or
               0.005555555555554648 hours or
               0.33333333333327886 minutes or
               19.99999999999673 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0...4]: [[['0' '0' '0' '1' '0' '0' '0' '1' '0' '0' '0' '1']
     ['0' '0' '1' '0' '0' '0' '1' '0' '0' '0' '1' '0']
     ['0' '1' '0' '0' '0' '1' '0' '0' '0' '1' '0' '0']
     ['1' '0' '0' '0' '1' '0' '0' '0' '1' '0' '0' '0']]]
    <BLANKLINE>
        EnergyEdgeMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    """

    pass

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 22) and level == "L1":
            return True


class Visibility(ScienceData):
    """
    Compressed visibilities from selected pixels, detectors and energies.

    Examples
    --------
    # >>> from stixpy.data import test
    # >>> from stixpy.science import ScienceData
    # >>> visibility = ScienceData.from_fits(test.STIX_SCI_XRAY_VIZ)
    # >>> visibility
    # Visibility   <sunpy.time.timerange.TimeRange object at ...>
    #     Start: 2020-05-07 23:59:58
    #     End:   2020-05-08 00:00:14
    #     Center:2020-05-08 00:00:06
    #     Duration:0.00018518518518517713 days or
    #            0.004444444444444251 hours or
    #            0.26666666666665506 minutes or
    #            15.999999999999304 seconds
    #     DetectorMasks
    #     [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    # <BLANKLINE>
    #     PixelMasks
    #     [0]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    #     [1]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    #     [2]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    #     [3]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    #     [4]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
    #  ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    # <BLANKLINE>
    #     EnergyEdgeMasks
    #     [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    # <BLANKLINE>
    """

    def __init__(self, *, header, control, data, energies):
        super().__init__(meta=header, control=control, data=data, energies=energies)
        self.pixel_masks = PixelMasks(self.pixels)

    @property
    def pixels(self):
        return np.vstack([self.data[f"pixel_mask{i}"][0] for i in range(1, 6)])

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 23) and level == "L1":
            return True


class Spectrogram(ScienceData, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Spectrogram from selected pixels, detectors and energies.

    Parameters
    ----------
    meta : `astropy.fits.Header`
    control : `astropy.table.QTable`
    data : `astropy.table.QTable`
    energies : `astropy.table.QTable`

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.product import Product
    >>> spectogram = Product("http://dataarchive.stix.i4ds.net/fits/L1/2020/05/05/SCI/"
    ...                      "solo_L1_stix-sci-xray-spec_20200505T235959-20200506T000019_V02_0087031812-50886.fits")  # doctest: +REMOTE_DATA
    >>> spectogram  # doctest: +REMOTE_DATA
    Spectrogram   <sunpy.time.timerange.TimeRange ...
        Start: 2020-05-05 23:59:59
        End:   2020-05-06 00:00:19
        Center:2020-05-06 00:00:09
        Duration:0.00023148148148144365 days or
                0.005555555555554648 hours or
                0.33333333333327886 minutes or
                19.99999999999673 seconds
        DetectorMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0...4]: [['0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0']]
    <BLANKLINE>
        EnergyEdgeMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    <BLANKLINE>
    """

    def __init__(self, *, meta, control, data, energies, idb_versions):
        """

        Parameters
        ----------
        meta : astropy.fits.Header
        control : astropy.table.QTable
        data : astropy.table.QTable
        energies : astropy.table.QTable
        """
        super().__init__(meta=meta, control=control, data=data, energies=energies, idb_versions=idb_versions)
        self.count_type = "rate"
        self.detector_masks = DetectorMasks(self.control["detector_masks"])
        self.pixel_masks = PixelMasks(self.data["pixel_masks"])
        # self.energy_masks = EnergyEdgeMasks(self.control['energy_bin_edge_mask'])
        # self.dE = (energies['e_high'] - energies['e_low'])[self.energy_masks.masks[0] == 1]
        # self.dE = np.hstack([[1], np.diff(energies['e_low'][1:]).value, [1]]) * u.keV

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 24) and level == "L1":
            return True


class SliderCustomValue(Slider):
    """
    A slider with a customisable formatter
    """

    def __init__(self, *args, format_func=None, **kwargs):
        if format_func is not None:
            self._format = format_func
        super().__init__(*args, **kwargs)
