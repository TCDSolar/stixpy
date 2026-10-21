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
from astropy.visualization import quantity_support

from sunpy.time.timerange import TimeRange
from sunpy.util import deprecated

from stixpy.calibration.elut import get_elut_correction
from stixpy.io.readers import read_subc_params
from stixpy.product.product import L1Product
from stixpy.product.sources._background import _bkg_check, _bkg_indices_check, _bkg_sub, _energies_bkg_sub
from stixpy.product.sources._corrections import (
    _data_select,
    _elut_correction_sort,
    _indices_expand_ranges,
    _livefrac,
)
from stixpy.product.sources._selection import (
    _energy_indices_format,
    _indices_check,
    _on_indices,
    _time_indices_format,
)
from stixpy.visualisation.plotters import PixelPlotter

__all__ = [
    "ScienceData",
    "SelectedData",
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


class SelectedData(NamedTuple):
    r"""
    The data selected by `ScienceData.get_data`.

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
    """Livetime uncertainty in counts, or None without the livetime correction. Also None with a background
    subtracted, as it is then already included in `counts_err`."""
    elut_cor_fac: np.ndarray | None
    """ELUT correction factor applied to the counts, broadcastable against them, or None without the ELUT
    correction."""
    times: Time
    """Centre of each time bin."""
    energies: QTable
    """Energy table with ``e_low`` and ``e_high`` columns, one row per energy bin."""
    rcr: np.ndarray
    """RCR (attenuator) state of each time bin, aligned to the jumps in the counts as STIX-GSW does."""


def _expand(indices):
    """
    Flat indices of a flat or [start, end] pair selection.
    """
    indices = np.asarray(indices)
    if indices.ndim == 2:
        return np.asarray(_indices_expand_ranges(indices, nest=False), dtype=int)
    return indices.ravel()


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

    def align_rcr(
        self, *, detector_indices=None, pixel_indices=None, livetime_correction=True, elut_correction=True, bkg=None
    ):
        """
        Align the RCR state changes to the jumps in the counts as STIX-GSW does, for
        these settings, and keep the result as `rcr_shifted`.

        Parameters
        ----------
        detector_indices, pixel_indices, livetime_correction, elut_correction, bkg
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
            "bkg": bkg,
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
        livetime_correction=True,
        elut_correction=True,
        bkg=None,
    ):
        """
        Return the selected data, with optional livetime, ELUT and background corrections.

        Parameters
        ----------
        vtype : {'c', 'cr', 'dcr'}, optional
            Normalisation of the returned counts and uncertainties: counts ('c'),
            count rate in ct/s ('cr'), or differential count rate in ct/(s keV)
            ('dcr', default). With `livetime_correction`, the rates are also divided
            by the livetime fraction.
        time_indices : list, numpy.ndarray or astropy.time.Time, optional
            Flat integer indices keep those time bins, and [start, end] integer
            pairs sum each range. Strings or `~astropy.time.Time` are accepted as
            consecutive range edges or as [start, end] pairs, and resolve to the
            data bins that lie wholly inside each range. Indices and times outside
            the file raise a `ValueError`.
        energy_indices : list, numpy.ndarray or astropy.units.Quantity, optional
            Flat integer indices keep those energy bins, and [start, end] pairs sum
            each range. Indices number the rows of the product's energy table
            (`energies`), which only holds the bins in the energy mask. Energies can
            also be given as a Quantity, either flat edges (``[4, 10, 28] * u.keV``)
            or [start, end] ranges (``[[4, 10], [15, 28]] * u.keV``); each range
            takes the bins whose centres lie inside it.
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
        livetime_correction : bool, optional
            Apply the livetime correction. Default True. Must be True when `bkg` is
            given.
        elut_correction : bool, optional
            Apply the ELUT correction. Default True.
        bkg : ScienceData, optional
            Background product to subtract, e.g. a STIX background (BKG) file. It
            must be pixel data (sci-xray-rpd or sci-xray-cpd) with a single time
            bin, every energy bin of this product, and every detector and pixel
            used here switched on. A spectrogram (sci-xray-spec) is subtracted
            with the detectors and pixels summed on board. Default None, no
            background subtraction.

        Returns
        -------
        SelectedData
            Named tuple with the selected ``counts`` and ``counts_err`` (normalised
            according to `vtype`), ``t_norm``, ``e_norm``, ``livefrac``,
            ``livefrac_error``, ``elut_cor_fac``, ``times``, ``energies`` and
            ``rcr``. Access the fields by name, as more may be added.

        Raises
        ------
        TypeError
            If `bkg` is not a `ScienceData` product.
        ValueError
            If a selection is outside the file or the product's energy table, if a
            detector label is not recognised, if ``detector_indices="bkg"`` is given
            with pixels other than [2, 5], if a time range spans an RCR change, or
            if `vtype` is not 'c', 'cr' or 'dcr'. With `bkg`, also if
            `livetime_correction` is False, or the background does not match this
            product (see `bkg`).

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

        With `elut_correction`, each energy bin is scaled by the ratio of its
        science width to its calibrated (ELUT) width, averaged over the selected
        detectors and pixels that are on, so one factor per energy bin is applied
        to every detector.

        With `bkg`, as in STIX-GSW's ``stx_convert_pixel_data`` and
        ``stx_convert_spectrogram``, the science and background counts are both
        livetime corrected, and ELUT corrected with this product's factor. The
        background is converted to a count rate over its time bin, scaled to the
        duration of each science time bin and subtracted, and the result is scaled
        back by the mean livetime fraction of the detectors combined into each
        spectrum. The background uncertainty and both livetime uncertainties are
        added to ``counts_err`` in quadrature, so ``livefrac_error`` is None.
        Counts whose total over the detectors and pixels is negative are set to
        zero. The pixels are summed, as with the livetime correction alone.
        """
        energy_indices = _energy_indices_format(energy_indices, self.energies)
        if time_indices is not None:
            time_indices = _time_indices_format(time_indices, self.times, self.data["timedel"])

        # _indices_check fills in every detector and pixel that is on when none are given, but get_data has always
        # kept the full axes then, so only take its result for selections that were made
        bkg_label = isinstance(detector_indices, str) and detector_indices.lower() == "bkg"
        checked_detectors, checked_pixels, energy_indices = _indices_check(
            self, detector_indices, pixel_indices, energy_indices
        )
        if detector_indices is not None:
            detector_indices = checked_detectors
        if pixel_indices is not None or bkg_label:
            pixel_indices = checked_pixels

        is_spectrogram = self.data["counts"].ndim < 4
        n_det_spectrogram = (
            int(np.count_nonzero(np.asarray(self.detector_masks.masks, dtype=bool).any(axis=0)))
            if is_spectrogram
            else None
        )

        detector_masks = getattr(self, "detector_masks", None)
        pixel_masks = getattr(self, "pixel_masks", None)
        counts = _full_layout(self.data["counts"], detector_masks, pixel_masks, name="counts")
        err_name = "counts_comp_err" if "counts_comp_err" in self.data.colnames else "counts_comp_comp_err"
        counts_err = _full_layout(self.data[err_name], detector_masks, pixel_masks, name=err_name)
        if is_spectrogram:
            shape = counts.shape
            counts = counts.reshape(shape[0], 1, 1, shape[-1])
            counts_err = counts_err.reshape(shape[0], 1, 1, shape[-1])

        if elut_correction:
            _, _, bins, bins_actual = get_elut_correction(np.array(self.energies["channel"]), self)

            if is_spectrogram:
                detector_indices_elut = _on_indices(self.detector_masks.masks)
                pixel_indices_elut = _on_indices(self.pixel_masks.masks)
            else:
                # the selected detectors and pixels that are on, as [start, end] ranges may span ones that are off
                detector_indices_elut = np.intersect1d(
                    _expand(checked_detectors), _on_indices(self.detector_masks.masks)
                )
                pixel_indices_elut = np.intersect1d(_expand(checked_pixels), _on_indices(self.pixel_masks.masks))

            elut_cor_fac = _elut_correction_sort(
                bins, bins_actual, True, pixel_indices_elut, detector_indices_elut, is_spectrogram
            )

        else:
            elut_cor_fac = None

        if bkg is not None:
            _bkg_check(self, bkg, checked_detectors, checked_pixels, livetime_correction)

            energy_indices_bkg = _energies_bkg_sub(self, bkg)

            pixel_indices_bkg, detector_indices_bkg = _bkg_indices_check(self, bkg)

        if livetime_correction:
            livefrac, livefrac_error = _livefrac(self, counts, elut_cor_fac, checked_pixels)

            if bkg is not None:
                counts_bkg = _full_layout(bkg.data["counts"], bkg.detector_masks, bkg.pixel_masks, name="counts")
                err_name_bkg = "counts_comp_err" if "counts_comp_err" in bkg.data.colnames else "counts_comp_comp_err"
                counts_err_bkg = _full_layout(
                    bkg.data[err_name_bkg], bkg.detector_masks, bkg.pixel_masks, name=err_name_bkg
                )
                livefrac_bkg, livefrac_error_bkg = _livefrac(
                    bkg, counts_bkg, elut_cor_fac, pixel_indices_bkg, energy_indices_bkg
                )

        else:
            livefrac = None
            livefrac_error = None

        if bkg is None:
            sci_data = _data_select(
                self,
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
                n_detectors=n_det_spectrogram,
            )

        else:
            sci_data_all = _bkg_sub(
                self,
                bkg,
                counts,
                counts_err,
                counts_bkg,
                counts_err_bkg,
                checked_detectors,
                checked_pixels,
                detector_indices_bkg,
                pixel_indices_bkg,
                energy_indices_bkg,
                livefrac,
                livefrac_error,
                livefrac_bkg,
                livefrac_error_bkg,
                elut_cor_fac,
                np.asarray(self.data["rcr"]),
            )
            counts_sub, counts_var_sub, _, _, livefrac_sub, _, elut_cor_fac_sub, _, _, _ = sci_data_all

            sci_data = _data_select(
                self,
                counts_sub,
                counts_var_sub,
                detector_indices,
                pixel_indices,
                energy_indices,
                time_indices,
                livefrac_sub,
                None,
                elut_cor_fac_sub,
                sum_all_times,
                n_detectors=n_det_spectrogram,
                bkg=True,
            )

        counts, counts_err, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr = sci_data

        e_norm = e_norm[np.newaxis, np.newaxis, np.newaxis, :]
        t_norm = t_norm[:, np.newaxis, np.newaxis, np.newaxis].to(u.s)

        if livetime_correction:
            livefrac = np.nanmean(livefrac, axis=2, keepdims=True)

        if vtype == "c":
            norm = 1

        elif vtype == "cr":
            norm = 1 / t_norm
            if livetime_correction:
                norm = 1 / (t_norm * livefrac)

        elif vtype == "dcr":
            norm = 1 / (e_norm * t_norm)

            if livetime_correction:
                norm = 1 / (e_norm * t_norm * livefrac)

        else:
            raise ValueError("vtype must be one of 'c', 'cr', 'dcr'.")

        return SelectedData(
            counts=counts * norm,
            counts_err=counts_err * norm,
            t_norm=t_norm,
            e_norm=e_norm,
            livefrac=livefrac,
            livefrac_error=livefrac_error,
            elut_cor_fac=elut_cor_fac,
            times=times,
            energies=energies,
            rcr=rcr,
        )

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
