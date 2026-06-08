from pathlib import Path
from functools import cached_property
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import ConciseDateFormatter, DateFormatter, HourLocator
from matplotlib.widgets import Slider

import astropy.units as u
from astropy.table import vstack
from astropy.time import Time
from astropy.visualization import quantity_support

from sunpy.time.timerange import TimeRange
from sunpy.util import deprecated

from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.config.instrument import STIX_INSTRUMENT
from stixpy.io.readers import read_subc_params
from stixpy.product.product import L1Product

__all__ = [
    "ScienceData",
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

from stixpy.visualisation.plotters import PixelPlotter

quantity_support()


SubCollimatorConfig = read_subc_params(
    Path(__file__).parent.parent.parent / "config" / "data" / "detector" / "stx_subc_params.csv"
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

        Parameters
        ----------
        axes : optional `matplotlib.axes`
            The axes the plot the spectrogram.
        vtype : str
           Type of value to return control the default normalisation:
               * 'c' - count [c]
               * 'cr' - count rate [c/s]
               * 'dcr' - differential count rate [c/(s keV)]
        time_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        pixel_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `pixel_indices=[0, 2, 5]` would return only the first, third and
            sixth pixels while `pixel_indices=[[0, 2],[3, 5]]` would sum the data between.
        detector_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `detector_indices=[0, 2, 5]` would return only the first, third and
            sixth detectors while `detector_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
        **plot_kwargs : `dict`
            Any additional arguments are passed to :meth:`~matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        `matplotlib.axes`

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
            if detector_indices == "all":
                did = [[0, 31]]
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

            if pixel_indices == "all":
                pid = [[0, 11]]
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

        counts, errors, times, timedeltas, energies = self.get_data(
            vtype=vtype,
            detector_indices=did,
            pixel_indices=pid,
            time_indices=time_indices,
            energy_indices=energy_indices,
        )
        timedeltas = timedeltas.to(u.s)

        e_edges = np.hstack([energies["e_low"], energies["e_high"][-1]]).value
        t_edges = Time(
            np.concatenate([times - timedeltas.reshape(-1) / 2, times[-1] + timedeltas.reshape(-1)[-1:] / 2])
        )

        pcolor_kwargs = {"norm": LogNorm(), "shading": "flat"}
        pcolor_kwargs.update(plot_kwargs)
        im = axes.pcolormesh(t_edges.datetime, e_edges[1:-1], counts[:, 0, 0, 1:-1].T.value, **pcolor_kwargs)  # noqa

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

        if detector_indices == "all":
            detector_indices = [[0, 31]]

        if pixel_indices == "all":
            pixel_indices = [[0, 11]]

        counts, errors, times, timedeltas, energies = self.get_data(
            vtype=vtype,
            detector_indices=detector_indices,
            pixel_indices=pixel_indices,
            time_indices=time_indices,
            energy_indices=energy_indices,
        )

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
    def time(self):
        """
        An `astropy.time.Time` array representing the center of the observed time bins.
        """
        return self.data["time"]

    @property
    @deprecated(name="times", since="0.2", message="Use `time` instead", warning_type=DeprecationWarning)
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
        An `astropy.units.Quantity` array giving the duration or integration time.

        Shape is ``(T, 1, 1, 1)`` to broadcast against ``(T, n_det, n_pix, n_energy)``.
        """
        return self.data["timedel"][:, None, None, None]

    @cached_property
    def livetime_fraction(self):
        """
        Livetime fraction.

        Shape is ``(T, n_det, 1, 1)`` to broadcast against ``(T, n_det, n_pix, n_energy)``.
        """
        if self.data["triggers"].ndim == 1:
            triggers = self.data["triggers"].reshape(-1, 1).astype(float)
        else:
            trigger_to_detector = STIX_INSTRUMENT.subcol_adc_mapping
            triggers = self.data["triggers"][:, trigger_to_detector].astype(float)
        livefrac, *_ = get_livetime_fraction(triggers / self.data["timedel"].to("s").reshape(-1, 1))
        return livefrac[:, :, None, None]

    @cached_property
    def livetime(self):
        """
        Livetime.

        Shape is ``(T, n_det, 1, 1)`` to broadcast against ``(T, n_det, n_pix, n_energy)``.
        """
        return self.livetime_fraction * self.data["timedel"].to("s")[:, None, None, None]

    @property
    def pixel_area(self):
        """
        Pixel areas.

        Shape is ``(1, 1, n_pix, 1)`` (or ``(T, 1, n_pix, 1)`` when pixel masks vary per time)
        to broadcast against ``(T, n_det, n_pix, n_energy)``.
        """
        from stixpy.config.instrument import STIX_INSTRUMENT

        base_areas = STIX_INSTRUMENT.pixel_config["Area"].to("cm2")  # (n_pix,)
        if "pixel_masks" in self.data.colnames:
            mask = self.pixel_masks.masks[0].astype(bool)  # (12,)
            return base_areas[mask][None, None, :, None]  # (1, 1, n_active_pix, 1)
        return base_areas[None, None, :, None]  # (1, 1, n_pix, 1)

    @cached_property
    def elut(self):
        """
        ELUT correction
        """
        from types import SimpleNamespace

        from stixpy.calibration.energy import get_elut

        energy_mask = self.energy_masks.energy_mask.astype(bool)
        elut = get_elut(self.time_range.center)
        ebin_edges_low = np.zeros((32, 12, 32), dtype=float)
        ebin_edges_low[..., 1:] = elut.e_actual
        ebin_edges_low = ebin_edges_low[..., energy_mask]
        ebin_edges_high = np.zeros((32, 12, 32), dtype=float)
        ebin_edges_high[..., 0:-1] = elut.e_actual
        ebin_edges_high[..., -1] = np.nan
        ebin_edges_high = ebin_edges_high[..., energy_mask]
        ebin_widths = ebin_edges_high - ebin_edges_low
        ebin_sci_edges_low = elut.e[..., 0:-1].value[..., energy_mask]
        ebin_sci_edges_high = elut.e[..., 1:].value[..., energy_mask]
        return SimpleNamespace(
            ebin_edges_low=ebin_edges_low,
            ebin_edges_high=ebin_edges_high,
            ebin_widths=ebin_widths,
            ebin_sci_edges_low=ebin_sci_edges_low,
            ebin_sci_edges_high=ebin_sci_edges_high,
        )

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
        elut_correction=False,
    ):
        r"""
        Return the counts, errors, times, durations and energies for selected data.

        Optionally summing in time and or energy.

        Parameters
        ----------
        vtype : str
            Type of value to return (vtype) controls the normalisation:
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
        sum_all_times : `bool`
            Flag to sum all give time intervals into one
        livetime_correction : `bool`
            If `True`, divide counts by the per-detector livetime before applying any vtype normalisation.
            Applied after detector selection so the correct per-detector fractions are used before any
            summing over detectors.
        elut_correction : `bool`
            If `True` (default), scale the boundary energy bins by the ELUT fractional-coverage
            correction, matching the correction applied in `create_meta_pixels`. Applied within
            the energy selection step so the true boundary bins of the selection are corrected.
            Only active for full pixel data (32 detectors, 12 pixels) with ``energy_masks`` available.

        Returns
        -------
        `tuple`
            Counts, errors, times, deltatimes,  energies

        """
        sum_times, sum_energies, sum_detectors, sum_pixels = [
            _is_sum(index) for index in [time_indices, energy_indices, detector_indices, pixel_indices]
        ]

        counts = self.data["counts"]
        try:
            counts_comp_var = self.data["counts_comp_err"] ** 2
        except KeyError:
            counts_comp_var = self.data["counts_comp_comp_err"] ** 2
        shape = counts.shape
        if len(shape) < 4:
            counts = counts.reshape(shape[0], 1, 1, shape[-1])
            counts_comp_var = counts_comp_var.reshape(shape[0], 1, 1, shape[-1])

        counts_stat_var = counts.copy()

        if livetime_correction:
            # Correct each detector's counts independently before any summation so that
            # detectors with different livetime fractions are brought to a common effective exposure.
            lf = self.livetime_fraction  # (T, n_det, 1, 1), dimensionless
            counts = counts / lf
            counts_stat_var = counts_stat_var / lf**2
            counts_comp_var = counts_comp_var / lf**2

        elut = self.elut
        det_mask = self.detector_masks.masks[0].astype(bool)
        pix_mask = self.pixel_masks.masks[0].astype(bool)

        # Default edges — all bins, shape (1, n_det, n_pix, n_energy, 2), last axis is [lo, hi]
        actual_edges = np.stack([elut.ebin_edges_low, elut.ebin_edges_high], axis=-1)  # (32, 12, n_energy, 2)
        actual_edges = actual_edges[np.ix_(det_mask, pix_mask)][None]  # (1, n_det, n_pix, n_energy, 2)
        sci_edges = np.stack([elut.ebin_sci_edges_low, elut.ebin_sci_edges_high], axis=-1)[
            None, None, None
        ]  # (1, 1, 1, n_energy, 2)
        dE = actual_edges[..., 1] - actual_edges[..., 0]  # (1, n_det, n_pix, n_energy)

        time = self.time
        area = self.pixel_area
        duration = self.durations
        # After livetime correction all detectors share the same effective livetime = duration
        livetime = duration if livetime_correction else self.livetime

        # Normalise elut_correction: True → 'overlap'
        if elut_correction is True:
            elut_correction = "overlap"

        if elut_correction == "overlap" and energy_indices is not None:
            det_lo = elut.ebin_edges_low[..., None]  # (32, 12, n_energy, 1)
            det_hi = elut.ebin_edges_high[..., None]
            sci_lo = elut.ebin_sci_edges_low[None, None, None, :]  # (1, 1, 1, n_sci)
            sci_hi = elut.ebin_sci_edges_high[None, None, None, :]
            overlap = np.maximum(0.0, np.minimum(det_hi, sci_hi) - np.maximum(det_lo, sci_lo))
            W = (overlap / (det_hi - det_lo))[np.ix_(det_mask, pix_mask)]  # (n_det, n_pix, n_det_bins, n_sci_bins)

        if energy_indices is not None and sum_energies is False:
            energy_mask = np.full(counts.shape[-1], False)
            energy_mask[energy_indices] = True
            sel_indices = np.where(energy_mask)[0]
            if elut_correction is False:
                counts = counts[..., energy_mask]
                counts_comp_var = counts_comp_var[..., energy_mask]
                counts_stat_var = counts_stat_var[..., energy_mask]
                actual_edges = np.stack(
                    [elut.ebin_edges_low[..., sel_indices], elut.ebin_edges_high[..., sel_indices]], axis=-1
                )  # (32, 12, n_sel, 2)
                actual_edges = actual_edges[np.ix_(det_mask, pix_mask)][None]  # (1, n_det, n_pix, n_sel, 2)
                sci_edges = np.stack(
                    [elut.ebin_sci_edges_low[sel_indices], elut.ebin_sci_edges_high[sel_indices]], axis=-1
                )[None, None, None]
            elif elut_correction == "overlap":
                W_sel = W[..., energy_mask]
                counts = np.einsum("tdpj,dpjk->tdpk", counts, W_sel)
                counts_stat_var = np.einsum("tdpj,dpjk->tdpk", counts_stat_var, W_sel**2)
                counts_comp_var = np.einsum("tdpj,dpjk->tdpk", counts_comp_var, W_sel**2)
                sci_edges = np.stack(
                    [elut.ebin_sci_edges_low[sel_indices], elut.ebin_sci_edges_high[sel_indices]], axis=-1
                )[None, None, None]
                actual_edges = sci_edges
            elif elut_correction == "edge_weight":
                # IDL single-bin formula: counts[k] *= sci_width[k] / det_width[k]
                sci_widths = (elut.ebin_sci_edges_high - elut.ebin_sci_edges_low)[sel_indices]  # (n_sel,)
                det_widths = elut.ebin_widths[np.ix_(det_mask, pix_mask)][..., sel_indices][
                    None
                ]  # (1, n_det, n_pix, n_sel)
                corr = sci_widths / det_widths  # (1, n_det, n_pix, n_sel)
                counts = counts[..., energy_mask] * corr
                counts_stat_var = counts_stat_var[..., energy_mask] * corr**2
                counts_comp_var = counts_comp_var[..., energy_mask] * corr**2
                sci_edges = np.stack(
                    [elut.ebin_sci_edges_low[sel_indices], elut.ebin_sci_edges_high[sel_indices]], axis=-1
                )[None, None, None]
                actual_edges = sci_edges
            dE = actual_edges[..., 1] - actual_edges[..., 0]

        elif sum_energies:
            energy_indices = np.array(energy_indices)
            low_idx = energy_indices[..., 0]
            high_idx = energy_indices[..., -1]
            if elut_correction is False:
                ebin_idx = np.arange(counts.shape[-1])
                mask = (ebin_idx[:, None] >= low_idx) & (ebin_idx[:, None] <= high_idx)
                counts = (counts[..., None] * mask).sum(axis=-2)
                counts_comp_var = (counts_comp_var[..., None] * mask).sum(axis=-2)
                counts_stat_var = (counts_stat_var[..., None] * mask).sum(axis=-2)
                actual_edges = np.stack(
                    [elut.ebin_edges_low[..., low_idx], elut.ebin_edges_high[..., high_idx]], axis=-1
                )  # (32, 12, n_ranges, 2)
                actual_edges = actual_edges[np.ix_(det_mask, pix_mask)][None]
                sci_edges = np.stack([elut.ebin_sci_edges_low[low_idx], elut.ebin_sci_edges_high[high_idx]], axis=-1)[
                    None, None, None
                ]
            elif elut_correction == "overlap":
                counts_list, comp_var_list, stat_var_list = [], [], []
                for lo, hi in zip(low_idx, high_idx):
                    W_range = W[..., lo : hi + 1]
                    counts_list.append(np.einsum("tdpj,dpjk->tdpk", counts, W_range).sum(axis=-1, keepdims=True))
                    comp_var_list.append(
                        np.einsum("tdpj,dpjk->tdpk", counts_comp_var, W_range**2).sum(axis=-1, keepdims=True)
                    )
                    stat_var_list.append(
                        np.einsum("tdpj,dpjk->tdpk", counts_stat_var, W_range**2).sum(axis=-1, keepdims=True)
                    )
                counts = np.concatenate(counts_list, axis=-1)
                counts_comp_var = np.concatenate(comp_var_list, axis=-1)
                counts_stat_var = np.concatenate(stat_var_list, axis=-1)
                sci_edges = np.stack([elut.ebin_sci_edges_low[low_idx], elut.ebin_sci_edges_high[high_idx]], axis=-1)[
                    None, None, None
                ]
                actual_edges = sci_edges
            elif elut_correction == "edge_weight":
                # IDL multi-bin formula: scale boundary bins by edge-overlap fraction then sum.
                # When lo==hi both corrections multiply the same bin (replicates IDL behaviour).
                ew_edges_high = elut.ebin_edges_high[np.ix_(det_mask, pix_mask)]  # (n_det, n_pix, n_energy)
                ew_edges_low = elut.ebin_edges_low[np.ix_(det_mask, pix_mask)]
                ew_widths = elut.ebin_widths[np.ix_(det_mask, pix_mask)]
                counts_list, comp_var_list, stat_var_list = [], [], []
                for lo, hi in zip(low_idx, high_idx):
                    sci_lo_edge = elut.ebin_sci_edges_low[lo]
                    sci_hi_edge = elut.ebin_sci_edges_high[hi]
                    w_low = (ew_edges_high[..., lo] - sci_lo_edge) / ew_widths[..., lo]  # (n_det, n_pix)
                    w_high = (sci_hi_edge - ew_edges_low[..., hi]) / ew_widths[..., hi]  # (n_det, n_pix)
                    # Build per-bin weight array (1, n_det, n_pix, n_bins)
                    # When lo==hi, both w_low and w_high multiply the same bin → replicates IDL behaviour
                    n_bins = hi - lo + 1
                    w = np.ones((1,) + w_low.shape + (n_bins,), dtype=float)
                    w[..., 0] *= w_low.unmasked  # (32, 8) broadcasts against (1, 32, 8, n_bins)[..., 0]
                    w[..., -1] *= w_high.unmasked
                    c = np.asarray(counts[..., lo : hi + 1], dtype=float) * w
                    vs = np.asarray(counts_stat_var[..., lo : hi + 1], dtype=float) * w**2
                    vc = np.asarray(counts_comp_var[..., lo : hi + 1], dtype=float) * w**2
                    counts_list.append(c.sum(axis=-1, keepdims=True))
                    stat_var_list.append(vs.sum(axis=-1, keepdims=True))
                    comp_var_list.append(vc.sum(axis=-1, keepdims=True))
                counts = np.concatenate(counts_list, axis=-1)
                counts_comp_var = np.concatenate(comp_var_list, axis=-1)
                counts_stat_var = np.concatenate(stat_var_list, axis=-1)
                sci_edges = np.stack([elut.ebin_sci_edges_low[low_idx], elut.ebin_sci_edges_high[high_idx]], axis=-1)[
                    None, None, None
                ]
                actual_edges = sci_edges
            dE = actual_edges[..., 1] - actual_edges[..., 0]

        # Pixel slicing or summation
        if pixel_indices is not None and sum_pixels is False:
            pixel_mask = np.full(12, False)
            pixel_mask[pixel_indices] = True
            num_pixels = counts.shape[2]
            counts = counts[..., pixel_mask[:num_pixels], :]
            counts_comp_var = counts_comp_var[..., pixel_mask[:num_pixels], :]
            counts_stat_var = counts_stat_var[..., pixel_mask[:num_pixels], :]
            area = area[..., pixel_mask[:num_pixels], :]
        elif sum_pixels:
            counts = np.concatenate(
                [np.sum(counts[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices], axis=2
            )
            counts_comp_var = np.concatenate(
                [np.sum(counts_comp_var[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices],
                axis=2,
            )
            counts_stat_var = np.concatenate(
                [np.sum(counts_stat_var[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices],
                axis=2,
            )
            area = np.concatenate(
                [np.sum(area[..., pl : ph + 1, :], axis=-2, keepdims=True) for pl, ph in pixel_indices], axis=-2
            )

        # Detector slicing or summation
        if detector_indices and sum_detectors is False:
            detector_mask = np.full(32, False)
            detector_mask[detector_indices] = True
            counts = counts[:, detector_mask, ...]
            counts_comp_var = counts_comp_var[:, detector_mask, ...]
            counts_stat_var = counts_stat_var[:, detector_mask, ...]
            livetime = livetime[:, detector_mask, ...]
        elif sum_detectors:
            counts = np.concatenate(
                [np.sum(counts[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices], axis=1
            )
            counts_comp_var = np.concatenate(
                [np.sum(counts_comp_var[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices],
                axis=1,
            )
            counts_stat_var = np.concatenate(
                [np.sum(counts_stat_var[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices],
                axis=1,
            )
            livetime = np.concatenate(
                [np.mean(livetime[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices], axis=1
            )

        # Time slicing or summation
        if time_indices is not None and sum_times is False:
            time_mask = np.full(time.shape, False)
            time_mask[time_indices] = True
            counts = counts[time_mask, ...]
            counts_comp_var = counts_comp_var[time_mask, ...]
            counts_stat_var = counts_stat_var[time_mask, ...]
            time = time[time_mask]
            duration = duration[time_mask]
            livetime = livetime[time_mask]
        elif sum_times:
            new_time_centers = []
            for tl, th in time_indices:
                ts = time[tl] - duration[tl, 0, 0, 0] * 0.5
                te = time[th] + duration[th, 0, 0, 0] * 0.5
                new_time_centers.append(ts + (te - ts) * 0.5)

            time = Time(new_time_centers)
            duration = np.concatenate(
                [np.sum(duration[tl : th + 1], axis=0, keepdims=True) for tl, th in time_indices], axis=0
            )
            livetime = np.concatenate(
                [np.sum(livetime[tl : th + 1], axis=0, keepdims=True) for tl, th in time_indices], axis=0
            )
            counts = np.concatenate(
                [np.sum(counts[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices], axis=0
            )
            counts_comp_var = np.concatenate(
                [np.sum(counts_comp_var[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices], axis=0
            )
            counts_stat_var = np.concatenate(
                [np.sum(counts_stat_var[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices], axis=0
            )

        return (
            counts,
            counts_comp_var,
            counts_stat_var,
            actual_edges,
            sci_edges,
            dE,
            time,
            duration,
            livetime,
            area,
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
        if all([isinstance(o, type(self)) for o in others]):
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


def _is_sum(indices):
    if indices is None:
        return False

    arr = np.asarray(indices)
    if arr.ndim == 2:
        return True

    return False


def _validate_intervals(indices, name, size):
    if indices is None:
        return None

    arr = np.asarray(indices)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} expected (n,2) intervals")

    if np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError(f"{name} invalid interval: start must be < end")

    if np.any(arr < 0) or np.any(arr > size):
        raise ValueError("Interval out of bounds")


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
