from pathlib import Path
from itertools import product
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import ConciseDateFormatter, DateFormatter, HourLocator
from matplotlib.widgets import Slider

from ndcube import NDMeta
from ndcube import NDCubeSequence, NDCollection

from sunkit_spex.spectrum.spectrum import SpectralAxis, Spectrum
from sunkit_spex.spectrum.uncertainty import PoissonUncertainty

import astropy.units as u
from astropy.table import QTable, vstack, Table
from astropy.time import Time
from astropy.visualization import quantity_support

from sunpy.time.timerange import TimeRange
from sunpy.util import deprecated
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

from stixpy.calibration.elut import get_elut_correction
from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.calibration.transmission import Transmission
from stixpy.config.instrument import STIX_INSTRUMENT
from stixpy.io.readers import read_subc_params
from stixpy.product.product import L1Product
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.coordinates.flare_angle import flare_spacecraft_angle

# from stixpy.calibration.flare_location import estimate_flare_location

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
    "calc_count_rate",
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

        counts, errors, timedeltas, _, _, _, _, times, energies, _ = self.get_data(
            vtype=vtype,
            detector_indices=did,
            pixel_indices=pid,
            time_indices=time_indices,
            energy_indices=energy_indices,
            livetime_correction=False,
            elut_correction=False
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


        counts, errors, timedeltas, _, _, _, _, times, energies, _ = self.get_data(
            vtype=vtype,
            detector_indices=detector_indices,
            pixel_indices=pixel_indices,
            time_indices=time_indices,
            energy_indices=energy_indices,
            livetime_correction=False,
            elut_correction=False
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
    def rcr_shifted(self):
        """
        The rcr state
        """ 
        return ScienceData._rcr_shift(self.data["rcr"],self.data["counts"])


    @property
    def rcr_raw(self):
        """
        The rcr state
        """
        return self.data["rcr"]


    @staticmethod
    def _indices_check(product, detector_indices, pixel_indices):
        """
        Validate and normalize the requested detector and pixel indices against what is
        actually available in the product's masks.

        If `detector_indices` is None, all detectors available in `product.detector_masks`
        are used. The special string "top24" selects a fixed set of 24 detector indices
        excluding the background/CFL detectors. Otherwise, indices may be given as either
        a flat 1D list of individual indices or a 2D list of [start, end] range pairs; in
        both cases a warning is raised if any requested index/range is not present in the
        product. The same logic applies independently to `pixel_indices` using
        `product.pixel_masks`. If `sunkit_spex_spectrum=True`, both `detector_indices` and
        `pixel_indices` (if not None) must be 1D, otherwise a ValueError is raised.

        Parameters
        ----------
        product : ScienceData
            The data product whose `detector_masks` and `pixel_masks` are used to
            determine which indices are actually available.
        detector_indices : list, numpy.ndarray, str, or None
            Requested detector indices (flat list, [start, end] pairs, "top24", or None
            to use all available detectors).
        pixel_indices : list, numpy.ndarray, or None
            Requested pixel indices (flat list, [start, end] pairs, or None to use all
            available pixels).
        sunkit_spex_spectrum : bool
            If True, enforces that `detector_indices` and `pixel_indices` are 1D.

        Returns
        -------
        tuple of numpy.ndarray
            The validated `detector_indices` and `pixel_indices` as numpy arrays.
        """

        # --- Detector indices ---

        if detector_indices is not None:

            if len(product.data['counts'].shape) < 4:

                warnings.warn(f"As a spectrogram file is being used, the user selected detector indices \
                                {detector_indices} will not be used, defaulting to the indices used in the creation \
                                of the spectrgram file.")

                detector_indices = None
            
            else:

                detector_indices_working = detector_indices

                if detector_indices_working == "top24":
                    detector_indices_working = np.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
                    detector_indices = detector_indices_working
                else:
                    detector_indices_full = np.where(product.detector_masks.__dict__["masks"] == 1)[1]

                    if np.ndim(detector_indices_working) == 2:
                        # [[start, end], ...] range format
                        for start, end in detector_indices_working:
                            requested = np.arange(start, end + 1)
                            missing = np.setdiff1d(requested, detector_indices_full)
                            if missing.size > 0:
                                warnings.warn(f"Detector indices {missing.tolist()} in range [{start}, {end}] are not available in the product.")
                    else:
                        missing = np.setdiff1d(detector_indices_working, detector_indices_full)
                        if missing.size > 0:
                            warnings.warn(f"The following detector indices are not available in the product: {missing.tolist()}")
                
        else:

            if len(product.data['counts'].shape) < 4:
                detector_indices = None
            else:
                detector_indices = np.where(product.detector_masks.__dict__["masks"] == 1)[1]

        # --- Pixel indices ---
        if pixel_indices is not None:
            pixel_indices_full = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]

            if len(product.data['counts'].shape) < 4:

                warnings.warn(f"As a spectrogram file is being used, the user selected detector indices \
                                {pixel_indices} will not be used, defaulting to the indices used in the creation \
                                of the spectrgram file.")
                pixel_indices = None
            
            else:
                    
                if np.ndim(pixel_indices) == 2:
                    for start, end in pixel_indices:
                        requested = np.arange(start, end + 1)
                        missing = np.setdiff1d(requested, pixel_indices_full)
                        if missing.size > 0:
                            warnings.warn(f"Pixel indices {missing.tolist()} in range [{start}, {end}] are not available in the product.")
                else:
                    missing = np.setdiff1d(pixel_indices, pixel_indices_full)
                    if missing.size > 0:
                        warnings.warn(f"The following pixel indices are not available in the product: {missing.tolist()}")

        else:

            if len(product.data['counts'].shape) < 4:
                pixel_indices = None
            else:
                pixel_indices = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]


        return np.array(detector_indices), np.array(pixel_indices)

    @staticmethod
    def _livetime_uncertainty(counts_var, livefrac_error):
        """
        Combine count variance with livetime-fraction error, if available.

        If `livefrac_error` is provided, propagates its uncertainty into the count
        variance in quadrature. Otherwise, the count variance is returned unchanged.

        Parameters
        ----------
        counts_var : astropy.units.Quantity
            Variance (or uncertainty) on the counts.
        livefrac : astropy.units.Quantity or None
            Livetime fraction. Unused directly in this function but kept for
            signature consistency with callers.
        livefrac_error : astropy.units.Quantity or None
            Uncertainty on the livetime fraction. If None, no correction is applied.

        Returns
        -------
        astropy.units.Quantity
            Count variance, optionally combined in quadrature with the livetime
            fraction error, in units of counts.
        """

        if livefrac_error is not None:
        
            counts_var_lvtcorr = np.sqrt(((counts_var**2).value) + (livefrac_error.value**2))

            return counts_var_lvtcorr * u.ct
        
        else:

            return counts_var

    @staticmethod        
    def _apply_livetime(counts, counts_var, livefrac, groups):
        counts_corr = counts / livefrac
        counts_var_corr = counts_var / livefrac
        counts_out = counts.copy()
        counts_var_out = counts_var.copy()
        new_livefrac = livefrac.copy()
        for g in groups:
            g = np.atleast_1d(np.asarray(g))
            num = np.nansum(counts[:, g, :, :], axis=(1, 2, 3), keepdims=True)
            den = np.nansum(counts_corr[:, g, :, :], axis=(1, 2, 3), keepdims=True)
            eff_lt = num / den                              # scalar per time bin
            counts_out[:, g, :, :] = counts_corr[:, g, :, :] * eff_lt
            counts_var_out[:, g, :, :] = counts_var_corr[:, g, :, :] * eff_lt
            new_livefrac[:, g, :, :] = np.broadcast_to(eff_lt, new_livefrac[:, g, :, :].shape)
        return counts_out, counts_var_out, new_livefrac

    @staticmethod
    def _data_select(product,
                    detector_indices,
                    pixel_indices,
                    energy_indices,
                    time_indices,
                    livefrac,
                    livefrac_error,
                    elut_cor_fac,
                    rcr,
                    sum_all_times,
                    systematic,
                    sunkit_spex_detector_sum,
                    bkg):

        """
        Select and/or sum counts, variance, livetime fraction, and associated metadata
        along the detector, pixel, energy, and time axes according to the requested
        indices.

        Accepts either a `ScienceData` product (from which counts, variance, time
        normalization, energy normalization, times, and energies are extracted) or a
        pre-unpacked tuple of the same quantities (e.g. as produced by `_bkg_sub`). For
        each of detector, pixel, energy, and time axes, indices given as a flat 1D
        array are treated as a boolean mask/selection, while indices given as a 2D
        array of [start, end] pairs are summed (or averaged, for livetime fraction)
        within each pair and concatenated across pairs. If `sum_all_times=True` and
        multiple time bins were requested, all resulting time bins are further summed
        into one.

        Parameters
        ----------
        product : ScienceData or tuple
            The science data product, or a pre-extracted tuple of
            (counts, counts_var, t_norm, e_norm, livefrac, elut_cor_fac, times, energies).
        detector_indices : numpy.ndarray or None
            Detector indices to select/sum, as a flat array or 2D array of ranges.
        pixel_indices : list, numpy.ndarray, or None
            Pixel indices to select/sum, as a flat array or 2D array of ranges.
        energy_indices : list, numpy.ndarray, or None
            Energy indices to select/sum, as a flat array or 2D array of ranges.
        time_indices : list, numpy.ndarray, or None
            Time indices to select/sum, as a flat array or 2D array of ranges. If the
            first element is a string or `Time` object, time selection is skipped.
        livefrac : numpy.ndarray or None
            Livetime fraction array, selected/averaged alongside the other axes.
        livefrac_error : numpy.ndarray or None
            Uncertainty on the livetime fraction, selected/combined alongside
            `livefrac`.
        elut_cor_fac : numpy.ndarray or None
            ELUT correction factor, selected/averaged along the energy axis.
        sum_all_times : bool
            If True and `time_indices` produced multiple bins, sum all bins into one.

        Returns
        -------
        tuple
            (counts, counts_var, t_norm, e_norm, livefrac, livefrac_error,
            elut_cor_fac, times, energies) after applying the requested selection
            and/or summation.
        """

        if isinstance(product, ScienceData):

            e_norm = product.dE
            counts = product.data["counts"]

            shape = counts.shape

            try:
                counts_var = product.data["counts_comp_err"] ** 2
            except KeyError:
                counts_var = product.data["counts_comp_comp_err"] ** 2

            if len(shape) < 4:
                counts = counts.reshape(shape[0], 1, 1, shape[-1])
                counts_var = counts_var.reshape(shape[0], 1, 1, shape[-1])

                detector_indices = None
                pixel_indices = None

            counts_var = np.sqrt(counts.value + counts_var.value) * u.ct

            t_norm = product.data["timedel"]
            times = product.times
            energies = product.energies
            rcr = product.rcr_shifted

        else:

            counts, counts_var, t_norm, e_norm, livefrac,livefrac_error, elut_cor_fac, times, energies, rcr = product

        if not bkg:

            if elut_cor_fac is not None:

                counts = counts * elut_cor_fac
                counts_var = counts_var * elut_cor_fac

        # ------------------------------------------------------------------ #
        # Livetime helper.
        # `groups` is a list of detector-axis index arrays; each entry lists
        # the detectors that will be combined into ONE output spectrum. For each
        # group we divide each detector by its own livefrac, then rescale by a
        # single eff-livetime fraction (summed over detector, pixel AND energy)
        # so the group total matches the raw ELUT-corrected total — the IDL
        # `spec_in_corr` scheme. `livefrac` is replaced by eff_lt (broadcast
        # across the group) so the downstream mean() over the group is a no-op
        # and the exposure carries eff_lt, not raw livefrac. Numerators are read
        # from the untouched input `counts`, so groups cannot alias.
        # ------------------------------------------------------------------ #
        # def _apply_livetime(counts, counts_var, livefrac, groups):
        #     counts_corr = counts / livefrac
        #     counts_var_corr = counts_var / livefrac
        #     counts_out = counts.copy()
        #     counts_var_out = counts_var.copy()
        #     new_livefrac = livefrac.copy()
        #     for g in groups:
        #         g = np.atleast_1d(np.asarray(g))
        #         num = np.nansum(counts[:, g, :, :], axis=(1, 2, 3), keepdims=True)
        #         den = np.nansum(counts_corr[:, g, :, :], axis=(1, 2, 3), keepdims=True)
        #         eff_lt = num / den                              # scalar per time bin
        #         counts_out[:, g, :, :] = counts_corr[:, g, :, :] * eff_lt
        #         counts_var_out[:, g, :, :] = counts_var_corr[:, g, :, :] * eff_lt
        #         new_livefrac[:, g, :, :] = np.broadcast_to(eff_lt, new_livefrac[:, g, :, :].shape)
        #     return counts_out, counts_var_out, new_livefrac

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
                    [np.sqrt(np.sum(counts_var[..., pl : ph + 1, :]**2, axis=2, keepdims=True)) for pl, ph in pixel_indices], axis=2
                )

                # FIXED: was iterating over `detector_indices` (crashes when it is
                # None, e.g. the spectrogram path). Rebin livefrac over pixels.
                if livefrac is not None:
                    livefrac = np.concatenate(
                        [np.mean(livefrac[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices],
                        axis=2,
                    )

                if livefrac_error is not None:
                    livefrac_error = np.concatenate(
                        [np.sqrt(np.mean(livefrac_error[..., pl : ph + 1, :]**2, axis=2, keepdims=True)) for pl, ph in pixel_indices],
                        axis=2,
                    )

        if energy_indices is not None:
            energy_indices = np.asarray(energy_indices)
            if energy_indices.ndim == 1:
                energy_mask = np.full(shape[-1], False)
                energy_mask[energy_indices] = True
                counts = counts[..., energy_mask]
                counts_var = counts_var[..., energy_mask]
                e_norm = e_norm[energy_mask]
                energies = energies[energy_mask]
                if elut_cor_fac is not None:
                    elut_cor_fac = elut_cor_fac[energy_mask]

            if energy_indices.ndim == 2:
                counts = np.concatenate(
                    [np.sum(counts[..., el : eh + 1], axis=-1, keepdims=True) for el, eh in energy_indices], axis=-1
                )

                counts_var = np.concatenate(
                    [np.sqrt(np.sum(counts_var[..., el : eh + 1]**2, axis=-1, keepdims=True)) for el, eh in energy_indices], axis=-1
                )

                e_norm = np.hstack([(energies["e_high"][eh] - energies["e_low"][el]) for el, eh in energy_indices])

                # NOTE (unverified — see message): this rebins elut_cor_fac from
                # counts_var, which is almost certainly a copy/paste bug. Left
                # verbatim because ELUT is already applied to counts above, so the
                # returned elut_cor_fac is likely unused downstream in the no-bkg
                # path. Fix before relying on 2-D energy grouping.
                if elut_cor_fac is not None:
                    elut_cor_fac = np.concatenate(
                        [np.mean(counts_var[..., el : eh + 1]) for el, eh in energy_indices], axis=-1
                    )

                energies = np.atleast_2d(
                    [
                        (energies["e_low"][el].value, energies["e_high"][eh].value)
                        for el, eh in energy_indices
                    ]
                )
                energies = QTable(energies * u.keV, names=["e_low", "e_high"])

        # ---- livetime fallback: no detector selection supplied -------------- #
        # Covers the spectrogram path (detector axis size 1, detector_indices
        # forced to None above) and any caller that omits detector selection.
        # The detector-averaged (pooled) livetime is applied ONLY when
        # sunkit_spex_detector_sum=True; otherwise counts stay raw and the raw
        # per-detector livefrac is folded into the exposure downstream.
        if not bkg and livefrac is not None and detector_indices is None and sunkit_spex_detector_sum:
            n_det = counts.shape[1]
            groups = [np.arange(n_det)]
            counts, counts_var, livefrac = ScienceData._apply_livetime(counts, counts_var, livefrac, groups)

        if detector_indices is not None:

            detector_indices = np.asarray(detector_indices)   # "top24" must already be resolved to indices upstream

            if systematic:
                e_low = energies["e_low"].value
                systematic_err_percentage = np.select(
                    [e_low < 7, (e_low < 10) & (e_low >= 7), e_low >= 10],
                    [0.07, 0.05, 0.03],
                )

                if sunkit_spex_detector_sum:
                    # -------- CASE A: sum=True --------
                    # All selected detectors are one combined entity. Derive from the
                    # GRAND total over the full selected set and spread evenly, so that
                    # flat "top24" and any nested partition of the same 24 detectors
                    # reconcile to the identical number once bins are quadrature-combined
                    # downstream. (This is the existing behaviour.)
                    if detector_indices.ndim == 1:
                        all_selected = detector_indices
                    else:
                        all_selected = np.concatenate([np.arange(dl, dh + 1) for dl, dh in detector_indices])

                    n_total = len(all_selected) * counts.shape[2]           # detectors × remaining pixels
                    grand_total = counts[:, all_selected, :, :].sum(axis=(1, 2), keepdims=True)

                    sys_err_total = systematic_err_percentage * grand_total
                    sys_err_elem = sys_err_total / np.sqrt(n_total)

                    sys_err_full = np.broadcast_to(
                        sys_err_elem.value, counts[:, all_selected, :, :].shape
                    ) * sys_err_elem.unit

                    counts_var[:, all_selected, :, :] = np.sqrt(
                        counts_var[:, all_selected, :, :].value**2 + sys_err_full.value**2
                    ) * u.ct

                else:
                    # -------- CASE B: sum=False --------
                    # Each OUTPUT bin carries a systematic derived from ITS OWN counts.
                    if detector_indices.ndim == 1:
                        # Flat: each detector stays its own bin (no detector collapse),
                        # so apply p × (that detector's own pixel-summed counts). Spread
                        # over the pixel axis by sqrt(n_pix) so downstream pixel pooling
                        # reconstructs it; if pixels are already pooled n_pix==1 and this
                        # is just p × count.
                        n_pix = counts.shape[2]
                        det_total = counts[:, detector_indices, :, :].sum(axis=2, keepdims=True)
                        sys_err_elem = (systematic_err_percentage * det_total) / np.sqrt(n_pix)

                        sys_err_full = np.broadcast_to(
                            sys_err_elem.value, counts[:, detector_indices, :, :].shape
                        ) * sys_err_elem.unit

                        cv_sel = counts_var[:, detector_indices, :, :]
                        counts_var[:, detector_indices, :, :] = np.sqrt(
                            cv_sel.value**2 + sys_err_full.value**2
                        ) * u.ct

                    else:
                        # Nested: each group collapses to one bin, so derive from THAT
                        # GROUP's own total and spread by sqrt(n_group) so the group's
                        # quadrature collapse reconstructs p × group_total for that group.
                        for dl, dh in detector_indices:
                            n_group = (dh - dl + 1) * counts.shape[2]
                            group_total = counts[:, dl:dh + 1, :, :].sum(axis=(1, 2), keepdims=True)

                            sys_err_total = systematic_err_percentage * group_total
                            sys_err_elem = sys_err_total / np.sqrt(n_group)

                            sys_err_full = np.broadcast_to(
                                sys_err_elem.value, counts[:, dl:dh + 1, :, :].shape
                            ) * sys_err_elem.unit

                            counts_var[:, dl:dh + 1, :, :] = np.sqrt(
                                counts_var[:, dl:dh + 1, :, :].value**2 + sys_err_full.value**2
                            ) * u.ct

            # ---- livetime: detector-averaged (pooled) eff_lt --------------------- #
            # Applied ONLY when sunkit_spex_detector_sum=True, i.e. when detectors
            # are combined into one spectrum. Runs while detectors are still at full
            # resolution so the pooling is over the raw per-detector counts. When
            # sum=False we skip this entirely: counts stay raw (ELUT-corrected) and
            # the raw per-detector livefrac is carried through to the exposure, so
            # each detector keeps its own livetime. (For a single-detector group the
            # pooled eff_lt equals that detector's livefrac, so skipping is exact.)
            if not bkg and livefrac is not None and sunkit_spex_detector_sum:
                if detector_indices.ndim == 1:
                    groups = [detector_indices]                          # all selected -> one spectrum
                else:  # ndim == 2 : each (dl, dh) range -> one output spectrum
                    groups = [np.arange(dl, dh + 1) for dl, dh in detector_indices]
                
                counts_var = ScienceData._livetime_uncertainty(counts_var,livefrac_error)

                counts, counts_var, livefrac = ScienceData._apply_livetime(counts, counts_var, livefrac, groups)

            # -------- selection / collapse: unchanged from your original --------
            if detector_indices.ndim == 1:
                detector_mask = np.full(32, False)
                detector_mask[detector_indices] = True
                counts = counts[:, detector_mask, ...]
                counts_var = counts_var[:, detector_mask, ...]
                if livefrac is not None:
                    livefrac = livefrac[:, detector_mask, :, :]
                if livefrac_error is not None:
                    livefrac_error = livefrac_error[:, detector_mask, :, :]

            if detector_indices.ndim == 2:
                counts = np.hstack(
                    [np.sum(counts[:, dl:dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices]
                )
                counts_var = np.concatenate(
                    [np.sqrt(np.sum(counts_var[:, dl:dh + 1, ...]**2, axis=1, keepdims=True)) for dl, dh in detector_indices],
                    axis=1,
                )
                if livefrac is not None:
                    livefrac = np.concatenate(
                        [np.mean(livefrac[:, dl:dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices],
                        axis=1,
                    )
                if livefrac_error is not None:
                    livefrac_error = np.concatenate(
                        [np.sqrt(np.mean(livefrac_error[:, dl:dh + 1, ...]**2, axis=1, keepdims=True)) for dl, dh in detector_indices],
                        axis=1,
                    )

        # (old standalone `if not bkg: if livefrac is not None:` block removed —
        #  its work is now done inside the detector conditional / fallback above.)

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
                    livefrac = np.vstack([np.mean(livefrac[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices])

                if livefrac_error is not None:
                    livefrac_error = np.vstack([np.sqrt(np.mean(livefrac_error[tl : th + 1, ...]**2, axis=0, keepdims=True)) for tl, th in time_indices])

                counts_var = np.vstack(
                    [np.sqrt(np.sum(counts_var[tl : th + 1, ...]**2, axis=0, keepdims=True)) for tl, th in time_indices]
                )
                t_norm = dt

                if sum_all_times and len(new_times) > 1:
                    counts = np.sum(counts, axis=0, keepdims=True)
                    counts_var = np.sum(counts_var, axis=0, keepdims=True)
                    t_norm = np.sum(dt)

        return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr
    
    # def _data_select(product, 
    #                 detector_indices,
    #                 pixel_indices,
    #                 energy_indices,
    #                 time_indices,
    #                 livefrac,
    #                 livefrac_error,
    #                 elut_cor_fac,
    #                 rcr,
    #                 sum_all_times,
    #                 systematic,
    #                 sunkit_spex_detector_sum,
    #                 bkg):
        
    #     """
    #     Select and/or sum counts, variance, livetime fraction, and associated metadata
    #     along the detector, pixel, energy, and time axes according to the requested
    #     indices.

    #     Accepts either a `ScienceData` product (from which counts, variance, time
    #     normalization, energy normalization, times, and energies are extracted) or a
    #     pre-unpacked tuple of the same quantities (e.g. as produced by `_bkg_sub`). For
    #     each of detector, pixel, energy, and time axes, indices given as a flat 1D
    #     array are treated as a boolean mask/selection, while indices given as a 2D
    #     array of [start, end] pairs are summed (or averaged, for livetime fraction)
    #     within each pair and concatenated across pairs. If `sum_all_times=True` and
    #     multiple time bins were requested, all resulting time bins are further summed
    #     into one.

    #     Parameters
    #     ----------
    #     product : ScienceData or tuple
    #         The science data product, or a pre-extracted tuple of
    #         (counts, counts_var, t_norm, e_norm, livefrac, elut_cor_fac, times, energies).
    #     detector_indices : numpy.ndarray or None
    #         Detector indices to select/sum, as a flat array or 2D array of ranges.
    #     pixel_indices : list, numpy.ndarray, or None
    #         Pixel indices to select/sum, as a flat array or 2D array of ranges.
    #     energy_indices : list, numpy.ndarray, or None
    #         Energy indices to select/sum, as a flat array or 2D array of ranges.
    #     time_indices : list, numpy.ndarray, or None
    #         Time indices to select/sum, as a flat array or 2D array of ranges. If the
    #         first element is a string or `Time` object, time selection is skipped.
    #     livefrac : numpy.ndarray or None
    #         Livetime fraction array, selected/averaged alongside the other axes.
    #     livefrac_error : numpy.ndarray or None
    #         Uncertainty on the livetime fraction, selected/combined alongside
    #         `livefrac`.
    #     elut_cor_fac : numpy.ndarray or None_data_select
    #         ELUT correction factor, selected/averaged along the energy axis.
    #     sum_all_times : bool
    #         If True and `time_indices` produced multiple bins, sum all bins into one.

    #     Returns
    #     -------
    #     tuple
    #         (counts, counts_var, t_norm, e_norm, livefrac, livefrac_error,
    #         elut_cor_fac, times, energies) after applying the requested selection
    #         and/or summation.
    #     """

    #     if isinstance(product,ScienceData):

    #         e_norm = product.dE
    #         counts = product.data["counts"]

    #         shape = counts.shape

    #         try: 
    #             counts_var = product.data["counts_comp_err"] ** 2
    #         except KeyError:
    #             counts_var = product.data["counts_comp_comp_err"] ** 2

    #         if len(shape) < 4:
    #             counts = counts.reshape(shape[0], 1, 1, shape[-1])
    #             counts_var = counts_var.reshape(shape[0], 1, 1, shape[-1])
                
    #             detector_indices = None
    #             pixel_indices = None

    #         counts_var = np.sqrt(counts.value + counts_var.value) *u.ct

    #         t_norm = product.data["timedel"]
    #         times = product.times
    #         energies = product.energies
    #         rcr = product.rcr_shifted

    #     else:

    #         counts, counts_var, t_norm, e_norm, livefrac, elut_cor_fac, times, energies, rcr = product
        

    #     # if systematic:

    #     #     e_low = energies["e_low"].value

    #     #     energy_conditions = [e_low < 7, (e_low < 10) & (e_low >= 7), e_low >= 10]
    #     #     percentage = [0.07, 0.05, 0.03]

    #     #     systematic_err_percentage = np.select(energy_conditions, percentage)

    #     #     idx = np.ix_(detector_indices, pixel_indices)

    #     #     # Select relevant detectors/pixels, collapse coherently to get the true total
    #     #     counts_selected = counts[:, idx[0], idx[1], :]
    #     #     counts_collapsed_for_sys = counts_selected.sum(axis=(1, 2), keepdims=True)


    #     #     # Correct, correlated systematic error on the TOTAL
    #     #     systematic_err_total = systematic_err_percentage * counts_collapsed_for_sys  

    #     #     # Number of elements being collapsed over (detector x pixel)
    #     #     N = len(detector_indices) * len(pixel_indices)

    #     #     # Pre-divide by sqrt(N) so that when this gets quadrature-summed
    #     #     # downstream over the same axes, it reconstructs the correct total
    #     #     systematic_err_per_element = systematic_err_total / np.sqrt(N)  

    #     #     # Broadcast back to the FULL original shape so it
    #     #     # aligns with counts_var before any slicing/compression happens
    #     #     systematic_err_full = np.broadcast_to(
    #     #         systematic_err_per_element.value,
    #     #         counts.shape
    #     #     ) * systematic_err_per_element.unit

    #     #     # Now combine BEFORE collapsing, since downstream code expects that shape
    #     #     counts_var = np.sqrt(counts_var.value**2 + systematic_err_full.value**2) * u.ct

    #     if not bkg:

    #         if elut_cor_fac is not None:

    #             counts = counts * elut_cor_fac 
    #             counts_var = counts_var * elut_cor_fac 
      

    #     if pixel_indices is not None:

    #         pixel_indices = np.asarray(pixel_indices)
    #         if pixel_indices.ndim == 1:
    #             pixel_mask = np.full(12, False)
    #             pixel_mask[pixel_indices] = True
    #             num_pixels = counts.shape[2]
    #             counts = counts[..., pixel_mask[:num_pixels], :]
    #             counts_var = counts_var[..., pixel_mask[:num_pixels], :]
    #             if livefrac is not None and livefrac.shape[2] !=1:
    #                 livefrac = livefrac[:,:,pixel_mask[:num_pixels],:]
    #             if livefrac_error is not None and livefrac_error.shape[2] !=1:
    #                 livefrac_error = livefrac_error[:,:,pixel_mask[:num_pixels],:]      


    #         if pixel_indices.ndim == 2:
    #             counts = np.concatenate(
    #                 [np.sum(counts[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices], axis=2
    #             )

    #             counts_var = np.concatenate(
    #                 [np.sqrt(np.sum(counts_var[..., pl : ph + 1, :]**2, axis=2, keepdims=True)) for pl, ph in pixel_indices], axis=2
    #             )

    #             if livefrac is not None:                
    #                 livefrac = np.concatenate(
    #                 [np.mean(livefrac[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in detector_indices],
    #                 axis=2,
    #                 )

    #             if livefrac_error is not None:                
    #                 livefrac_error = np.concatenate(
    #                 [np.sqrt(np.mean(livefrac_error[..., pl : ph + 1, :]**2, axis=2, keepdims=True)) for pl, ph in detector_indices],
    #                 axis=2,
    #                 )

    #     if energy_indices is not None:
    #         energy_indices = np.asarray(energy_indices)
    #         if energy_indices.ndim == 1:
    #             energy_mask = np.full(shape[-1], False)
    #             energy_mask[energy_indices] = True
    #             counts = counts[..., energy_mask]
    #             counts_var = counts_var[..., energy_mask]
    #             e_norm = e_norm[energy_mask]
    #             energies = energies[energy_mask]
    #             if elut_cor_fac is not None:
    #                 elut_cor_fac = elut_cor_fac[energy_mask]

    #         if energy_indices.ndim == 2:
    #             counts = np.concatenate(
    #                 [np.sum(counts[..., el : eh + 1], axis=-1, keepdims=True) for el, eh in energy_indices], axis=-1
    #             )

    #             counts_var = np.concatenate(
    #                 [np.sqrt(np.sum(counts_var[..., el : eh + 1]**2, axis=-1, keepdims=True)) for el, eh in energy_indices], axis=-1
    #             )

        
    #             e_norm = np.hstack([(energies["e_high"][eh] - energies["e_low"][el]) for el, eh in energy_indices])

    #             if elut_cor_fac is not None:
    #                 elut_cor_fac = np.concatenate(
    #                 [np.mean(counts_var[..., el : eh + 1]) for el, eh in energy_indices], axis=-1
    #             )

    #             energies = np.atleast_2d(
    #                 [
    #                     (energies["e_low"][el].value, energies["e_high"][eh].value)
    #                     for el, eh in energy_indices
    #                 ]
    #             )
    #             energies = QTable(energies * u.keV, names=["e_low", "e_high"])


    #     if detector_indices is not None:

    #         detector_indices = np.asarray(detector_indices)   # "top24" must already be resolved to indices upstream

    #         if systematic:
    #             e_low = energies["e_low"].value
    #             systematic_err_percentage = np.select(
    #                 [e_low < 7, (e_low < 10) & (e_low >= 7), e_low >= 10],
    #                 [0.07, 0.05, 0.03],
    #             )

    #             if sunkit_spex_detector_sum:
    #                 # -------- CASE A: sum=True --------
    #                 # All selected detectors are one combined entity. Derive from the
    #                 # GRAND total over the full selected set and spread evenly, so that
    #                 # flat "top24" and any nested partition of the same 24 detectors
    #                 # reconcile to the identical number once bins are quadrature-combined
    #                 # downstream. (This is the existing behaviour.)
    #                 if detector_indices.ndim == 1:
    #                     all_selected = detector_indices
    #                 else:
    #                     all_selected = np.concatenate([np.arange(dl, dh + 1) for dl, dh in detector_indices])

    #                 n_total = len(all_selected) * counts.shape[2]           # detectors × remaining pixels
    #                 grand_total = counts[:, all_selected, :, :].sum(axis=(1, 2), keepdims=True)

    #                 sys_err_total = systematic_err_percentage * grand_total
    #                 sys_err_elem = sys_err_total / np.sqrt(n_total)

    #                 sys_err_full = np.broadcast_to(
    #                     sys_err_elem.value, counts[:, all_selected, :, :].shape
    #                 ) * sys_err_elem.unit

    #                 counts_var[:, all_selected, :, :] = np.sqrt(
    #                     counts_var[:, all_selected, :, :].value**2 + sys_err_full.value**2
    #                 ) * u.ct

    #             else:
    #                 # -------- CASE B: sum=False --------
    #                 # Each OUTPUT bin carries a systematic derived from ITS OWN counts.
    #                 if detector_indices.ndim == 1:
    #                     # Flat: each detector stays its own bin (no detector collapse),
    #                     # so apply p × (that detector's own pixel-summed counts). Spread
    #                     # over the pixel axis by sqrt(n_pix) so downstream pixel pooling
    #                     # reconstructs it; if pixels are already pooled n_pix==1 and this
    #                     # is just p × count.
    #                     n_pix = counts.shape[2]
    #                     det_total = counts[:, detector_indices, :, :].sum(axis=2, keepdims=True)
    #                     sys_err_elem = (systematic_err_percentage * det_total) / np.sqrt(n_pix)

    #                     sys_err_full = np.broadcast_to(
    #                         sys_err_elem.value, counts[:, detector_indices, :, :].shape
    #                     ) * sys_err_elem.unit

    #                     cv_sel = counts_var[:, detector_indices, :, :]
    #                     counts_var[:, detector_indices, :, :] = np.sqrt(
    #                         cv_sel.value**2 + sys_err_full.value**2
    #                     ) * u.ct

    #                 else:
    #                     # Nested: each group collapses to one bin, so derive from THAT
    #                     # GROUP's own total and spread by sqrt(n_group) so the group's
    #                     # quadrature collapse reconstructs p × group_total for that group.
    #                     for dl, dh in detector_indices:
    #                         n_group = (dh - dl + 1) * counts.shape[2]
    #                         group_total = counts[:, dl:dh + 1, :, :].sum(axis=(1, 2), keepdims=True)

    #                         sys_err_total = systematic_err_percentage * group_total
    #                         sys_err_elem = sys_err_total / np.sqrt(n_group)

    #                         sys_err_full = np.broadcast_to(
    #                             sys_err_elem.value, counts[:, dl:dh + 1, :, :].shape
    #                         ) * sys_err_elem.unit

    #                         counts_var[:, dl:dh + 1, :, :] = np.sqrt(
    #                             counts_var[:, dl:dh + 1, :, :].value**2 + sys_err_full.value**2
    #                         ) * u.ct

    #         # -------- selection / collapse: unchanged from your original --------
    #         if detector_indices.ndim == 1:
    #             detector_mask = np.full(32, False)
    #             detector_mask[detector_indices] = True
    #             counts = counts[:, detector_mask, ...]
    #             counts_var = counts_var[:, detector_mask, ...]
    #             if livefrac is not None:
    #                 livefrac = livefrac[:, detector_mask, :, :]
    #             if livefrac_error is not None:
    #                 livefrac_error = livefrac_error[:, detector_mask, :, :]

    #         if detector_indices.ndim == 2:
    #             counts = np.hstack(
    #                 [np.sum(counts[:, dl:dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices]
    #             )
    #             counts_var = np.concatenate(
    #                 [np.sqrt(np.sum(counts_var[:, dl:dh + 1, ...]**2, axis=1, keepdims=True)) for dl, dh in detector_indices],
    #                 axis=1,
    #             )
    #             if livefrac is not None:
    #                 livefrac = np.concatenate(
    #                     [np.mean(livefrac[:, dl:dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices],
    #                     axis=1,
    #                 )
    #             if livefrac_error is not None:
    #                 livefrac_error = np.concatenate(
    #                     [np.sqrt(np.mean(livefrac_error[:, dl:dh + 1, ...]**2, axis=1, keepdims=True)) for dl, dh in detector_indices],
    #                     axis=1,
    #                 )

    #     if not bkg:

    #         if livefrac is not None:
    #             counts_corr = counts / livefrac
    #             counts_var_corr = counts_var / livefrac
    #             # single effective livetime fraction per time bin
    #             # (collapse detector, pixel AND energy, matching IDL's total-over-energy scalar)
    #             num = np.nansum(counts,      axis=(1, 2, 3), keepdims=True)
    #             den = np.nansum(counts_corr, axis=(1, 2, 3), keepdims=True)
    #             eff_lt = num / den                      # scalar per time bin, broadcast over E

    #             counts = counts_corr * eff_lt          # == IDL spec_in_corr after detector sum
    #             counts_var = counts_var_corr * eff_lt  
    #             # make the downstream exposure use eff_lt, not raw livefrac,
    #             # so livetime isn't applied twice:
    #             livefrac = np.broadcast_to(eff_lt, livefrac.shape).copy()            


    #     if time_indices is not None:
    #         time_indices = np.asarray(time_indices)
    #         if time_indices.ndim == 1:
    #             time_mask = np.full(times.shape, False)
    #             time_mask[time_indices] = True
    #             counts = counts[time_mask, ...]
    #             counts_var = counts_var[time_mask, ...]
    #             t_norm = t_norm[time_mask]
    #             rcr = rcr[time_mask]
    #             if livefrac is not None:
    #                 livefrac = livefrac[time_mask, ...]
    #             times = times[time_mask]

    #         if time_indices.ndim == 2:
    #             new_times = []
    #             dt = []
    #             for tl, th in time_indices:

    #                 ts = times[tl] - t_norm[tl] * 0.5
    #                 te = times[th] + t_norm[th] * 0.5
    #                 td = te - ts
    #                 tc = ts + (td * 0.5)
    #                 dt.append(td.to("s"))
    #                 new_times.append(tc) 

    #             dt = np.hstack(dt)
    #             times = Time(new_times)

    #             counts = np.vstack([np.sum(counts[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices])
    #             rcr = np.vstack([np.mean(rcr[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices])

    #             if livefrac is not None:
    #                 livefrac = np.vstack([np.mean(livefrac[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices])
                
    #             if livefrac_error is not None:
    #                 livefrac_error = np.vstack([np.sqrt(np.mean(livefrac[tl : th + 1, ...]**2, axis=0, keepdims=True)) for tl, th in time_indices])

    #             counts_var = np.vstack(
    #                 [np.sqrt(np.sum(counts_var[tl : th + 1, ...]**2, axis=0, keepdims=True)) for tl, th in time_indices]
    #             )
    #             t_norm = dt

    #             if sum_all_times and len(new_times) > 1:
    #                 counts = np.sum(counts, axis=0, keepdims=True)
    #                 counts_var = np.sum(counts_var, axis=0, keepdims=True)
    #                 t_norm = np.sum(dt)
    
    #     return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr
    
    @staticmethod
    def _bkg_sub(product,
                bkg,
                detector_indices,
                pixel_indices,
                sunkit_spex_detector_sum,
                detector_indices_bkg,
                pixel_indices_bkg,
                energy_indices_bkg,
                livefrac,
                livefrac_error,
                livefrac_bkg,
                livefrac_error_bkg,
                elut_cor_fac,
                rcr): 
        
        """
        Perform livetime- and ELUT-corrected background subtraction of a science
        product using a matched background product.

        Computes the livetime- and uncorrected count rates for both the science and
        background data, scales the background counts to the science product's
        integration times, and subtracts the scaled background from the science
        counts. Uncertainties are propagated in quadrature. Handles removal of the
        zero-energy bin and any trailing NaN energy bin, and computes an effective
        livetime fraction from the ratio of uncorrected to livetime-corrected counts
        summed over energy.

        Parameters
        ----------
        product : ScienceData
            The science data product to background-subtract.
        bkg : ScienceData
            The background data product.
        detector_indices_bkg : list or numpy.ndarray
            Detector indices in the background product matching those available in
            `product`.
        pixel_indices_bkg : list or numpy.ndarray
            Pixel indices in the background product matching those available in
            `product`.
        energy_indices_bkg : numpy.ndarray
            Energy indices in the background product matching the energy bins of
            `product`.
        livefrac : numpy.ndarray
            Livetime fraction for the science product.
        livefrac_error : numpy.ndarray
            Uncertainty on the livetime fraction for the science product.
        livefrac_bkg : numpy.ndarray
            Livetime fraction for the background product.
        livefrac_error_bkg : numpy.ndarray
            Uncertainty on the livetime fraction for the background product.
        elut_cor_fac : numpy.ndarray
            ELUT correction factor to apply to both science and background counts.

        Returns
        -------
        tuple
            (counts, counts_var, t_norm, e_norm, livefrac, elut_cor_fac, times,
            energies) for the background-subtracted science data, where `livefrac`
            here is the effective livetime fraction derived from the subtraction.
        """


        e_norm = product.dE
        counts = product.data["counts"]
        shape = counts.shape

        try:
            counts_var = (product.data["counts_comp_err"].value ** 2)*u.ct
        except KeyError:
            counts_var =  (product.data["counts_comp_comp_err"].value ** 2)*u.ct

        counts_bkg = bkg.data["counts"]

        try:
            counts_var_bkg = (bkg.data["counts_comp_err"].value ** 2) *u.ct
        except KeyError:
            counts_var_bkg = (bkg.data["counts_comp_comp_err"].value ** 2) *u.ct
        
        counts_var_bkg = np.sqrt(counts_bkg + counts_var_bkg) 
        counts_var_bkg = ScienceData._livetime_uncertainty(counts_var_bkg,livefrac_error_bkg) 


        counts_bkg = counts_bkg[:,detector_indices_bkg,:,:]
        counts_bkg = counts_bkg[:,:,pixel_indices_bkg,:]
        counts_bkg = counts_bkg[:,:,:,energy_indices_bkg]

        counts_var_bkg = counts_var_bkg[:,detector_indices_bkg,:,:]
        counts_var_bkg = counts_var_bkg[:,:,pixel_indices_bkg,:]
        counts_var_bkg = counts_var_bkg[:,:,:,energy_indices_bkg]

        if len(shape) < 4:

            counts = counts.reshape(shape[0], 1, 1, shape[-1])
            counts_var = counts_var.reshape(shape[0], 1, 1, shape[-1])

            livefrac = np.nanmean(livefrac,axis=1, keepdims=True)
            livefrac_error = np.nanmean(livefrac_error,axis=(1,2), keepdims=True)

            counts_bkg = np.nansum(counts_bkg, axis=(1,2), keepdims=True)
            counts_var_bkg = np.nansum(counts_var_bkg, axis=(1,2), keepdims=True)

            livefrac_bkg = np.nanmean(livefrac_bkg,axis=1, keepdims=True)
            livefrac_error_bkg = np.nanmean(livefrac_error_bkg,axis=(1,2), keepdims=True)

        
        counts_var = np.sqrt(counts + counts_var) 

        t_norm = product.data["timedel"]
        times = product.times
        energies = product.energies

        counts_var = ScienceData._livetime_uncertainty(counts_var,livefrac_error)   


        t_norm_bkg = bkg.data["timedel"]
        t_norm = t_norm.to(u.s)
        t_norm_bkg = t_norm_bkg.to(u.s)

        counts_uncorr = counts[...,:] * elut_cor_fac
        counts_lvtcorr = (counts[...,:] * elut_cor_fac) / livefrac

    

        counts_uncorr_bkg = counts_bkg[...,:] * elut_cor_fac
        counts_lvtcorr_bkg = (counts_bkg / livefrac_bkg)[...,:] * elut_cor_fac

        count_rate_uncorr_bkg = counts_uncorr_bkg  / t_norm_bkg.mean()
        count_uncorr_scaled_bkg = t_norm.reshape(len(t_norm), 1,1,1) * count_rate_uncorr_bkg


        count_rate_lvtcorr_bkg = counts_lvtcorr_bkg / t_norm_bkg.mean()
        count_lvtcorr_scaled_bkg = t_norm.reshape(len(t_norm), 1,1,1) * count_rate_lvtcorr_bkg

        counts_var_lvtcorr = (counts_var[...,:] * elut_cor_fac) / livefrac
        counts_var_lvtcorr_bkg = (counts_var_bkg / livefrac_bkg)[...,:] * elut_cor_fac
        counts_var_lvtcorr_scaled_bkg = (counts_var_lvtcorr_bkg / t_norm_bkg.mean()) * t_norm.reshape(len(t_norm), 1,1,1)

        spec_in_corr = counts_lvtcorr - count_lvtcorr_scaled_bkg
        spec_in = counts_uncorr - count_uncorr_scaled_bkg

        spec_in_err = np.sqrt( (counts_var_lvtcorr**2) + (counts_var_lvtcorr_scaled_bkg**2) )

        spec_in_corr_lvt = counts_lvtcorr
        spec_in_lvt = counts_uncorr

        if energies["e_low"][0].value == 0:
            spec_in = spec_in[..., 1:]
            spec_in_lvt = spec_in_lvt[..., 1:]
            spec_in_corr_lvt = spec_in_corr_lvt[..., 1:]
            spec_in_corr = spec_in_corr[..., 1:]
            spec_in_err = spec_in_err[..., 1:]
            energies = energies[1:]
            e_norm = e_norm[1:]
            elut_cor_fac = elut_cor_fac[1:]

        if np.isnan(energies["e_high"][-1].value):
            spec_in = spec_in[..., :-1]
            spec_in_corr = spec_in_corr[..., :-1]
            spec_in_lvt = spec_in_lvt[..., :-1]
            spec_in_corr_lvt = spec_in_corr_lvt[..., :-1]
            spec_in_err = spec_in_err[..., :-1]
            energies = energies[:-1]            
            e_norm = e_norm[:-1]
            elut_cor_fac = elut_cor_fac[:-1]


        # eff_livefrac = np.nansum(spec_in_lvt,axis=(3)) /  np.nansum(spec_in_corr_lvt,axis=(3)) 
        
        # eff_livefrac_used = np.nansum(spec_in_lvt[:, idx[0], idx[1], :], axis=(1, 2, 3), keepdims=True) / np.nansum(spec_in_corr_lvt[:, idx[0], idx[1], :], axis=(1, 2, 3), keepdims=True)
        # spec_in_final = spec_in_corr * eff_livefrac_used
        # spec_in_err_final = spec_in_err * eff_livefrac_used

        # spec_in_final = spec_in_corr * eff_livefrac[...,None]
        # spec_in_err_final = spec_in_err * eff_livefrac[...,None]

        # if detector_indices.ndim == 2:
        #      detector_indices = ScienceData._indices_expand_ranges(detector_indices,nest=False)
        
        # if pixel_indices.ndim == 2:
        #      pixel_indices = ScienceData._indices_expand_ranges(pixel_indices,nest=False)

        detector_groups = None
        if detector_indices.ndim == 2:
            detector_groups = ScienceData._indices_expand_ranges(detector_indices, nest=True)   # list of per-group arrays
            detector_indices = np.concatenate(detector_groups)                                   # flat — identical to nest=False

        if pixel_indices.ndim == 2:
            pixel_indices = ScienceData._indices_expand_ranges(pixel_indices, nest=False)

        # counts = spec_in_final

        if sunkit_spex_detector_sum:

            idx = np.ix_(detector_indices, pixel_indices)

            eff_livefrac= np.nansum(spec_in_lvt[:, idx[0], idx[1], :], axis=(1, 2, 3), keepdims=True) / np.nansum(spec_in_corr_lvt[:, idx[0], idx[1], :], axis=(1, 2, 3), keepdims=True)
            
            
            spec_in_final = spec_in_corr * eff_livefrac
            spec_in_err_final = spec_in_err * eff_livefrac


            counts = spec_in_final

            counts_check = np.nansum(spec_in_final[:, idx[0], idx[1], :], axis=(1,2), keepdims=True)
            counts = np.where(counts_check < 0, 0, counts)
            # spec_in_err_final = np.where(counts_check < 0, 0, spec_in_err_final)

            # print('err shape = ',spec_in_err_final[:, idx[0], idx[1], :].shape)
            err_f = spec_in_err_final[:, idx[0], idx[1], :]
            # np.save('err_check.npy',np.array(np.sqrt(np.nansum(err_f[165:174]**2,axis=(0,1,2)))))

            counts_var = spec_in_err_final
            livefrac =  np.broadcast_to(eff_livefrac, counts.shape)

            # idx = np.ix_(detector_indices, pixel_indices)

            # # --- Aggregate eff_livefrac: unchanged, still drives `counts` ---
            # eff_livefrac = np.nansum(spec_in_lvt[:, idx[0], idx[1], :], axis=(1, 2, 3), keepdims=True) \
            #             / np.nansum(spec_in_corr_lvt[:, idx[0], idx[1], :], axis=(1, 2, 3), keepdims=True)

            # spec_in_final = spec_in_corr * eff_livefrac
            # counts = spec_in_final


            # counts_check = np.nansum(spec_in_final[:, idx[0], idx[1], :], axis=(1, 2), keepdims=True)
            # counts = np.where(counts_check < 0, 0, counts)

            # # --- Per-detector/pixel eff_livefrac: used ONLY for the error term ---
            # eff_livefrac_per_dp = np.nansum(spec_in_lvt, axis=(3)) / np.nansum(spec_in_corr_lvt, axis=(3))
            # # shape: (time, n_detectors_total, n_pixels_total)

            # spec_in_err_final = spec_in_err * eff_livefrac_per_dp[..., None]

            # err_f = spec_in_err_final[:, idx[0], idx[1], :]
            # np.save('err_check.npy', np.array(np.sqrt(np.nansum(err_f**2, axis=(0, 1, 2)))))

            # print(np.isnan(spec_in_err[:, idx[0], idx[1], :]).sum())

            # # Check for NaNs in the components feeding eff_livefrac
            # print(np.isnan(spec_in_lvt[:, idx[0], idx[1], :]).sum())
            # print(np.isnan(spec_in_corr_lvt[:, idx[0], idx[1], :]).sum())

            # counts_var = spec_in_err_final
            # livefrac = np.broadcast_to(eff_livefrac, counts.shape)

        # else:

        #     eff_livefrac = np.nansum(spec_in_lvt,axis=(3)) /  np.nansum(spec_in_corr_lvt,axis=(3))
        #     spec_in_final = spec_in_corr * eff_livefrac[...,None]
        #     spec_in_err_final = spec_in_err * eff_livefrac[...,None]
        #     counts = spec_in_final

        #     counts = np.where(counts < 0, 0, counts)
        
        #     counts_var = spec_in_err_final

        #     livefrac = eff_livefrac[:, :, :, np.newaxis]


        else:  # sunkit_spex_detector_sum is False

            if detector_groups is None:
                # ---- flat: genuinely per-detector/pixel, unchanged ----
                eff_livefrac = np.nansum(spec_in_lvt, axis=3) / np.nansum(spec_in_corr_lvt, axis=3)
                spec_in_final = spec_in_corr * eff_livefrac[..., None]
                spec_in_err_final = spec_in_err * eff_livefrac[..., None]
                counts = np.where(spec_in_final < 0, 0, spec_in_final)
                counts_var = spec_in_err_final
                livefrac = eff_livefrac[:, :, :, np.newaxis]

            else:
                # ---- nested: each inner list is its own mini detector-sum ----
                spec_in_final = spec_in_corr.copy()
                spec_in_err_final = spec_in_err.copy()
                eff_livefrac_full = np.full(
                    (spec_in_lvt.shape[0], spec_in_lvt.shape[1], spec_in_lvt.shape[2], 1),
                    np.nan,
                )

                for group_dets in detector_groups:
                    gidx = np.ix_(group_dets, pixel_indices)

                    # count-weighted ratio over THIS group's detectors + selected pixels,
                    # exactly like the sum=True combined ratio but per group
                    group_eff = np.nansum(spec_in_lvt[:, gidx[0], gidx[1], :], axis=(1, 2, 3), keepdims=True) \
                            / np.nansum(spec_in_corr_lvt[:, gidx[0], gidx[1], :], axis=(1, 2, 3), keepdims=True)

                    # write the group's single ratio onto every detector in the group
                    # (all pixels), so _data_select's later per-group mean returns it unchanged
                    eff_livefrac_full[:, group_dets, :, :] = group_eff

                    spec_in_final[:, group_dets, :, :] = spec_in_corr[:, group_dets, :, :] * group_eff
                    spec_in_err_final[:, group_dets, :, :] = spec_in_err[:, group_dets, :, :] * group_eff

                counts = np.where(spec_in_final < 0, 0, spec_in_final)
                counts_var = spec_in_err_final
                livefrac = eff_livefrac_full

        return counts, counts_var, t_norm, e_norm, livefrac,livefrac_error, elut_cor_fac, times, energies, rcr
                                                                       
    @staticmethod
    def _energies_bkg_sub(product,bkg):

        """
        Find the energy bin indices in the background product that correspond to the
        energy bins present in the science product.

        Parameters
        ----------
        product : ScienceData
            The science data product whose energy bins define the reference set.
        bkg : ScienceData
            The background data product to be matched against the science product's
            energy bins.

        Returns
        -------
        numpy.ndarray
            Indices into `bkg.energies` corresponding to the energy bins shared with
            `product.energies`, in the order matching `product.energies["e_low"]`.
        """
        _, _, indices_sub = np.intersect1d(product.energies["e_low"], bkg.energies["e_low"], return_indices=True)

        return indices_sub

    @staticmethod
    def _bkg_indices_check(product, bkg):

        """
        Determine which detector and pixel indices are common to both a science
        product and its background product.

        Parameters
        ----------
        product : ScienceData
            The science data product.
        bkg : ScienceData
            The background data product.

        Returns
        -------
        tuple of list
            `pixel_indices` and `detector_indices` present in both `product` and
            `bkg`, ordered as they appear in `product`.
        """

        pixel_indices_full = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]
        pixel_indices_full_bkg = np.where(bkg.pixel_masks.__dict__["masks"] == 1)[1]
        pixel_indices = [d for i, d in enumerate(pixel_indices_full) if d in pixel_indices_full_bkg]            

        detector_indices_full = np.where(product.detector_masks.__dict__["masks"] == 1)[1]
        detector_indices_full_bkg = np.where(bkg.detector_masks.__dict__["masks"] == 1)[1]
        detector_indices = [d for i, d in enumerate(detector_indices_full) if d in detector_indices_full_bkg]   

        return pixel_indices, detector_indices

    @staticmethod
    def _livefrac(product):

        """
        Compute the livetime fraction and its uncertainty for a data product from its
        trigger counts.

        Maps trigger counts onto detectors, converts to a trigger rate using the
        integration time, and derives the livetime fraction via
        `get_livetime_fraction`. The uncertainty is estimated by propagating the
        trigger count uncertainty through the livetime fraction calculation and taking
        half the resulting spread in corrected counts.

        Parameters
        ----------
        product : ScienceData
            The data product from which triggers, trigger errors, and integration
            times are taken.

        Returns
        -------
        tuple of numpy.ndarray
            `livefrac` and `livefrac_error`, each broadcast to shape
            (n_times, n_detectors, 1, 1).
        """

        trigger_to_detector = STIX_INSTRUMENT.subcol_adc_mapping
        shape = product.data['counts'].shape

        if len(shape) < 4:

            counts = product.data['counts'].reshape(shape[0], 1, 1, shape[-1])

            # Need to average over the different triggers
            triggers = product.data["triggers"] / 16
            triggers_error = product.data["triggers"] / 16

            triggers_lower = triggers - triggers_error
            triggers_upper = triggers + triggers_error

            livefrac,_, _ = get_livetime_fraction(triggers / product.data["timedel"].to("s"))
            livefrac_lower,_, _ = get_livetime_fraction(triggers_lower / product.data["timedel"].to("s"))
            livefrac_upper,_, _ = get_livetime_fraction(triggers_upper / product.data["timedel"].to("s"))
            
            livefrac = livefrac.reshape(livefrac.shape + (1, 1, 1))
            livefrac_lower = livefrac_lower.reshape(livefrac_lower.shape + (1, 1, 1))
            livefrac_upper = livefrac_upper.reshape(livefrac_upper.shape + (1, 1, 1))

        else:

            counts = product.data['counts']

            triggers = product.data["triggers"][:, trigger_to_detector].astype(float)[...]

            triggers_error = product.data["triggers_comp_err"][:, trigger_to_detector].astype(float)[...]

            triggers_lower = triggers - triggers_error
            triggers_upper = triggers + triggers_error

            livefrac,_, _ = get_livetime_fraction(triggers / product.data["timedel"].to("s").reshape(-1, 1))
            livefrac_lower,_, _ = get_livetime_fraction(triggers_lower / product.data["timedel"].to("s").reshape(-1, 1))
            livefrac_upper,_, _ = get_livetime_fraction(triggers_upper / product.data["timedel"].to("s").reshape(-1, 1))
            
            livefrac = livefrac.reshape(livefrac.shape + (1, 1))
            livefrac_lower = livefrac_lower.reshape(livefrac_lower.shape + (1, 1))
            livefrac_upper = livefrac_upper.reshape(livefrac_upper.shape + (1, 1))

        counts_upper = (counts /  livefrac_upper)
        counts_lower = (counts /  livefrac_lower)

        livefrac_error = (counts_lower - counts_upper) / 2 


        return livefrac, livefrac_error

    @staticmethod
    def _return_spec_object(case,
                            sci_data,
                            flare_angle,
                            distance,
                            srm_dict,
                            bkg):

        """
        Build a `sunkit_spex` `Spectrum` object from selected science data for a given
        detector/pixel summation case.

        Sums counts and propagates uncertainties over the appropriate axes depending
        on `case` (whether detectors/pixels are collapsed or expanded, and whether the
        input is a single time bin or a sequence), computes a livetime-weighted mean
        exposure time, optionally adds an energy-dependent systematic uncertainty, and
        assembles the spectral response matrix (SRM), photon axis, and other metadata
        needed by the `Spectrum` object.

        Parameters
        ----------
        case : str
            One of 'spec_1D_detector_collapse', 'spec_sequence_detector_collapse',
            'spec_1D_detector_expand', or 'spec_sequence_detector_expand', selecting
            which axes to sum over.
        sci_data : tuple
            Tuple of (counts, counts_uncertainty, t_norm, e_norm, livefrac, ...,
            times, energies) for the (possibly detector/pixel-indexed) data to
            convert.print('shape_counts = ',np.shape(counts))
        flare_location : dict
            Flare location information, expected to contain 'stx' and 'hpc' keys.
        detector_indices : list or numpy.ndarray
            Detector indices used to build the spectrum (used for SRM/metadata
            purposes upstream; not directly summed here).
        pixel_indices : list or numpy.ndarray
            Pixel indices used to build the spectrum.
        flare_angle : astropy.units.Quantity
            Angle between the spacecraft and the flare location.
        distance : astropy.units.Quantityif len(shape) < 4:
            Distance from the spacecraft to the Sun.
        srm_dict : dict
            Dictionary containing the spectral response matrix ('srm'), photon axis
            ('ph_axis'), and geometric area ('geo_area'), as returned by
            `get_masked_srm`.
        systematic : bool
            If True, adds an energy-dependent systematic uncertainty (as a percentage
            of counts) in quadrature with the statistical uncertainty.

        Returns
        -------
        sunkit_spex.spectrum.Spectrum
            A 1D spectrum with counts, propagated uncertainty, spectral axis, and
            metadata (exposure time, geometric area, angle, distance, SRM, photon
            axis, time range).
        """

        counts, counts_uncertainity, t_norm, _, livefrac, _, elut_cor_fac, times_full, energies, _ = sci_data

        t_norm = t_norm.to(u.s)


        if energies["e_low"][0].value == 0:
            counts = counts[..., 1:]
            counts_uncertainity = counts_uncertainity[..., 1:]
            energies = energies[1:]
            elut_cor_fac = elut_cor_fac[1:]

        if np.isnan(energies["e_high"][-1].value):
            counts = counts[...,:-1]
            counts_uncertainity = counts_uncertainity[...,:-1]
            energies = energies[:-1]
            elut_cor_fac = elut_cor_fac[:-1]


        counts_axis = np.concatenate([energies["e_low"], [energies["e_high"][-1]]])

        # counts_uncertainity[counts < 0] = 0
        # counts[counts < 0] = 0

        shape = counts.shape

        if case == 'spec_1D_detector_collapse':

            # counts[counts  < 0] = 0
            print('ctsshape =',counts.shape)
            counts_final = np.nansum(counts,axis=(0,1,2))
            # print('ctsshape =',counts_final.shape)
            # counts[counts  < 0] = 0

            # counts_final[counts_final  < 0] = 0
            print('cts_err_shape = ',counts_uncertainity.shape)
            np.save('err_check_2.npy',np.array(np.sqrt(np.nansum(counts_uncertainity**2,axis=(0,1,2)))))

            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2,axis=(0,1,2)))

            t_norm = t_norm[:,None,None,None] * livefrac
            t_norm = t_norm.mean(axis=(1,2,3))

        elif case == 'spec_sequence_detector_collapse' or case == 'spec_1D_detector_expand':

            counts_final = np.nansum(counts,axis=(0,1))
            counts_final[counts_final  < 0] = 0
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2,axis=(0,1)))

            t_norm = t_norm * livefrac
            t_norm = t_norm.mean(axis=(0,1,2))
        
        elif case == 'spec_sequence_detector_expand':

            counts_final = np.nansum(counts,axis=(0))
            counts_final[counts_final  < 0] = 0
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2,axis=(0)))

            t_norm = t_norm * livefrac
            t_norm = t_norm.mean(axis=(0))

        e_low = energies["e_low"].value

        # if systematic:

        #     energy_conditions = [e_low < 7, (e_low < 10) & (e_low >= 7), e_low >= 10]
        #     percentage = [0.07, 0.05, 0.03]

        #     systematic_err_percentage = np.select(energy_conditions, percentage)
        #     systematic_err = systematic_err_percentage * counts_final

        #     np.save('stixpy_sys.npy',np.array(systematic_err.value))
        #     np.save('stixpy_err.npy',np.array(counts_uncertainity_final.value))

        #     counts_uncertainity_pu = PoissonUncertainty(np.sqrt(counts_uncertainity_final.value**2 + systematic_err.value**2) * u.ct)

        # else:

        counts_uncertainity_pu = PoissonUncertainty(counts_uncertainity_final)
        
        counts_spectral_axis = SpectralAxis(counts_axis, bin_specification="edges")

        meta = NDMeta()

        time_range_actual =  Time([(times_full - 0.5 * t_norm).value, 
                                (times_full + 0.5 * t_norm).value])

        ct_de = np.diff(counts_axis.value)

        srm = srm_dict["srm"] * ct_de[None, :]
        ph_ax_mids = srm_dict["ph_axis"][:-1] + 0.5 * np.diff(srm_dict["ph_axis"])

        index = np.where(ph_ax_mids <= 2.9)[0]

        print('ct_de = ',ct_de)

        # srm_trim = srm[index[-1] :]

        # ph_ax_bins = np.column_stack((srm_dict["ph_axis"][:-1], srm_dict["ph_axis"][1:]))

        # ph_ax_bins_trim = ph_ax_bins[index[-1] :]

        # ph_energies_trim = np.concatenate([ph_ax_bins_trim[:, 0], ph_ax_bins_trim[:, 1][-1:]])

        meta.add("exposure_time", np.sum(t_norm))
        meta.add("geo_area", srm_dict["geo_area"])
        meta.add("angle", flare_angle)
        meta.add("distance", distance)
        meta.add("srm", srm)
        # meta.add("srm", srm_trim)
        # meta.add("ph_axis", ph_energies_trim * u.keV)
        meta.add("ph_axis", srm_dict["ph_axis"] * u.keV)
        meta.add("time_range", time_range_actual)

        spec_1d = Spectrum(
            data=counts_final, uncertainty=counts_uncertainity_pu, spectral_axis=counts_spectral_axis, meta=meta
        )

        return spec_1d

    @staticmethod
    def _indices_expand_ranges(pairs, nest=True):
        """
        Expand a list of [start, end] pairs into a flat, inclusive list of integers.

        Parameters
        ----------
        pairs : list of list or tuple
            List of [start, end] pairs, e.g. [[1, 5], [9, 10]].

        Returns
        -------
        list of int
            Flat list of all integers covered by the given ranges, inclusive of both
            endpoints, e.g. [1, 2, 3, 4, 5, 9, 10].
        """
        result = []
        for pair in pairs:
            if nest:
                result.append(np.arange(pair[0], pair[1] + 1,1))
            else:
                result.extend(np.arange(pair[0], pair[1] + 1,1))
        return result


    @staticmethod
    def _srm_format_flat_or_ranges(indices, case):
        """
        Normalize detector/pixel indices given either as a flat list of ints or as a
        list of [start, end] range pairs into a single flat list of ints.

        Parameters
        ----------
        indices : list, tuple, or None
            Either a flat list of indices (e.g. [1, 2, 3, 4, 5]) or a list of
            [start, end] pairs (e.g. [[1, 5], [9, 10]]). If None, an empty list is
            returned.

        Returns
        -------
        list of int
            Flat list of indices. Flat input is returned unchanged (as a list);
            range-pair input is expanded via `_indices_expand_ranges`.
        """


        if indices is None:
            return []
     
        elif isinstance(indices[0], (int, np.integer)):

            return indices
        
        elif isinstance(indices[0], (list, np.ndarray)):

            
            indices = ScienceData._indices_expand_ranges(indices)

            if case in ('spec_1D_detector_collapse', 'spec_sequence_detector_collapse'):
                return [idx for ls in indices for idx in ls]
            
            elif case in ('spec_1D_detector_expand', 'spec_sequence_detector_expand'):
                return indices


    @staticmethod
    def _srm_det_pix_indices_format(detector_indices, pixel_indices, case):
        """
        Format detector and pixel indices into the flat-list form expected by
        `get_masked_srm`, using a different expansion rule depending on the
        detector/pixel summation case.

        Parameters
        ----------
        detector_indices : list, tuple, or None
            Detector indices, as either a flat list or a list of [start, end] pairs.
        pixel_indices : list, tuple, or None
            Pixel indices, as either a flat list or a list of [start, end] pairs.
        case : str
            One of 'spec_1D_detector_collapse', 'spec_sequence_detector_collapse',
            'spec_1D_detector_expanded', or 'spec_sequence_detector_expanded'.
            Collapse cases expand both `detector_indices` and `pixel_indices` via
            `_srm_format_flat_or_ranges`; expanded cases expand `detector_indices` via
            `_srm_format_single_or_range` and `pixel_indices` via
            `_srm_format_flat_or_ranges`.

        Returns
        -------
        tuple of list
            The formatted (`detector_indices`, `pixel_indices`) as flat lists of ints.
        """

        det_formatted = ScienceData._srm_format_flat_or_ranges(detector_indices,case)
        pix_formatted = ScienceData._srm_format_flat_or_ranges(pixel_indices,case)


        return det_formatted, pix_formatted

    @staticmethod
    def _get_sunkit_spex_spectrum(product,
                    detector_indices,
                    pixel_indices,
                    sci_data, 
                    flare_location,
                    flare_angle,
                    systematic,
                    detector_sum=True,
                    rcr=None,
                    bkg=False):
        """
        Convert selected science data into one or more `sunkit_spex` spectral
        products (a single `Spectrum`, an `NDCubeSequence` of spectra, or an
        `NDCollection` of spectra/sequences), depending on whether detectors are
        summed and whether the data spans a single time bin or multiple.

        If `detector_sum=True`, detectors are collapsed into a single spectrum (or a
        sequence of spectra over time). If `detector_sum=False`, a separate spectrum
        (or sequence of spectra) is produced for each detector, and results across
        detectors are combined into an `NDCollection`. In all cases, a masked spectral
        response matrix (SRM) is computed via `product.get_masked_srm` for the
        relevant detector/pixel combination.

        Parameters
        ----------
        product : ScienceData
            The science data product used for flare angle, distance, and SRM
            calculations.
        detector_indices : list or numpy.ndarray
            Detector indices included in the data.
        pixel_indices : list or numpy.ndarray
            Pixel indices included in the data.
        sci_data : tuple
            Tuple of (counts, counts_uncertainty, t_norm, e_norm, livefrac, ...,
            elut_cor_fac, times_full, energies) as returned by `_data_select` /
            `get_data`.
        flare_location : dict
            Flare location information, expected to contain 'stx' and 'hpc' keys.
        detector_sum : bool
            If True, sum over detectors to produce one spectrum (or sequence of
            spectra); if False, produce one spectrum (or sequence) per detector.
        rcr : optional
            Currently unused; reserved for RCR-state-aware processing.

        Returns
        -------
        sunkit_spex.spectrum.Spectrum or ndcube.NDCubeSequence or
        ndcube.NDCollection
            A single spectrum if the data has one time bin and detectors are summed;
            an `NDCubeSequence` of spectra if there are multiple time bins and
            detectors are summed; or an `NDCollection` (of spectra or sequences) keyed
            by detector index if `detector_sum=False`.
        """

        counts, counts_uncertainity, t_norm, e_norm, livefrac, _, elut_cor_fac, times_full, energies, rcr = sci_data

        if flare_location is not None:
            flare_location_stx = np.array([flare_location['stx'].Tx.value, flare_location['stx'].Ty.value])
        else:
            flare_location_stx = None

        if flare_angle is None:
            flare_angle = product._flare_angle(product,flare_location)

        distance = (product.meta["DSUN_OBS"] * u.m).to(u.AU)
        rcr_unique = np.unique(rcr)

        shape = np.shape(product.data['counts'])

        if len(shape) < 4:

            detector_indices = np.where(product.detector_masks.__dict__["masks"] == 1)[1]
            pixel_indices = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]
            detector_sum = True

        if detector_sum:

            if np.shape(counts)[0] == 1:

                case = 'spec_1D_detector_collapse'

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)

                srm_dict = product.get_masked_srm(flare_location=flare_location_stx,
                                            detector_indices_input=detector_indices_srm, 
                                            pixel_indices_input=pixel_indices_srm,
                                            rcr=rcr_unique[0])

                return ScienceData._return_spec_object(case,
                            sci_data,
                            flare_angle,
                            distance,
                            srm_dict,
                            bkg)

            else:

                case = 'spec_sequence_detector_collapse'
                
                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)

                rcr_unique =  np.unique(rcr)

                srm_dict_by_rcr = {
                    rcr_val: product.get_masked_srm(
                        flare_location=flare_location_stx,
                        detector_indices_input=detector_indices_srm,
                        pixel_indices_input=pixel_indices_srm,
                        rcr=rcr_val,
                    )
                    for rcr_val in rcr_unique
                }

                spec_list_working = []

                for i in range(np.shape(counts)[0]):

                    counts, counts_uncertainity, t_norm, e_norm, livefrac,_, elut_cor_fac, times_full, energies, rcr = sci_data

                    sci_data_indexed = ( counts[i,...], 
                                        counts_uncertainity[i,...], 
                                        t_norm[i,...], 
                                        e_norm, 
                                        livefrac[i,...],
                                        _, 
                                        elut_cor_fac, 
                                        times_full[i,...], 
                                        energies,
                                        rcr)

                    spec_1d =  ScienceData._return_spec_object(case,
                                sci_data_indexed,
                                flare_angle,
                                distance,
                                srm_dict_by_rcr[int(rcr[i][0])],
                                bkg)

                    spec_list_working.append(spec_1d)

                spec_sequence = NDCubeSequence(spec_list_working,
                            meta={"detector": "det1", "instrument": "STIX"},  
                            common_axis=0
                                )    

                return spec_sequence              

        else:
            
            if np.shape(counts)[0] == 1:

                case = 'spec_1D_detector_expand'

                spec_list_working = []


                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)

                for i in range(np.shape(counts)[1]):

                    srm_dict = product.get_masked_srm(flare_location=flare_location_stx,
                                            detector_indices_input=detector_indices_srm[i], 
                                            pixel_indices_input=pixel_indices_srm,rcr=rcr_unique)

                    counts, counts_uncertainity, t_norm, e_norm, livefrac,_, elut_cor_fac, times_full, energies, rcr = sci_data

                    sci_data_indexed = ( counts[:,i,...], 
                                        counts_uncertainity[:,i,...], 
                                        t_norm, 
                                        e_norm, 
                                        livefrac[:,i,...],
                                        _, 
                                        elut_cor_fac, 
                                        times_full, 
                                        energies,
                                        rcr)

                    spec_1d =  ScienceData._return_spec_object(case,
                                sci_data_indexed,
                                flare_angle,
                                distance,
                                srm_dict,
                                bkg)

                    spec_list_working.append((f"{detector_indices[i]}",spec_1d))
                
                spec_collection = NDCollection(spec_list_working,
                                               aligned_axes="all" ) 

                return spec_collection 

            else:
          
                spec_list_collection_working = []

                case = 'spec_sequence_detector_expand'

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)


                for i in range(np.shape(counts)[1]):

                    rcr_unique =  np.unique(rcr)

                    srm_dict_by_rcr = {
                        rcr_val: product.get_masked_srm(
                            flare_location=flare_location_stx,
                            detector_indices_input=detector_indices_srm[i],
                            pixel_indices_input=pixel_indices_srm,
                            rcr=rcr_val,
                        )
                        for rcr_val in rcr_unique
                    }

                    counts, counts_uncertainity, t_norm, e_norm, livefrac,_, elut_cor_fac, times_full, energies, rcr = sci_data

                    counts = counts[:,i,...]
                    counts_uncertainity = counts_uncertainity[:,i,...]
                    livefrac = livefrac[:,i,...]

                    spec_list_sequence_working = []

                    for j in range(np.shape(counts)[0]):

                        sci_data_indexed = (counts[j,...], 
                                            counts_uncertainity[j,...], 
                                            t_norm[j,...], 
                                            e_norm, 
                                            livefrac[j,...],
                                            _, 
                                            elut_cor_fac, 
                                            times_full[j,...], 
                                            energies,
                                            rcr)


                        spec_1d =  ScienceData._return_spec_object(case,
                                    sci_data_indexed,
                                    flare_angle,
                                    distance,
                                    srm_dict_by_rcr[int(rcr[j][0])],
                                    bkg)

                        spec_list_sequence_working.append(spec_1d)
                
                    spec_sequence = NDCubeSequence(spec_list_sequence_working,
                            meta={"detector": "det1", "instrument": "STIX"},  # sequence-level
                            common_axis=0
                                )
                
                    spec_list_collection_working.append((f'{detector_indices[i]}',spec_sequence))

                spec_collection = NDCollection(spec_list_collection_working,
                                               aligned_axes="all" ) 

                return spec_collection               
    
    @staticmethod
    def _flare_angle(product, flare_location):

        """
        Compute the angle between the spacecraft-to-Sun line and the spacecraft-to-flare
        line at the start of the product's time range.

        Parameters
        ----------
        product : ScienceData
            The data product whose time range is used to determine spacecraft
            pointing and position.
        flare_location : dict
            Flare location information, expected to contain an 'hpc' key giving the
            flare's helioprojective coordinates.

        Returns
        -------
        astropy.units.Quantity
            The angle between the spacecraft and the flare as seen from the Sun.
        """
        
        _, solo_xyz, _ = get_hpc_info(product.time_range.start, product.time_range.start)

        solo = HeliographicStonyhurst(*solo_xyz, obstime=product.time_range.center, representation_type="cartesian")

        flare_angle = flare_spacecraft_angle(solo,flare_location['hpc'])

        return flare_angle

    @staticmethod
    def _check_shadowing(product,detector_indices):

        """
        Check for possible pixel shadowing by comparing summed counts in the top vs.
        bottom pixel rows for the given detectors, and warn if either row's total
        exceeds the other's by more than a set tolerance.

        Only performs the check if both the top pixels (0-3) and bottom pixels (4-8)
        are present in the product's pixel mask.

        Parameters
        ----------
        product : ScienceData
            The data product whose counts and pixel masks are used for the check.
        detector_indices : list or numpy.ndarray
            Detector indices to include in the shadowing check.

        Returns
        -------
        None
            Issues a `warnings.warn` if the top-to-bottom or bottom-to-top count
            ratio (summed over the first 25 energy bins) exceeds the tolerance
            (1.05); otherwise returns nothing.
        """

        tolerance = 1.05

        pixels_top = np.arange(0,4)
        pixels_bot = np.arange(4,9)

        pixels_top_bot = np.concatenate([pixels_top,pixels_bot])      

        pixel_indices_full = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]

        counts = product.data["counts"]
        counts = counts[:, detector_indices, ...]

        if set(pixels_top_bot).issubset(set(pixel_indices_full)):

            rat_top_bot = counts[:,:,pixels_top,0:25] / counts[:,:,pixels_bot,0:25]
            rat_bot_top = counts[:,:,pixels_bot,0:25] / counts[:,:,pixels_top,0:25]

            if rat_top_bot >= tolerance:
                warnings.warn(f'Top pixel total 5% higher than bottom row with a ratio of {np.round(rat_top_bot,2)}. Possible pixel shadowing. Recommend using only top pixels for analysis.')

            elif rat_bot_top >= tolerance:
                warnings.warn(f'Bottom pixel total 5% higher than top row with a ratio of {np.round(rat_bot_top,2)}. Possible pixel shadowing. Recommend using only top pixels for analysis.')

    @staticmethod
    def _time_indices_format(time_indices,times,dt,rcr):

        """
        Normalize a user-supplied `time_indices` specification into a canonical list
        of integer indices or [start, end] integer pairs, and apply RCR-state
        consistency checks.

        Supports four input formats:
            - A flat list of integer indices (RCR state is checked with a warning).
            - A flat list of strings/`Time` objects, treated as bin edges and
            converted to [start, end] integer pairs via `_handle_datetime_strings`.
            - A list of [start, end] string/`Time` pairs, similarly converted.
            - A list of [start, end] integer pairs (RCR state is checked within each
            pair, raising an error on a within-pair change and warning on a
            between-pair difference).

        Parameters
        ----------
        time_indices : list
            The user-supplied time selection, in any of the supported formats.
        times : list or astropy.time.Time
            The full array of times associated with the data, used to resolve
            string/`Time` bin edges to integer indices.
        rcr : list
            Full RCR (rate control regime) state array, indexed the same as `times`.

        Returns
        -------
        list
            The time indices normalized to either a flat list of ints or a list of
            [start, end] integer pairs.

        Raises
        ------
        ValueError
            If the format of `time_indices` cannot be determined, or if nested
            pairs are not valid [start, end] integer or time pairs.
        """
        
        first = time_indices[0]

        if isinstance(first, int):
            ScienceData._rcr_warning(time_indices, rcr)
            return time_indices
        

        if isinstance(first, (str, Time)):

            if isinstance(first, (str, Time)) and not isinstance(time_indices[0], (list, tuple)):
                bins = [[time_indices[i], time_indices[i+1]] for i in range(len(time_indices) - 1)]
            else:
                bins = time_indices
            result = ScienceData._handle_datetime_strings(bins, times, dt)
            ScienceData._handle_nested_pairs(result, rcr)
            print(result )
            return result

        if isinstance(first, (list, tuple)):
            if isinstance(first[0], (str, Time)):

                result = ScienceData._handle_datetime_strings(time_indices, times, dt)
                ScienceData._handle_nested_pairs(result, rcr)
                return result
            if len(first) == 2 and all(isinstance(v, int) for v in first):
                ScienceData._handle_nested_pairs(time_indices, rcr)
                return time_indices
            raise ValueError(
                f"Nested lists must be [start, end] integer or time pairs, got: {first}"
            )

        raise ValueError(
            f"Cannot determine format from first element: {first!r}"
        )   



    @staticmethod
    def _rcr_warning(time_indices, rcr):

        """
        Warn if the RCR (rate control regime) state is not constant across a flat
        list of time indices.

        Parameters
        ----------
        time_indices : list of int
            Time indices to check for RCR state consistency.
        rcr : list
            Full RCR state array, indexed the same as the data's time axis.

        Returns
        -------
        None
            Issues a `warnings.warn` if any index in `time_indices` has a different
            RCR state than the first index; otherwise returns nothing.
        """

        first_rcr = rcr[time_indices[0]]
        for i in time_indices[1:]:
            if rcr[i] != first_rcr:
                warnings.warn(
                    f"RCR state change detected "
                    f"index {time_indices[0]} has RCR={first_rcr!r}, "
                    f"index {i} has RCR={rcr[i]!r}."
                    f"Use with caution!"
                )
        return None

    @staticmethod
    def _rcr_shift(rcr,counts):

        """
        Shift/align an RCR (state) array to match segment boundaries derived from
        discontinuities in the summed counts data.

        The method first identifies the indices where `rcr` changes value (state
        transitions). It then independently detects "jumps" in the total counts
        (summed over the last two axes of `counts`, using the 3rd slice along the
        last axis) that exceed a threshold of 1e4. These jump indices are used to
        redefine segment boundaries, and each segment is filled with the
        corresponding state value from `rcr`, producing a new array
        (`rcr_shifted`) that is aligned to the counts-derived segments rather
        than the original `rcr` transition points.

        This is useful when the original `rcr` state boundaries are believed to
        be misaligned (e.g., off by a few indices) relative to where the counts
        actually change, and you want to "shift" the state labels to match the
        true count-based transitions.

        Parameters
        ----------
        rcr : array-like
            1D array of state/category labels (e.g., integers) for each time
            index. If all values are <= 0, no shifting is performed and `rcr`
            is returned unchanged.
        counts : ndarray
            4D array of counts data with shape (time, ..., ..., channels).
            The last axis is expected to have at least 3 entries; index 2
            (the 3rd channel) is summed over axes (1, 2) to produce a 1D
            counts-per-time-index array used for jump detection.

        Returns
        -------
        ndarray
            If `np.max(rcr) > 0`: a 1D array the same length as `counts` along
            axis 0, where each segment (defined by detected counts jumps) is
            filled with the corresponding state value from `rcr`.
            Otherwise: the original `rcr` array, unmodified.

        Notes
        -----
        - Jump detection uses a fixed threshold (`> 1e4`) on the absolute
        difference between consecutive summed-counts values.
        - Consecutive detected jump indices that are adjacent (`curr == prev + 1`)
        are collapsed into a single boundary via `inds_clipped`.
        - This function assumes at least one jump is detected when
        `np.max(rcr) > 0`; if `inds` is empty, `inds_clipped = [inds[0]]` will
        raise an IndexError.
        """

        if np.max(rcr) > 0:

            rcr = np.asarray(rcr)

            diffs = rcr[1:] - rcr[:-1]
            q = np.where(diffs != 0)[0]

            index = np.concatenate(([0], q + 1))
            state = rcr[index]

            shape = counts.shape

            if len(shape) < 4:
                counts = counts.reshape(shape[0], 1, 1, shape[-1])

            cts_collapse = np.nansum(counts[:,:,:,2], axis=(1,2)).astype(np.int64)

            inds = []

            for i in range(len(cts_collapse)-1):

                if abs(cts_collapse[i] - cts_collapse[i+1]).value > 1e4:

                    inds.append(i+1)
        
            inds_clipped = [inds[0]]

            for prev, curr in zip(inds, inds[1:]):
                if curr != prev + 1:
                    inds_clipped.append(curr)

            length = counts.shape[0]

            # Length of each state segment
            range_vals = np.concatenate(([0], inds_clipped, [length]))
            segment_lengths = np.diff(range_vals)


            rcr_shift_lists = []
            for i in range(len(segment_lengths)):
                rg = np.full(segment_lengths[i],state[i])
                rcr_shift_lists.append(rg)
            
            rcr_shifted = np.concatenate(rcr_shift_lists)

            return rcr_shifted

        else:

            return rcr



    @staticmethod
    def _rcr_error(indices, rcr):
        """
        Raise a ValueError if the RCR (rate control regime) state is not uniform
        across the given indices.

        Parameters
        ----------
        indices : list of int
            Indices into `rcr` to check for state consistency.
        rcr : list
            Full RCR state array.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any index in `indices` has a different RCR value from the first.
        """
        if not indices:
            return None
        first_rcr = rcr[indices[0]]
        for i in indices[1:]:
            if rcr[i] != first_rcr:
                raise ValueError(
                    f"RCR state change detected. "
                    f"index {indices[0]} has RCR={first_rcr!r}, "
                    f"index {i} has RCR={rcr[i]!r}."
                )

    @staticmethod
    def _handle_datetime_strings(
        time_bin: list[list[str | Time]],
        times: list[str | Time],
        dt) -> list[list[int]]:

        """
        Convert a list of [start, end] time bins, given as strings or `Time` objects,
        into [start, end] integer index pairs by matching against a reference time
        array.

        Parameters
        ----------
        time_bin : list of list of str or astropy.time.Time
            List of [start, end] time bins.
        times : list of str or astropy.time.Time
            Reference time array to search for indices falling within each bin.

        Returns
        -------
        list of list of int
            For each input bin, the [first, last] index into `times` whose value
            falls within [start, end] (inclusive).

        Raises
        ------
        ValueError
            If any bin does not contain exactly 2 elements.
        """

        data_bin_start = times -  (0.5 * dt)
        data_bin_end = times + (0.5 * dt)

        results = []
        for n, bin in enumerate(time_bin):
            if len(bin) != 2:
                raise ValueError(
                    f"Each time bin must have exactly 2 elements [start, end], "
                    f"got {len(bin)} at index {n}."
                )

            bin_start = Time(bin[0])
            bin_end   = Time(bin[1])

            matched = [
                i for i, t in enumerate(times)
                if (bin_start <= data_bin_start[i]) and (data_bin_end[i] <= bin_end)
            ]

            results.append([matched[0], matched[-1]])
        
        return results

    @staticmethod
    def _handle_nested_pairs(time_indices: list[list[int]], rcr: list) -> list[list[int]]:
        """
        Handle Format 2: list of [start, end] integer pairs.

        - Raises an error if RCR state changes within any pair (between si and ei inclusive).
        - Warns if RCR state is consistent within each pair but differs between pairs.

        Args:
            time_indices: List of [start, end] integer pairs.
            rcr:          Full RCR state list.

        Returns:
            The original time_indices list unchanged.

        Raises:
            ValueError: If RCR state changes within any individual pair.
        """
        # Check within each pair
        for n, pair in enumerate(time_indices):
            indices_in_pair = list(range(pair[0], pair[1] + 1))
            ScienceData._rcr_error(
                indices_in_pair, rcr
            )

        # Warn if RCR state differs across pairs
        pair_representatives = [rcr[pair[0]] for pair in time_indices]
        if len(set(pair_representatives)) > 1:
            warnings.warn(
                f"RCR state differs across nested pairs: "
                f"{[f'pair {n}={r!r}' for n, r in enumerate(pair_representatives)]}."
            )

        return time_indices

    @staticmethod
    def _find_bin_index(start, end, e_low, e_high):
        """
        Find the index of the energy bin whose [e_low, e_high] range contains
        the given value.

        Parameters
        ----------
        start : float
            Lower energy bin edge.
        end   : float
            upper energy bin edge.
        e_low : numpy.ndarray
            Lower bin edges in keV.
        e_high : numpy.ndarray
            Upper bin edges in keV.

        Returns
        -------
        list
            List of indices to sum over.

        Raises
        ------
        ValueError
            If no bin contains the given value.
        """

        matches = np.where((e_low >= start) & (e_high <= end))[0]

        if matches.size == 0:
            raise ValueError(
                f"Energy range [{start} - {end}] keV does not fall within any product energy bin."
            )
        
        return [np.min(matches),np.max(matches)]


    @staticmethod
    def _energy_indices_from_flat_edges(values, e_low, e_high):
        """
        Convert a flat array of N energy values, treated as N-1 consecutive bin
        edges, into a list of [start_idx, end_idx] integer bin-index pairs.

        Parameters
        ----------
        values : numpy.ndarray
            Flat array of energy values in keV, e.g. [5, 10, 15, 25], treated
            as consecutive edges producing ranges (5-10), (10-15), (15-25).
        e_low : numpy.ndarray
            Lower bin edges of the product's energy bins, in keV.
        e_high : numpy.ndarray
            Upper bin edges of the product's energy bins, in keV.

        Returns
        -------
        list of list of int
            List of [start_idx, end_idx] integer bin-index pairs, one per
            consecutive edge pair in `values`.
        """
        pairs = []
        for i in range(len(values) - 1):
            idx = ScienceData._find_bin_index(values[i],values[i+1], e_low, e_high)
            pairs.append(idx)
        return pairs


    @staticmethod
    def _energy_indices_from_range_pairs(values, e_low, e_high):
        """
        Convert a 2D array of explicit [start, end] energy ranges into a list of
        [start_idx, end_idx] integer bin-index pairs.

        Parameters
        ----------
        values : numpy.ndarray
            2D array of [start, end] energy values in keV, e.g.
            [[5, 10], [15, 25]].
        e_low : numpy.ndarray
            Lower bin edges of the product's energy bins, in keV.
        e_high : numpy.ndarray
            Upper bin edges of the product's energy bins, in keV.

        Returns
        -------
        list of list of int
            List of [start_idx, end_idx] integer bin-index pairs, one per
            [start, end] pair in `values`.
        """
        pairs = []
        for start_val, end_val in values:
            idx = ScienceData._find_bin_index(start_val, end_val, e_low, e_high)
            pairs.append(idx)
        return pairs


    @staticmethod
    def _energy_indices_format(energy_indices, energies):
        """
        Convert an astropy Quantity energy selection into integer [start, end]
        bin-index pairs, matched against the product's energy bin edges.

        If `energy_indices` is not an astropy Quantity, it is returned unchanged
        (assumed to already be integer indices or index pairs).

        Two Quantity input formats are supported:
            - A flat 1D Quantity array of N energy values, treated as N-1
            consecutive bin edges, e.g. [5, 10, 15, 25]*u.keV produces ranges
            (5-10), (10-15), (15-25).
            - A 2D Quantity array (or list of pairs) giving explicit
            [start, end] energy ranges directly, e.g.
            [[5, 10], [15, 25]]*u.keV.

        In both cases, values are converted to keV and matched to the product
        energy bin whose [e_low, e_high] range contains them.

        Parameters
        ----------
        energy_indices : astropy.units.Quantity, list, numpy.ndarray, or None
            The user-supplied energy selection.
        energies : astropy.table.QTable
            The product's energy table, with "e_low" and "e_high" columns.

        Returns
        -------
        list of list of int or None
            Energy indices as a list of [start_idx, end_idx] integer pairs,
            suitable for use in `_data_select`. Returns None if `energy_indices`
            is None, or the original input unchanged if it is not a Quantity.

        Raises
        ------
        ValueError
            If a requested energy value does not fall within any product energy
            bin, or if the Quantity input is neither 1D nor 2D.
        """

        if not isinstance(energy_indices, u.Quantity):
            return energy_indices

        energy_indices = energy_indices.to(u.keV)

        e_low = energies["e_low"].value
        e_high = energies["e_high"].value

        if energy_indices.ndim == 1:
            return ScienceData._energy_indices_from_flat_edges(
                energy_indices.value, e_low, e_high
            )

        elif energy_indices.ndim == 2:
            return ScienceData._energy_indices_from_range_pairs(
                energy_indices.value, e_low, e_high
            )

        else:
            raise ValueError(
                "energy_indices given as a Quantity must be either 1D "
                "(flat list of bin edges) or 2D (list of [start, end] pairs)."
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
        elut_correction=True,
        sunkit_spex_spectrum=False,
        flare_location=None,
        flare_angle=None,
        bkg=None,
        sunkit_spex_systematic_error=False,
        sunkit_spex_detector_sum=True
    ):
    
        r"""
        Return the counts, errors, times, durations and energies for selected data,
        optionally applying livetime and ELUT corrections, background subtraction,
        and/or summing over time, energy, detector, or pixel axes.

        Parameters
        ----------
        vtype : str
            Type of value to return. Controls the normalisation:
                * 'c' - counts [ct]
                * 'cr' - count rate [ct/s]
                * 'dcr' - differential count rate [ct/(s keV)]
        time_indices : list or numpy.ndarray, optional
            If a 1xN array, treated as a mask; if a 2xN array (or list of
            [start, end] pairs), sums data between the given indices. Also accepts
            strings or `~astropy.time.Time` objects as bin edges, which are resolved
            against `self.times`. For example `time_indices=[0, 2, 5]` returns only
            the first, third, and sixth times, while `time_indices=[[0, 2], [3, 5]]`
            sums the data between those indices.
        energy_indices : list or numpy.ndarray, optional
            If a 1xN array, treated as a mask; if a 2xN array, sums data between the
            given indices. For example `energy_indices=[0, 2, 5]` returns only the
            first, third, and sixth energy bins, while `energy_indices=[[0, 2], [3, 5]]`
            sums the data between those indices.
        detector_indices : list, numpy.ndarray, or str, optional
            If a 1xN array, treated as a mask; if a 2xN array, sums data between the
            given indices. The special string "top24" selects a fixed set of 24
            detectors. If None, all detectors available in the product are used.
        pixel_indices : list or numpy.ndarray, optional
            If a 1xN array, treated as a mask; if a 2xN array, sums data between the
            given indices. If None, all pixels available in the product are used.
        sum_all_times : bool
            If True, sums all requested time bins into a single time bin.
        livetime_correction : bool
            If True, applies a livetime-fraction correction to the counts and
            propagates the associated uncertainty.
        elut_correction : bool
            If True, applies an energy lookup table (ELUT) correction factor to the
            counts.
        sunkit_spex_spectrum : bool
            If True, returns the data as one or more `sunkit_spex` spectral objects
            (see Returns) instead of raw arrays. When True, `vtype` is ignored and
            all data is returned as counts.
        flare_location : dict, optional
            Flare location information, required if `sunkit_spex_spectrum=True`.
            Expected to contain 'stx' (Helioprojective Tx/Ty) and 'hpc' (SkyCoord)
            keys.
        bkg : ScienceData, optional
            A background data product to subtract from the science data. If
            provided, `livetime_correction` and `elut_correction` are forced to True.
        sunkit_spex_systematic_error : bool
            If True (and `sunkit_spex_spectrum=True`), adds an energy-dependent
            systematic uncertainty in quadrature with the statistical uncertainty.
        sunkit_spex_detector_sum : bool
            If True (and `sunkit_spex_spectrum=True`), sums over detectors to
            produce a single spectrum (or sequence of spectra); if False, produces
            one spectrum (or sequence) per detector.

        Returns
        -------
        tuple or sunkit_spex.spectrum.Spectrum or ndcube.NDCubeSequence or ndcube.NDCollection
            If `sunkit_spex_spectrum=False` (default), returns a tuple of
            `(counts, counts_var, t_norm, e_norm, livefrac, livefrac_error,
            elut_cor_fac, times, energies)` normalised according to `vtype`.

            If `sunkit_spex_spectrum=True`, returns a `sunkit_spex` spectral product:
            a single `Spectrum` if there is one time bin and detectors are summed;
            an `NDCubeSequence` of spectra if there are multiple time bins and
            detectors are summed; or an `NDCollection` (of spectra or sequences)
            keyed by detector index if `sunkit_spex_detector_sum=False`.
        """

        # =====================================================
        # livetime
        # =====================================================
        rcr=self.rcr_shifted

        if energy_indices is not None:
            energy_indices = self._energy_indices_format(energy_indices,self.energies)

        if time_indices is not None:
            time_indices = self._time_indices_format(time_indices, self.times, self.durations, rcr)

        detector_indices, pixel_indices = self._indices_check(self,
                                                              detector_indices,
                                                              pixel_indices)

        if bkg:
            livetime_correction = True
            elut_correction = True

        if livetime_correction:

            livefraction_sci,livefraction_sci_error = self._livefrac(self)


            if bkg and isinstance(bkg, ScienceData):

                livefraction_bkg,livefraction_bkg_error = self._livefrac(bkg)

        else:
            livefraction_sci = None
            livefraction_sci_error = None

        # =====================================================
        # elut
        # =====================================================

        if elut_correction:

            _, _, elut_cor_fac = get_elut_correction(np.array(self.energies["channel"]), 
                                                       self)
        else:

            elut_cor_fac = None

        # =====================================================
        # data selection and background subtraction
        # =====================================================

        if not bkg:


            background_boolean = False

            print('ELUUUTTTTTT = ',elut_cor_fac)

            sci_data = self._data_select(self,
                                    detector_indices,
                                    pixel_indices,
                                    energy_indices,
                                    time_indices,
                                    livefraction_sci,
                                    livefraction_sci_error,
                                    elut_cor_fac,
                                    rcr,
                                    sum_all_times,
                                    sunkit_spex_systematic_error,
                                    sunkit_spex_detector_sum,
                                    bkg=background_boolean)

        else:

            background_boolean = True
            warnings.warn('For background subtraction elut_correction and livetime_correction set as True.')

            energy_indices_bkg = self._energies_bkg_sub(self,
                                                        bkg)
            

            pixel_indices_bkg, detector_indices_bkg = self._bkg_indices_check(self,
                                                                              bkg)

            print(detector_indices)
            print(pixel_indices)

            sci_data_all = self._bkg_sub(self,
                                    bkg,
                                    detector_indices,
                                    pixel_indices,
                                    sunkit_spex_detector_sum,
                                    detector_indices_bkg,
                                    pixel_indices_bkg,
                                    energy_indices_bkg,
                                    livefraction_sci,
                                    livefraction_sci_error,
                                    livefraction_bkg,
                                    livefraction_bkg_error,
                                    elut_cor_fac,
                                    rcr) 


            sci_data = self._data_select(sci_data_all,
                                    detector_indices,
                                    pixel_indices,
                                    energy_indices,
                                    time_indices,
                                    livefraction_sci,
                                    None,
                                    elut_cor_fac,
                                    rcr,
                                    sum_all_times,
                                    sunkit_spex_systematic_error,
                                    sunkit_spex_detector_sum,
                                    bkg=background_boolean)
 
        # =====================================================
        # data_sum
        # =====================================================

        if sunkit_spex_spectrum:
           
            warnings.warn('As sunkit_spex_spectrum = True, all data will be output as counts.' \
                        'Normalisation selection (vtype) will not be taken into account.')

            sunkit_spex_spectrum = self._get_sunkit_spex_spectrum(self,
                                                        detector_indices,
                                                        pixel_indices,
                                                        sci_data,
                                                        flare_location,
                                                        flare_angle,
                                                        systematic=sunkit_spex_systematic_error,
                                                        detector_sum=sunkit_spex_detector_sum,
                                                        rcr=rcr,
                                                        bkg=background_boolean)

            return sunkit_spex_spectrum
        
        else:
            
            counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr = sci_data
            
            e_norm = e_norm[np.newaxis, np.newaxis, np.newaxis, :]  
            t_norm = t_norm[:, np.newaxis, np.newaxis, np.newaxis].to(u.s) 
 
            if vtype == "c":
                norm = 1

            elif vtype == "cr":
                norm = 1 / t_norm
                if livetime_correction:
                    norm = 1 / (t_norm*livefrac)

            elif vtype == "dcr":
                norm = 1 / (e_norm * t_norm)
                if elut_correction:
                    norm = elut_cor_fac / (e_norm  * t_norm)
                elif livetime_correction:
                    norm = 1 / (e_norm  * t_norm * livefrac)
                elif livetime_correction and elut_correction:
                    norm = elut_cor_fac / (e_norm  * t_norm * livefrac)

            else:
                raise ValueError("vtype must be one of 'c', 'cr', 'dcr'.")
            
            counts = counts * norm
            
            if livetime_correction:
                counts_var = ScienceData._livetime_uncertainty(counts_var,livefrac_error) * livefrac
            
            counts_var = counts_var * norm

            return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr

        
    def get_masked_srm(self, flare_location, detector_indices_input, pixel_indices_input, rcr):

        """
        Build a spectral response matrix (SRM) masked/scaled for a given flare
        location and set of detectors and pixels.

        Loads the detector response matrix (DRM) and its photon/count energy grids
        from the on-disk calibration file, clips the DRM to the energy edges of the
        current product, applies grid and detector transmission corrections for the
        given flare location and detectors, rebins the DRM onto the product's count
        energy bins, and scales by the total active pixel area.

        Parameters
        ----------
        flare_location : array-like
            Flare location in Helioprojective Tx/Ty coordinates, e.g.sks_spec[0].meta

        Returns
        -------
        dict
            Dictionary with keys:
                - "srm": the masked, rebinned spectral response matrix.
                - "ph_axis": the (clipped) photon energy axis bin edges.
                - "geo_area": the total geometric area (cm^2) for the selected
                detectors and pixels.
        """

        HERE = Path(__file__).parent          
        ROOT = HERE.parent.parent            
        PATH_DRM = ROOT / "config" / "data" / "detector" / 'stx_detector_response_matrix.fits.gz'

        drm = np.array(Table.read(PATH_DRM,hdu=1)['DRM'])
        ph_energies = np.array(Table.read(PATH_DRM,hdu=2)['DRM'])
        ct_energies = np.array(Table.read(PATH_DRM,hdu=3)['DRM'])
    
        energies = self.energies
        # energy_masks = self.energy_masks.energy_mask
        # energy_exclude = [0, 31]

        # energy_final_index_values = [i for i, v in enumerate(energy_masks) if v != 0 and i not in energy_exclude]

        mask_emids_test = ~np.any(np.isclose(ph_energies[:, None], ct_energies[None, :], atol=1e-5), axis=1)

        # ph_energies_original = ph_energies[mask_emids_test]
        # e_mids_original = ph_energies_original[:-1] + (np.diff(ph_energies_original) /2)
        # test = get_grid_transmission(e_mids_original, detector_indices_input, flare_location)


        e_low = np.array(energies["e_low"])

        if e_low[0] == 0:
            e_low = e_low[1:]

        e_high = np.array(energies["e_high"])

        if e_high[-2] == 150:
            e_edges = e_low
            ct_e_diff = np.diff(e_edges)
        else:
            e_edges = np.concatenate([e_low, [e_high[-1]]])
            ct_e_diff = np.diff(e_edges)

        epsilon = 1e-4

        print(ct_e_diff)

        mask_not_in_e = ~np.isclose(ct_energies[:, None], e_edges[None, :], atol=epsilon).any(axis=1)

        values_to_remove = ct_energies[mask_not_in_e]

        indices_to_remove = np.where(
            np.isclose(ph_energies[:, None], values_to_remove[None, :], atol=epsilon).any(axis=1)
        )[0]

        print('INDICES = ',indices_to_remove)
        print('MASK = ',mask_not_in_e)

        drm_clipped = np.delete(drm, indices_to_remove, axis=0)
        drm_clipped = np.delete(drm_clipped, indices_to_remove, axis=1)

        ph_energies_clipped = np.delete(ph_energies, indices_to_remove)

        ph_e_diff = np.diff(ph_energies_clipped)

        pixel_areas_full = STIX_INSTRUMENT.pixel_config["Area"].to("cm2")
        
        pixel_areas = pixel_areas_full[pixel_indices_input].value


        area_scale = np.size(detector_indices_input) * np.sum(pixel_areas)

        energy_widths = np.diff(ph_energies_clipped)

        e_mids = ph_energies_clipped[:-1] + (energy_widths / 2)

        trans = Transmission()

        if rcr == 0:
            tot_trans = trans.get_transmission(energies=e_mids * u.keV)
        else:
            tot_trans = trans.get_transmission(energies=e_mids * u.keV,
                                                attenuator=True)     

        rcr_state_all = np.array([0.8096, 0.80961, 0.4048, 0.2024, 0.1012, 0.0396, 0.0198, 0.0099])
        pixel_indices_input_rcr = np.arange(0,12,1)

        rcr_state = rcr_state_all[int(rcr)]  
        rcr_factor = rcr_state / np.sum(pixel_areas_full[pixel_indices_input_rcr].value)

        attenuation = np.zeros(len(tot_trans["det-1"]))

        if np.size(detector_indices_input) !=1:
            for i, det in enumerate(detector_indices_input):
                attenuation += tot_trans[f"det-{det}"]
        else:
            attenuation += tot_trans[f"det-{detector_indices_input}"]

        attenuation = attenuation / np.size(detector_indices_input)

        print('att_shape=',attenuation.shape)
        print('att=',attenuation)
        np.save('/home/jmitchell/Documents/SOLER/case_studies/240310/data/reduced_data_subc/py_trans.npy',attenuation)

        drm_clipped = drm_clipped * ph_e_diff[None, :] * attenuation[:, None]

        drm_new = []

        for j in range(np.shape(drm_clipped)[0]):
            working = []

            for i in range(len(e_edges) - 1):
                indices_sum = np.where((ph_energies_clipped >= e_edges[i]) & (ph_energies_clipped < e_edges[i + 1]))[0]

                tot = drm_clipped[j, indices_sum].sum(axis=0)

                working.append(tot)

            drm_new.append(working)

        drm_new = np.array(drm_new)

        
        grid_transmission = get_grid_transmission(e_mids, detector_indices_input, flare_location)



        # if flare_location is not None:
        grid_transmission = grid_transmission.mean(axis=1)

        np.save('/home/jmitchell/Documents/SOLER/case_studies/240310/data/reduced_data_subc/pyu_grid_fac.npy',grid_transmission)
        #     print('gts_shape = ',grid_transmission)
        # else:
        #     grid_transmission = grid_transmission[energy_final_index_values]
        
        srm = (drm_new * grid_transmission[:, None]) / ct_e_diff[None, :]

        print('drm.shape =', drm.shape)
        print('ph_energies.shape =', ph_energies.shape)
        print('ct_energies.shape =', ct_energies.shape)
        print('e_edges.shape =', e_edges.shape)
        print('drm_new.shape =', np.array(drm_new).shape)

        return {"srm": srm, "ph_axis": ph_energies_clipped, "geo_area": area_scale*rcr_factor}
    

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


def calc_count_rate(dat):

    rate, rate_err, _, t_norm_cs, energies, _, cor = dat

    de = np.array(energies["e_high"] - energies["e_low"])

    rate = np.array(rate)
    rate_err = np.array(rate_err)

    t_norm = t_norm_cs.to(u.s).value

    counts_kev = rate * t_norm_cs
    counts_err_kev = rate_err * t_norm_cs

    counts = counts_kev * de
    counts_err = counts_err_kev * de

    result_count_rate = counts / t_norm
    result_count_rate_err = counts_err / t_norm

    result_count_rate = result_count_rate[:, :, :8, :].sum(axis=(1, 2)) * cor
    result_count_rate_err = result_count_rate_err[:, :, :8, :].sum(axis=(1, 2)) * cor

    return result_count_rate, result_count_rate_err
