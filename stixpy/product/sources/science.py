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

        counts, errors, timedeltas, _, _, _, _, times, energies = self.get_data(
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

# counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies
        counts, errors, timedeltas, _, _, _, _, times, energies = self.get_data(
            vtype=vtype,
            detector_indices=detector_indices,
            pixel_indices=pixel_indices,
            time_indices=time_indices,
            energy_indices=energy_indices,
            livetime_correction=False,
            elut_correction=False
        )

        print(errors.unit)

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
    def rcr(self):
        """
        The rcr state
        """
        return self.data["rcr"]


    @staticmethod
    def _indices_check(product, detector_indices, pixel_indices, sunkit_spex_spectrum):

        # --- Detector indices ---
        if detector_indices is not None:


            detector_indices_working = detector_indices

            if detector_indices_working == "top24":
                detector_indices_working = np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
                detector_indices = detector_indices_working
            else:
                detector_indices_full = np.where(product.detector_masks.__dict__["masks"] == 1)[1]

                if sunkit_spex_spectrum:
                    if np.ndim(detector_indices_working) != 1:
                        raise ValueError(f"If sunkit_spex_spectrum=True detector_indices must be a 1D list")

                if np.ndim(detector_indices_working) == 2:
                    # [[start, end], ...] range format
                    for start, end in detector_indices_working:
                        requested = np.arange(start, end + 1)
                        missing = np.setdiff1d(requested, detector_indices_full)
                        if missing.size > 0:
                            warnings.warn(f"Detector indices {missing.tolist()} in range [{start}, {end}] are not available in the product.")
                else:
                    # flat list of individual indices
                    missing = np.setdiff1d(detector_indices_working, detector_indices_full)
                    if missing.size > 0:
                        warnings.warn(f"The following detector indices are not available in the product: {missing.tolist()}")
            
        else:
            detector_indices = np.where(product.detector_masks.__dict__["masks"] == 1)[1]



        # --- Pixel indices ---
        if pixel_indices is not None:
            pixel_indices_full = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]

            if sunkit_spex_spectrum:
                if np.ndim(pixel_indices) != 1:
                    raise ValueError(f"If sunkit_spex_spectrum=True pixel_indices must be a 1D list")
                
            if np.ndim(pixel_indices) == 2:
                # [[start, end], ...] range format
                for start, end in pixel_indices:
                    requested = np.arange(start, end + 1)
                    missing = np.setdiff1d(requested, pixel_indices_full)
                    if missing.size > 0:
                        warnings.warn(f"Pixel indices {missing.tolist()} in range [{start}, {end}] are not available in the product.")
            else:
                # flat list of individual indices
                missing = np.setdiff1d(pixel_indices, pixel_indices_full)
                if missing.size > 0:
                    warnings.warn(f"The following pixel indices are not available in the product: {missing.tolist()}")

        else:
            pixel_indices = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]

        return np.array(detector_indices), np.array(pixel_indices)

    @staticmethod
    def _livetime_uncertainty(counts_var, livefrac, livefrac_error):

        if livefrac_error is not None:
        
            counts_var_lvtcorr = np.sqrt(((counts_var**2).value) + (livefrac_error.value**2))

            return counts_var_lvtcorr * u.ct
        
        else:

            return counts_var

    @staticmethod
    def _data_select(product, 
                    detector_indices,
                    pixel_indices,
                    energy_indices,
                    time_indices,
                    livefrac,
                    livefrac_error,
                    elut_cor_fac,
                    sum_all_times):
        

        if isinstance(product,ScienceData):

            e_norm = product.dE
            counts = product.data["counts"]
            try:
                counts_var = product.data["counts_comp_err"] ** 2
            except KeyError:
                counts_var = product.data["counts_comp_comp_err"] ** 2

            counts_var = np.sqrt(counts.value + counts_var.value) *u.ct

            t_norm = product.data["timedel"]
            shape = counts.shape
            times = product.times
            energies = product.energies

        else:

            counts, counts_var, t_norm, e_norm, livefrac, elut_cor_fac, times, energies = product



        if detector_indices is not None:  

            if detector_indices.ndim == 1:
                detector_mask = np.full(32, False)
                detector_mask[detector_indices] = True
                counts = counts[:, detector_mask, ...]
                counts_var = counts_var[:, detector_mask, ...]
                # t_norm = t_norm[:, detector_mask,:,:]
                if livefrac is not None:
                    livefrac = livefrac[:,detector_mask,:,:]
                if livefrac_error is not None:
                    livefrac_error = livefrac_error[:,detector_mask,:,:]                

            if detector_indices.ndim == 2:
                counts = np.hstack(
                    [np.sum(counts[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices]
                )

                counts_var = np.concatenate(
                    [np.sum(counts_var[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices],
                    axis=1,
                    )

                if livefrac is not None:                
                    livefrac = np.concatenate(
                    [np.mean(livefrac[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices],
                    axis=1,
                    )

                if livefrac_error is not None:                
                    livefrac_error = np.concatenate(
                    [np.sqrt(np.mean(livefrac_error[:, dl : dh + 1, ...]**2, axis=1, keepdims=True)) for dl, dh in detector_indices],
                    axis=1,
                    )

        if pixel_indices is not None:

            pixel_indices = np.asarray(pixel_indices)
            if pixel_indices.ndim == 1:
                pixel_mask = np.full(12, False)
                pixel_mask[pixel_indices] = True
                num_pixels = counts.shape[2]
                counts = counts[..., pixel_mask[:num_pixels], :]
                counts_var = counts_var[..., pixel_mask[:num_pixels], :]
                if livefrac is not None:
                    livefrac = livefrac[:,:,pixel_mask[:num_pixels],:]
                if livefrac_error is not None:
                    livefrac_error = livefrac_error[:,:,pixel_mask[:num_pixels],:]      


            if pixel_indices.ndim == 2:
                counts = np.concatenate(
                    [np.sum(counts[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices], axis=2
                )

                counts_var = np.concatenate(
                    [np.sum(counts_var[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices], axis=2
                )

                if livefrac is not None:                
                    livefrac = np.concatenate(
                    [np.mean(livefrac[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in detector_indices],
                    axis=2,
                    )

                if livefrac_error is not None:                
                    livefrac_error = np.concatenate(
                    [np.sqrt(np.mean(livefrac_error[..., pl : ph + 1, :]**2, axis=2, keepdims=True)) for pl, ph in detector_indices],
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
                    [np.sum(counts_var[..., el : eh + 1], axis=-1, keepdims=True) for el, eh in energy_indices], axis=-1
                )

        
                e_norm = np.hstack([(energies["e_high"][eh] - energies["e_low"][el]) for el, eh in energy_indices])

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

        if time_indices is not None:
            if isinstance(time_indices[0], (str, Time)):
                time_indices=None

        if time_indices is not None:
            time_indices = np.asarray(time_indices)
            if time_indices.ndim == 1:
                time_mask = np.full(times.shape, False)
                time_mask[time_indices] = True
                counts = counts[time_mask, ...]
                counts_var = counts_var[time_mask, ...]
                t_norm = t_norm[time_mask]
                if livefrac is not None:
                    livefrac = livefrac[time_mask, ...]
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

                if livefrac is not None:
                    livefrac = np.vstack([np.mean(livefrac[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices])

                if livefrac_error is not None:
                    livefrac_error = np.vstack([np.sqrt(np.mean(livefrac[tl : th + 1, ...]**2, axis=0, keepdims=True)) for tl, th in time_indices])


                counts_var = np.vstack(
                    [np.sum(counts_var[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices]
                )
                t_norm = dt

                if sum_all_times and len(new_times) > 1:
                    counts = np.sum(counts, axis=0, keepdims=True)
                    counts_var = np.sum(counts_var, axis=0, keepdims=True)
                    t_norm = np.sum(dt)
    

        return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies
    
    @staticmethod
    def _bkg_sub(product,
                bkg,
                detector_indices_bkg,
                pixel_indices_bkg,
                energy_indices_bkg,
                livefrac,
                livefrac_error,
                livefrac_bkg,
                livefrac_error_bkg,
                elut_cor_fac): 
        
        e_norm = product.dE
        counts = product.data["counts"]
        try:
            counts_var = (product.data["counts_comp_err"].value ** 2)*u.ct
        except KeyError:
            counts_var =  (product.data["counts_comp_comp_err"].value ** 2)*u.ct
        
        counts_var = np.sqrt(counts + counts_var) 

        t_norm = product.data["timedel"]
        times = product.times
        energies = product.energies

        counts_var = ScienceData._livetime_uncertainty(counts_var,livefrac,livefrac_error)    


        e_norm_bkg = bkg.dE[energy_indices_bkg]
        counts_bkg = bkg.data["counts"]
        try:
            counts_var_bkg = bkg.data["counts_comp_err"] ** 2
        except KeyError:
            counts_var_bkg = bkg.data["counts_comp_comp_err"] ** 2

        counts_var_bkg = ScienceData._livetime_uncertainty(counts_var_bkg,livefrac_bkg,livefrac_error_bkg) 

        counts_bkg = counts_bkg[:,detector_indices_bkg,:,:]
        counts_bkg = counts_bkg[:,:,pixel_indices_bkg,:]
        counts_bkg = counts_bkg[:,:,:,energy_indices_bkg]

        counts_var_bkg = counts_var_bkg[:,detector_indices_bkg,:,:]
        counts_var_bkg = counts_var_bkg[:,:,pixel_indices_bkg,:]
        counts_var_bkg = counts_var_bkg[:,:,:,energy_indices_bkg]

        t_norm_bkg = bkg.data["timedel"]
        times_bkg = bkg.times
        energies_bkg = bkg.energies[energy_indices_bkg]


        e_norm_corr = e_norm 
        e_norm_corr_bkg = e_norm_bkg 

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
        counts_var_lvtcorr_scaled_bkg = (counts_var_lvtcorr_bkg/ t_norm_bkg.mean()) * t_norm.reshape(len(t_norm), 1,1,1)


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


        eff_livefrac = np.nansum(spec_in_lvt,axis=(3)) /  np.nansum(spec_in_corr_lvt,axis=(3)) 


        spec_in_final = spec_in_corr * eff_livefrac[...,None]
        spec_in_err_final = spec_in_err * eff_livefrac[...,None]

        counts = spec_in_final
        counts_var = spec_in_err_final
        livefrac = eff_livefrac[:, :, :, np.newaxis]


        return counts, counts_var, t_norm, e_norm, livefrac, elut_cor_fac, times, energies
                                                                       
    @staticmethod
    def _energies_bkg_sub(product,bkg):
         
         _, _, indices_sub = np.intersect1d(product.energies["e_low"], bkg.energies["e_low"], return_indices=True)

         return indices_sub

    @staticmethod
    def _bkg_indices_check(product, bkg):
        
        pixel_indices_full = np.where(product.pixel_masks.__dict__["masks"] == 1)[1]
        pixel_indices_full_bkg = np.where(bkg.pixel_masks.__dict__["masks"] == 1)[1]
        pixel_indices = [d for i, d in enumerate(pixel_indices_full) if d in pixel_indices_full_bkg]            

        detector_indices_full = np.where(product.detector_masks.__dict__["masks"] == 1)[1]
        detector_indices_full_bkg = np.where(bkg.detector_masks.__dict__["masks"] == 1)[1]
        detector_indices = [d for i, d in enumerate(detector_indices_full) if d in detector_indices_full_bkg]   

        return pixel_indices, detector_indices

    @staticmethod
    def _livefrac(product):

        trigger_to_detector = STIX_INSTRUMENT.subcol_adc_mapping
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

        counts_upper = (product.data["counts"] /  livefrac_upper)
        counts_lower = (product.data["counts"] /  livefrac_lower)

        livefrac_error = (counts_lower - counts_upper) / 2        

        return livefrac, livefrac_error


    @staticmethod
    def _return_spec_object(case,
                            sci_data,
                            flare_location,
                            detector_indices,
                            pixel_indices,
                            event_time_range,
                            flare_angle,
                            distance,
                            srm_dict):

        counts, counts_uncertainity, t_norm, _, livefrac,_, _, times_full, energies = sci_data

        t_norm = t_norm.to(u.s)

        counts_axis = np.concatenate([energies["e_low"], [energies["e_high"][-1]]])

        counts_uncertainity[counts < 0] = 0
        counts[counts < 0] = 0


        if case == 'spec_1D_detector_collapse':

            counts_final = np.nansum(counts,axis=(0,1,2))
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2,axis=(0,1,2)))

            # counts_final = np.nansum(counts,axis=0)
            # counts_uncertainity_final= np.sqrt(np.nansum(counts_uncertainity**2,axis=0))

            t_norm = t_norm[:,None,None,None] * livefrac
            t_norm = t_norm.mean(axis=(1,2,3))

        elif case == 'spec_sequence_detector_collapse' or case == 'spec_1D_detector_expand':

            counts_final = np.nansum(counts,axis=(0,1))
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2,axis=(0,1)))

            t_norm = t_norm * livefrac
            t_norm = t_norm.mean(axis=(0,1,2))
        
        elif case == 'spec_sequence_detector_expand':

            counts_final = np.nansum(counts,axis=(0))
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2,axis=(0)))

            t_norm = t_norm * livefrac
            t_norm = t_norm.mean(axis=(0))


        e_low = energies["e_low"].value

        energy_conditions = [e_low < 7, (e_low < 10) & (e_low >= 7), e_low >= 10]
        percentage = [0.07, 0.05, 0.03]

        systematic_err_percentage = np.select(energy_conditions, percentage)

        systematic_err = systematic_err_percentage * counts_final

        counts_err_final_final = np.sqrt(counts_uncertainity_final.value**2 + systematic_err.value**2) * u.ct

        counts_uncertainity_pu = PoissonUncertainty(counts_err_final_final)
        counts_spectral_axis = SpectralAxis(counts_axis, bin_specification="edges")

        meta = NDMeta()

        ct_de = np.diff(counts_axis.value)

        srm = srm_dict["srm"] * ct_de[None, :]
        ph_ax_mids = srm_dict["ph_axis"][:-1] + 0.5 * np.diff(srm_dict["ph_axis"])

        index = np.where(ph_ax_mids <= 3.7)[0]

        srm_trim = srm[index[-1] :]

        ph_ax_bins = np.column_stack((srm_dict["ph_axis"][:-1], srm_dict["ph_axis"][1:]))

        ph_ax_bins_trim = ph_ax_bins[index[-1] :]

        ph_energies_trim = np.concatenate([ph_ax_bins_trim[:, 0], ph_ax_bins_trim[:, 1][-1:]])


        meta.add("exposure_time", np.sum(t_norm))
        meta.add("geo_area", srm_dict["geo_area"])
        meta.add("angle", flare_angle)
        meta.add("distance", distance)
        meta.add("srm", srm_trim)
        meta.add("ph_axis", ph_energies_trim * u.keV)
        meta.add("time_range", event_time_range)


        spec_1d = Spectrum(
            data=counts_final, uncertainty=counts_uncertainity_pu, spectral_axis=counts_spectral_axis, meta=meta
        )

        return spec_1d



    @staticmethod
    def _srm_expand_ranges(pairs):
        """Expand a list of [start, end] pairs into a flat inclusive list of ints."""
        result = []
        for start, end in pairs:
            result.extend(range(start, end + 1))
        return result

    @staticmethod
    def _srm_is_list_of_pairs(indices):
        """Check whether indices is a list of [start, end] pairs (each length 2)."""
        return len(indices) > 0 and all(
            isinstance(item, (list, tuple)) and len(item) == 2 for item in indices
        )

    @staticmethod
    def _srm_format_flat_or_ranges(indices):
        """
        Format indices given as either:
            - a flat list of ints, e.g. [1, 2, 3, 4, 5]  -> returned unchanged
            - a list of [start, end] pairs, e.g. [[1, 5], [9, 10]]
              -> expanded to [1, 2, 3, 4, 5, 9, 10]
        """
        if indices is None:
            return []

        if ScienceData._srm_is_list_of_pairs(indices):
            return ScienceData._srm_expand_ranges(indices)

        return list(indices)

    @staticmethod
    def _srm_format_single_or_range(indices):
        """
        Format indices given as either:
            - a single-entry list, e.g. [3]  -> returned unchanged
            - a two-entry list [start, end], e.g. [1, 5]
              -> expanded to [1, 2, 3, 4, 5]
            - anything else -> returned unchanged as a flat list
        """
        if indices is None:
            return []

        if len(indices) == 1:
            return list(indices)

        if len(indices) == 2:
            start, end = indices
            return list(range(start, end + 1))

        return list(indices)

    @staticmethod
    def _srm_det_pix_indices_format(detector_indices, pixel_indices, case):
        """
        Both detector_indices and pixel_indices can be:
            - a flat list of ints, e.g. [1, 2, 3, 4, 5]  -> returned unchanged
            - a list of [start, end] pairs, e.g. [[1, 5], [9, 10]]
              -> expanded to [1, 2, 3, 4, 5, 9, 10]
        """
        if case == 'spec_1D_detector_collapse' or 'spec_sequence_detector_collapse':
            det_formatted = ScienceData._srm_format_flat_or_ranges(detector_indices)
            pix_formatted = ScienceData._srm_format_flat_or_ranges(pixel_indices)
        
        if case =='spec_1D_detector_expanded' or 'spec_sequence_detector_expanded':
            det_formatted = ScienceData._srm_format_single_or_range(detector_indices)
            pix_formatted = ScienceData._srm_format_flat_or_ranges(pixel_indices)

        return det_formatted, pix_formatted


    @staticmethod
    def _get_sunkit_spex_spectrum(product,
                    detector_indices,
                    pixel_indices,
                    sci_data, 
                    event_time_range,
                    flare_location,
                    detector_sum=True,
                    rcr=None):
        
        counts, counts_uncertainity, t_norm, e_norm, livefrac, _, elut_cor_fac, times_full, energies = sci_data


        flare_location_stx = np.array([flare_location['stx'].Tx.value, flare_location['stx'].Ty.value])
        flare_angle = product._flare_angle(product,flare_location)
        distance = (product.meta["DSUN_OBS"] * u.m).to(u.AU)

        print('det_sum = ',detector_sum)

        if detector_sum:

            if np.shape(counts)[0] == 1:

                case = 'spec_1D_detector_collapse'

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)

                srm_dict = product.get_masked_srm(flare_location=flare_location_stx,
                                            detector_indices_input=detector_indices_srm, 
                                            pixel_indices_input=pixel_indices_srm)


                return ScienceData._return_spec_object(case,
                            sci_data,
                            flare_location,
                            detector_indices,
                            pixel_indices,
                            event_time_range,
                            flare_angle,
                            distance,
                            srm_dict)
            
            else:

                case = 'spec_sequence_detector_collapse'

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)

                srm_dict = product.get_masked_srm(flare_location=flare_location_stx,
                                            detector_indices_input=detector_indices_srm, 
                                            pixel_indices_input=pixel_indices_srm)

                spec_list_working = []

                for i in range(np.shape(counts)[0]):

                    counts, counts_uncertainity, t_norm, e_norm, livefrac,_, elut_cor_fac, times_full, energies = sci_data


                    sci_data_indexed = ( counts[i,...], 
                                        counts_uncertainity[i,...], 
                                        t_norm[i,...], 
                                        e_norm, 
                                        livefrac[i,...],
                                        _, 
                                        elut_cor_fac, 
                                        times_full[i,...], 
                                        energies)

    

                    spec_1d =  ScienceData._return_spec_object(case,
                                sci_data_indexed,
                                flare_location,
                                detector_indices,
                                pixel_indices,
                                event_time_range,
                                flare_angle,
                                distance,
                                srm_dict)

                    spec_list_working.append(spec_1d)
                
                spec_sequence = NDCubeSequence(spec_list_working,
                            meta={"detector": "det1", "instrument": "STIX"},  # sequence-level
                            common_axis=0
                                )    

                return spec_sequence              

        else:
            
            if np.shape(counts)[0] == 1:

                case = 'spec_1D_detector_expand'

                spec_list_working = []

                print(np.shape(counts))

                for i in range(np.shape(counts)[1]):

                    detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)

                    srm_dict = product.get_masked_srm(flare_location=flare_location_stx,
                                            detector_indices_input=detector_indices_srm, 
                                            pixel_indices_input=pixel_indices_srm)

                    counts, counts_uncertainity, t_norm, e_norm, livefrac,_, elut_cor_fac, times_full, energies = sci_data


                    sci_data_indexed = ( counts[:,i,...], 
                                        counts_uncertainity[:,i,...], 
                                        t_norm, 
                                        e_norm, 
                                        livefrac[:,i,...],
                                        _, 
                                        elut_cor_fac, 
                                        times_full, 
                                        energies)


                    spec_1d =  ScienceData._return_spec_object(case,
                                sci_data_indexed,
                                flare_location,
                                detector_indices,
                                pixel_indices,
                                event_time_range,
                                flare_angle,
                                distance,
                                srm_dict)

                    spec_list_working.append((f"{detector_indices[i]}",spec_1d))
                
                spec_collection = NDCollection(spec_list_working,
                                               aligned_axes="all" ) 

                return spec_collection 



            else:

          
                spec_list_collection_working = []

                case = 'spec_sequence_detector_expand'

                for i in range(np.shape(counts)[1]):


                    detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(detector_indices, pixel_indices, case)

                    srm_dict = product.get_masked_srm(flare_location=flare_location_stx,
                                            detector_indices_input=detector_indices_srm, 
                                            pixel_indices_input=pixel_indices_srm)

                    counts, counts_uncertainity, t_norm, e_norm, livefrac,_, elut_cor_fac, times_full, energies = sci_data


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
                                            energies)


                        spec_1d =  ScienceData._return_spec_object(case,
                                    sci_data_indexed,
                                    flare_location,
                                    detector_indices,
                                    pixel_indices,
                                    event_time_range,
                                    flare_angle,
                                    distance,
                                    srm_dict)

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
        
        roll, solo_xyz, pointing = get_hpc_info(product.time_range.start, product.time_range.start)

        solo = HeliographicStonyhurst(*solo_xyz, obstime=product.time_range.center, representation_type="cartesian")
        # flare = SkyCoord(1000*u.arcsec, -1388.43*u.arcsec, frame=Helioprojective(obstime=spec_prod.time_range.center, observer=solo))

        flare_angle = flare_spacecraft_angle(solo,flare_location['hpc'])

        return flare_angle

    @staticmethod
    def _check_shadowing(product,detector_indices):

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
    def _time_indices_format(time_indices,times,rcr,sunkit_spex_spectrum):
        
        first = time_indices[0]

        # if sunkit_spex_spectrum:
        #     if not isinstance(first, (str, Time)):
        #         raise ValueError(
        #         f"If sunkit_spex_spectrm=True, time_indices must be a time range in the format [start_time,end_time]."
        #     )
        #     if isinstance(first, (str, Time)):
        #         if isinstance(first, (str, Time)) and not isinstance(time_indices[0], (list, tuple)):
        #             bins = [time_indices]
        #         else:
        #             bins = time_indices
        #         result = ScienceData._handle_datetime_strings(bins, times)
        #         ScienceData._handle_nested_pairs(result, rcr)
        #         return time_indices
        # else:
        if isinstance(first, int):
            ScienceData._rcr_warning(time_indices, rcr)
            return time_indices
        

        if isinstance(first, (str, Time)):
            # Single bin ["st", "et"] or multiple bins [["st1","et1"],["st2","et2"]]
            # Wrap single bin in a list so _handle_datetime_strings always gets [[st,et],...]
            if isinstance(first, (str, Time)) and not isinstance(time_indices[0], (list, tuple)):
                bins = [[time_indices[i], time_indices[i+1]] for i in range(len(time_indices) - 1)]
            else:
                bins = time_indices
            result = ScienceData._handle_datetime_strings(bins, times)
            ScienceData._handle_nested_pairs(result, rcr)

            return result

        if isinstance(first, (list, tuple)):
            if isinstance(first[0], (str, Time)):
                # [["st1","et1"],["st2","et2"]]
                result = ScienceData._handle_datetime_strings(time_indices, times)
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

        first_rcr = rcr[time_indices[0]]
        for i in time_indices[1:]:
            if rcr[i] != first_rcr:
                warnings.warn(
                    f"RCR state change detected "
                    f"index {time_indices[0]} has RCR={first_rcr!r}, "
                    f"index {i} has RCR={rcr[i]!r}."
                    f"Use with caution!"
                )
        return 

    @staticmethod
    def _rcr_error(indices, rcr):
        """
        Raise a ValueError if the RCR state is not uniform across the given indices.

        Args:
            indices: List of indices into rcr to check.
            rcr:     Full RCR state list.
            context: Optional label for the error message.

        Raises:
            ValueError: If any index has a different RCR value from the first.
        """
        if not indices:
            return
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
        times: list[str | Time]) -> list[list[int]]:

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
                if bin_start <= Time(t) <= bin_end
            ]

            results.append([matched[0], matched[-1]])
        
        print()

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
        bkg=None,
        systematic_error=True,
        sunkit_spex_detector_sum=True
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

        Returns
        -------
        `tuple`
            Counts, errors, times, deltatimes,  energies

        """

        # =====================================================
        # livetime
        # =====================================================

        if time_indices is not None:
            time_indices = self._time_indices_format(time_indices, self.times, self.rcr, sunkit_spex_spectrum)

            print('t_check = ',time_indices)

        detector_indices, pixel_indices = self._indices_check(self,
                                                              detector_indices,
                                                              pixel_indices,
                                                              sunkit_spex_spectrum)
        
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

            sci_data = self._data_select(self,
                                    detector_indices,
                                    pixel_indices,
                                    energy_indices,
                                    time_indices,
                                    livefraction_sci,
                                    livefraction_sci_error,
                                    elut_cor_fac,
                                    sum_all_times)


        else:

            warnings.warn('For background subtraction elut_correction and livetime_correction set as True.')

            energy_indices_bkg = self._energies_bkg_sub(self,
                                                        bkg)
            

            pixel_indices_bkg, detector_indices_bkg = self._bkg_indices_check(self,
                                                                              bkg)

            sci_data_all = self._bkg_sub(self,
                                    bkg,
                                    detector_indices_bkg,
                                    pixel_indices_bkg,
                                    energy_indices_bkg,
                                    livefraction_sci,
                                    livefraction_sci_error,
                                    livefraction_bkg,
                                    livefraction_bkg_error,
                                    elut_cor_fac) 
            

            sci_data = self._data_select(sci_data_all,
                                    detector_indices,
                                    pixel_indices,
                                    energy_indices,
                                    time_indices,
                                    livefraction_sci,
                                    None,
                                    elut_cor_fac,
                                    sum_all_times)
 
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
                                                        time_indices,
                                                        flare_location,
                                                        detector_sum=sunkit_spex_detector_sum)
            
            return sunkit_spex_spectrum
        
        else:
            
            counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies = sci_data
            

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
                counts_var = ScienceData._livetime_uncertainty(counts_var,livefrac,livefrac_error) * livefrac
            
            counts_var = counts_var * norm

            return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies

        

    def get_masked_srm(self, flare_location, detector_indices_input, pixel_indices_input):


        HERE = Path(__file__).parent          
        ROOT = HERE.parent.parent            
        PATH_DRM = ROOT / "config" / "data" / "detector" / 'stx_detector_response_matrix.fits.gz'

        drm = np.array(Table.read(PATH_DRM,hdu=1)['DRM'])
        ph_energies = np.array(Table.read(PATH_DRM,hdu=2)['DRM'])
        ct_energies = np.array(Table.read(PATH_DRM,hdu=3)['DRM'])
    

        energies = self.energies
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

        mask_not_in_e = ~np.isclose(ct_energies[:, None], e_edges[None, :], atol=epsilon).any(axis=1)

        values_to_remove = ct_energies[mask_not_in_e]

        indices_to_remove = np.where(
            np.isclose(ph_energies[:, None], values_to_remove[None, :], atol=epsilon).any(axis=1)
        )[0]

        drm_clipped = np.delete(drm, indices_to_remove, axis=0)
        drm_clipped = np.delete(drm_clipped, indices_to_remove, axis=1)

        ph_energies_clipped = np.delete(ph_energies, indices_to_remove)

        ph_e_diff = np.diff(ph_energies_clipped)

        pixel_areas = STIX_INSTRUMENT.pixel_config["Area"].to("cm2")

        pixel_areas = pixel_areas[pixel_indices_input].value

        area_scale = len(detector_indices_input) * np.sum(pixel_areas)

        energy_widths = np.diff(ph_energies_clipped)

        e_mids = ph_energies_clipped[:-1] + (energy_widths / 2)

        trans = Transmission()

        tot_trans = trans.get_transmission(energies=e_mids * u.keV)

        attenuation = np.zeros(len(tot_trans["det-1"]))

        for i, det in enumerate(detector_indices_input):
            attenuation += tot_trans[f"det-{det}"]

        attenuation = attenuation / len(detector_indices_input)


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

        grid_transmission = get_grid_transmission(e_mids, flare_location)

        grid_transmission = grid_transmission.mean(axis=1)

        srm = (drm_new * grid_transmission[:, None]) / ct_e_diff[None, :]

        return {"srm": srm, "ph_axis": ph_energies_clipped, "geo_area": area_scale}
    

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
