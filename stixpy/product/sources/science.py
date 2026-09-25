import warnings
from itertools import product
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import ConciseDateFormatter, DateFormatter, HourLocator
from matplotlib.widgets import Slider
from ndcube import NDCollection, NDCubeSequence, NDMeta
from sunkit_spex.spectrum.spectrum import SpectralAxis, Spectrum
from sunkit_spex.spectrum.uncertainty import PoissonUncertainty

import astropy.units as u
from astropy.table import QTable, Table, vstack
from astropy.time import Time
from astropy.visualization import quantity_support

from sunpy.coordinates import HeliographicStonyhurst
from sunpy.time.timerange import TimeRange
from sunpy.util import deprecated

from stixpy.calibration.detector import tailing_matrix
from stixpy.calibration.elut import get_elut_correction
from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.calibration.transmission import Transmission
from stixpy.config.instrument import STIX_INSTRUMENT
from stixpy.coordinates.flare_angle import flare_spacecraft_angle
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.io.readers import read_subc_params
from stixpy.product.product import L1Product

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
            elut_correction=False,
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
            elut_correction=False,
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
        return ScienceData._rcr_shift(self.data["rcr"], self.data["counts"])

    @property
    def rcr_raw(self):
        """
        The rcr state
        """
        return self.data["rcr"]

    @staticmethod
    def _indices_check(product, detector_indices, pixel_indices, energy_indices):
        """
        Check the requested detector, pixel and energy indices against what the
        product contains, and fill in defaults.

        Detectors and pixels are checked against `product.detector_masks` and
        `product.pixel_masks`. They may be a flat list of indices or a list of
        [start, end] pairs; any that are not in the product raise a warning and the
        selection is kept as given. None selects every detector or pixel in the
        product.

        Detectors can also be given as a label, in any case. Labels are not checked
        against the mask:

        - "top24": the 24 imaging detectors of sub-collimators 3-10 (indices 0-7,
          13-15 and 19-31), as in STIX-GSW.
        - "bkg": the background detector (index 9) with its small-aperture pixels
          [2, 5]. If `pixel_indices` is None it is set to [2, 5] with a warning;
          any other pixels raise a ValueError. Use ``detector_indices=[9]`` to
          choose other pixels.

        For spectrogram products (counts with fewer than 4 dimensions) detector and
        pixel selections do not apply, and any that were given are dropped with a
        warning.

        Energy indices number the rows of `product.energies`, i.e. the last axis of
        the counts. That table only holds the bins in the product's energy mask, so
        an index outside it raises a ValueError naming the valid indices and energy
        range.

        Parameters
        ----------
        product : ScienceData
            Product whose masks and energy table define what is available.
        detector_indices : list, numpy.ndarray, str or None
            Flat detector indices, [start, end] pairs, "top24", "bkg", or None for
            all detectors in the product.
        pixel_indices : list, numpy.ndarray or None
            Flat pixel indices, [start, end] pairs, or None for all pixels in the
            product ([2, 5] with ``detector_indices="bkg"``).
        energy_indices : list, numpy.ndarray or None
            Flat energy bin indices or [start, end] pairs, numbering the rows of
            `product.energies`. None skips the energy check.

        Returns
        -------
        detector_indices : list, numpy.ndarray or None
            The detector selection, with a label resolved to indices. None for
            spectrogram products.
        pixel_indices : list, numpy.ndarray or None
            The pixel selection. None for spectrogram products.
        energy_indices : list, numpy.ndarray or None
            `energy_indices`, unchanged.

        Raises
        ------
        ValueError
            If a detector label is not recognised, if ``detector_indices="bkg"`` is
            given with pixels other than [2, 5], or if a requested energy bin is
            outside the product's energy table.

        Warns
        -----
        UserWarning
            If a requested detector or pixel is not in the product, if detector or
            pixel indices are given for a spectrogram product, or if
            ``detector_indices="bkg"`` sets the pixels to [2, 5].
        """

        # --- Detector indices ---

        if detector_indices is not None:
            if len(product.data["counts"].shape) < 4:
                warnings.warn(
                    f"As a spectrogram file is being used, the user selected detector indices \
                                {detector_indices} will not be used, defaulting to the indices used in the creation \
                                of the spectrgram file.",
                    stacklevel=3,
                )

                detector_indices = None

            else:
                detector_indices_working = detector_indices

                if isinstance(detector_indices_working, str):
                    # named detector sets, as in STIX-GSW stx_label2det_ind
                    detector_labels = {
                        "top24": [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            13,
                            14,
                            15,
                            19,
                            20,
                            21,
                            22,
                            23,
                            24,
                            25,
                            26,
                            27,
                            28,
                            29,
                            30,
                            31,
                        ],
                        "bkg": [9],
                    }
                    label = detector_indices_working.lower()
                    if label not in detector_labels:
                        raise ValueError(
                            f"Unknown detector label {detector_indices_working!r}, use one of {list(detector_labels)}."
                        )
                    detector_indices = np.array(detector_labels[label])

                    if label == "bkg":
                        if pixel_indices is None:
                            pixel_indices = [2, 5]
                            warnings.warn(
                                'detector_indices="bkg" with no pixel_indices given: using the BKG detector\'s '
                                "small-aperture pixels [2, 5].",
                                stacklevel=3,
                            )
                        elif np.ndim(pixel_indices) != 1 or sorted(np.asarray(pixel_indices).tolist()) != [2, 5]:
                            raise ValueError(
                                f'detector_indices="bkg" uses the BKG detector\'s small-aperture pixels [2, 5], '
                                f"but pixel_indices={pixel_indices} was given. Either leave pixel_indices unset "
                                f'(None) or set pixel_indices=[2, 5] if using the "bkg" preset.'
                                f"To use background detectors with other pixel_indices use detector_indices=[9]"
                                f"with your choice of pixel_indices."
                            )

                    detector_indices_full = np.unique(np.where(product.detector_masks.masks == 1)[1])
                    missing = np.setdiff1d(detector_indices, detector_indices_full)
                    if missing.size > 0:
                        usable = np.intersect1d(detector_indices, detector_indices_full)
                        raise ValueError(
                            f'detector_indices="{label}" includes detectors {missing.tolist()} that are switched off in '
                            f"this file (detectors on: {detector_indices_full.tolist()}). Give the detectors explicitly, "
                            f"e.g. detector_indices={usable.tolist()}."
                        )

                else:
                    detector_indices_full = np.unique(np.where(product.detector_masks.masks == 1)[1])

                    if np.ndim(detector_indices_working) == 2:
                        # [[start, end], ...] range format
                        for start, end in detector_indices_working:
                            requested = np.arange(start, end + 1)
                            missing = np.setdiff1d(requested, detector_indices_full)
                            if missing.size > 0:
                                raise ValueError(
                                    f"Detector range [{start}, {end}] includes detectors {missing.tolist()} that are "
                                    f"switched off in this file (detectors on: {detector_indices_full.tolist()}). Split "
                                    "the range around them or give the detectors as a flat list."
                                )
                    else:
                        missing = np.setdiff1d(detector_indices_working, detector_indices_full)
                        if missing.size > 0:
                            raise ValueError(
                                f"Detectors {missing.tolist()} are switched off in this file (detectors on: "
                                f"{detector_indices_full.tolist()}). Remove them, or leave detector_indices unset to "
                                "use every detector that is on."
                            )

        else:
            if len(product.data["counts"].shape) < 4:
                detector_indices = None
            else:
                detector_indices = np.where(product.detector_masks.masks == 1)[1]

        # --- Pixel indices ---
        if pixel_indices is not None:
            pixel_indices_full = np.unique(np.where(product.pixel_masks.masks == 1)[1])

            if len(product.data["counts"].shape) < 4:
                warnings.warn(
                    f"As a spectrogram file is being used, the user selected detector indices \
                                {pixel_indices} will not be used, defaulting to the indices used in the creation \
                                of the spectrgram file.",
                    stacklevel=3,
                )
                pixel_indices = None

            else:
                if np.ndim(pixel_indices) == 2:
                    for start, end in pixel_indices:
                        requested = np.arange(start, end + 1)
                        missing = np.setdiff1d(requested, pixel_indices_full)
                        if missing.size > 0:
                            raise ValueError(
                                f"Pixel range [{start}, {end}] includes pixels {missing.tolist()} that are switched off "
                                f"in this file (pixels on: {pixel_indices_full.tolist()}). Split the range around them "
                                "or give the pixels as a flat list."
                            )
                else:
                    missing = np.setdiff1d(pixel_indices, pixel_indices_full)
                    if missing.size > 0:
                        raise ValueError(
                            f"Pixels {missing.tolist()} are switched off in this file (pixels on: "
                            f"{pixel_indices_full.tolist()}). Remove them, or leave pixel_indices unset to use every "
                            "pixel that is on."
                        )

        else:
            if len(product.data["counts"].shape) < 4:
                pixel_indices = None
            else:
                pixel_indices = np.where(product.pixel_masks.masks == 1)[1]

        # --- Energy indices ---
        if energy_indices is not None:
            energy_indices_full = np.arange(len(product.energies))
            e_min = np.nanmin(product.energies["e_low"].value)
            e_max = np.nanmax(product.energies["e_high"].value)

            energy_range = (
                f"The product has energy indices {energy_indices_full[0]}-{energy_indices_full[-1]} "
                f"({e_min} - {e_max} keV)."
            )

            if np.ndim(energy_indices) == 2:
                for start, end in energy_indices:
                    requested = np.arange(start, end + 1)
                    missing = np.setdiff1d(requested, energy_indices_full)
                    if missing.size > 0:
                        raise ValueError(
                            f"Energy indices {missing.tolist()} in range [{start}, {end}] are not included in the product's energy mask. {energy_range}"
                        )
            else:
                missing = np.setdiff1d(energy_indices, energy_indices_full)
                if missing.size > 0:
                    raise ValueError(
                        f"The following energy indices are not included in the product's energy mask: {missing.tolist()}. {energy_range}"
                    )

        return detector_indices, pixel_indices, energy_indices

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _full_layout(product, column):
        """
        Place a (time, detector, pixel, energy) data column on the full 32 x 12 grid.

        L1 pixel data files store only the enabled detectors and pixels, so position
        n on those axes is not necessarily detector or pixel n. As in IDL's
        ``stx_read_pixel_data_fits_file``, the stored values are placed at their
        detector and pixel numbers, with zeros for detectors and pixels that are off.
        After this every detector and pixel index is a real detector or pixel number.

        Parameters
        ----------
        product : ScienceData
            Product to read from.
        column : str
            Column of ``product.data``, e.g. 'counts' or 'counts_comp_err'.

        Returns
        -------
        astropy.units.Quantity or numpy.ndarray
            The column with shape (time, 32, 12, energy). Returned unchanged if it
            already has that shape, is not 4-dimensional, or holds summed pixel sets.

        Raises
        ------
        KeyError
            If `column` is not in ``product.data``.
        ValueError
            If the detector mask changes within the file, or the stored shape does
            not match the masks.
        """
        values = product.data[column]
        if values.ndim != 4 or values.shape[1:3] == (32, 12) or product.pixel_masks.masks.ndim != 2:
            return values

        if len(product.detector_masks.masks) > 1:
            raise ValueError(
                f"The detector mask changes within the file, so {column!r} can not be placed on the full detector grid."
            )
        dets = np.flatnonzero(product.detector_masks.masks[0])
        pixs = np.flatnonzero(product.pixel_masks.masks.any(axis=0))  # pixels used at any time, as stored by STIXcore

        if values.shape[1] == 32:
            values = values[:, dets]
        if values.shape[2] == 12:
            values = values[:, :, pixs]
        if values.shape[1:3] != (dets.size, pixs.size):
            raise ValueError(
                f"{column!r} has shape {values.shape}, which does not match the masks "
                f"({dets.size} detectors, {pixs.size} pixels)."
            )

        full = np.zeros((values.shape[0], 32, 12, values.shape[3]), dtype=values.dtype)
        if isinstance(values, u.Quantity):
            full = full << values.unit
        full[:, dets[:, None], pixs, :] = values
        return full

    @staticmethod
    def _data_select(
        product,
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
        bkg,
    ):
        """
        Apply the requested detector, pixel, energy and time selection to the data.

        `product` is either a `ScienceData` product or the tuple returned by
        `_bkg_sub`. On the product path the count uncertainty starts as
        ``sqrt(counts + compression_error**2)``.

        In both cases the 0 keV bottom energy bin and the open top bin (upper edge
        NaN) are removed first. Energy indices number the rows of the product's
        energy table, so they are shifted down by one when the bottom bin is
        removed and then clipped into the remaining range.

        On every axis, flat indices keep those bins and [start, end] pairs sum each
        inclusive range into one bin (counts summed, uncertainties in quadrature,
        livetime values averaged). The steps run in this order:

        1. ELUT correction, except on the background path, where `_bkg_sub` has
           already applied it.
        2. Pixel selection.
        3. Energy selection; for ranges, `e_norm` and the energy table are rebuilt.
        4. Livetime correction by detector group (see `_apply_livetime`), then a sum
           over pixels. Only when a livetime fraction is available and this is not
           the background path. With no detector selection all detectors form one
           group, a flat selection forms one group, and each [start, end] range is
           its own group.
        5. Detector selection.
        6. Counts are set to zero wherever their total over the pooled axes (see
           `sunkit_spex_detector_sum`) is negative.
        7. Optionally, a systematic uncertainty is added.
        8. Time selection, and optionally a sum over all time bins.

        Parameters
        ----------
        product : ScienceData or tuple
            The science product, or the 10-element tuple returned by `_bkg_sub`.
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
            Livetime fraction from `_livefrac`. Only used when `product` is a
            `ScienceData`; on the background path it comes from the tuple.
        livefrac_error : astropy.units.Quantity or None
            Livetime uncertainty from `_livefrac`. Only used when `product` is a
            `ScienceData`.
        elut_cor_fac : numpy.ndarray or None
            ELUT correction factor from `_elut_correction_sort`. Only used when
            `product` is a `ScienceData`.
        rcr : array_like
            Not used; the RCR states are taken from `product`.
        sum_all_times : bool
            If True and `time_indices` is a list of [start, end] pairs, the
            resulting time bins are summed into one.
        systematic : bool
            If True, add a systematic uncertainty of 7% below 7 keV, 5% from 7 to
            10 keV and 3% from 10 keV of the pooled counts, spread so that a
            quadrature sum over the pooled axes returns that percentage.
        sunkit_spex_detector_sum : bool
            Which axes are pooled into one output bin for steps 6 and 7: detectors
            and pixels if True, pixels only if False.
        bkg : bool
            True when `product` is the tuple from `_bkg_sub`. Its count arrays are
            already trimmed, livetime corrected and ELUT corrected, so only the
            energy table is trimmed and those corrections are not repeated.

        Returns
        -------
        tuple
            ``(counts, counts_var, t_norm, e_norm, livefrac, livefrac_error,
            elut_cor_fac, times, energies, rcr)`` after the selection. `counts_var`
            holds the 1-sigma uncertainty, not the variance.
        """

        if isinstance(product, ScienceData):
            e_norm = product.dE
            counts = ScienceData._full_layout(product, "counts")

            shape = counts.shape

            try:
                counts_var = ScienceData._full_layout(product, "counts_comp_err") ** 2
            except KeyError:
                counts_var = ScienceData._full_layout(product, "counts_comp_comp_err") ** 2

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
            counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr = product

        if bkg:
            if energies["e_low"][0].value == 0:
                energies = energies[1:]
                if energy_indices is not None:
                    energy_indices = np.asarray(energy_indices) - 1

            if np.isnan(energies["e_high"][-1].value):
                energies = energies[:-1]

        if not bkg:
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
                if not bkg:
                    counts_var = counts_var[..., pixel_mask[:num_pixels], :]
                if not bkg and livefrac is not None and livefrac.shape[2] != 1:
                    livefrac = livefrac[:, :, pixel_mask[:num_pixels], :]
                if not bkg and livefrac_error is not None and livefrac_error.shape[2] != 1:
                    livefrac_error = livefrac_error[:, :, pixel_mask[:num_pixels], :]

            if pixel_indices.ndim == 2:
                counts = np.concatenate(
                    [np.sum(counts[..., pl : ph + 1, :], axis=2, keepdims=True) for pl, ph in pixel_indices], axis=2
                )
                if not bkg:
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
                            np.sqrt(np.mean(livefrac_error[..., pl : ph + 1, :] ** 2, axis=2, keepdims=True))
                            for pl, ph in pixel_indices
                        ],
                        axis=2,
                    )

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

                if bkg:
                    if livefrac is not None:
                        livefrac = livefrac[..., energy_mask]

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
                        [np.mean(elut_cor_fac[..., el : eh + 1]) for el, eh in energy_indices], axis=-1
                    )

                if bkg:
                    if livefrac is not None:
                        livefrac = np.concatenate(
                            [np.mean(livefrac[..., el : eh + 1], axis=2, keepdims=True) for el, eh in pixel_indices],
                            axis=2,
                        )

                if livefrac_error is not None:
                    livefrac_error = np.concatenate(
                        [
                            np.sqrt(np.mean(livefrac_error[..., el : eh + 1] ** 2, axis=2, keepdims=True))
                            for el, eh in pixel_indices
                        ],
                        axis=2,
                    )

                energies = np.atleast_2d(
                    [(energies["e_low"][el].value, energies["e_high"][eh].value) for el, eh in energy_indices]
                )
                energies = QTable(energies * u.keV, names=["e_low", "e_high"])

        if not bkg and livefrac is not None and detector_indices is None:
            # if not bkg and livefrac is not None and detector_indices is None and sunkit_spex_detector_sum:
            n_det = counts.shape[1]
            groups = [np.arange(n_det)]

            counts_var = ScienceData._livetime_uncertainty(counts_var, livefrac_error, livefrac)
            counts, counts_var, livefrac = ScienceData._apply_livetime(counts, counts_var, livefrac, groups)
            counts = np.nansum(counts, axis=2, keepdims=True)

        if detector_indices is not None:
            detector_indices = np.asarray(detector_indices)  # "top24" must already be resolved to indices upstream

            # ---- livetime -------------------------------------------------------
            # Skipped on the bkgsub path: _bkg_sub has already applied the livetime
            # correction and collapsed counts_var's pixel axis.
            if not bkg and livefrac is not None:
                if detector_indices.ndim == 1:
                    groups = [detector_indices]  # all selected -> one spectrum
                else:  # ndim == 2 : each (dl, dh) range -> one output spectrum
                    groups = [np.arange(dl, dh + 1) for dl, dh in detector_indices]

                counts_var = ScienceData._livetime_uncertainty(counts_var, livefrac_error, livefrac)
                counts, counts_var, livefrac = ScienceData._apply_livetime(counts, counts_var, livefrac, groups)
                counts = np.nansum(counts, axis=2, keepdims=True)

            # ---- detector selection / summing -----------------------------------
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
                    [np.sum(counts[:, dl : dh + 1, ...], axis=1, keepdims=True) for dl, dh in detector_indices]
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

            # ---- negative clip, then systematic ---------------------------------
            # Detectors and pixels are resolved by this point, so the sum below is
            # the quantity that becomes one output spectral bin. Clipping here (not
            # per-detector in _bkg_sub, and not after the time sum) puts it on the
            # collapsed value while the time axis is still intact, and lets the
            # systematic derive from post-clip counts.
            #
            # sunkit_spex_detector_sum=True  -> detectors collapse downstream, so the
            #   output bin is the sum over remaining detector AND pixel axes.
            # sunkit_spex_detector_sum=False -> each detector stays its own output
            #   bin, so only the pixel axis is pooled.
        sum_axes = (1, 2) if sunkit_spex_detector_sum else (2,)

        total = np.nansum(counts, axis=sum_axes, keepdims=True)
        counts = np.where(total < 0, 0, counts.value) * counts.unit

        if systematic:
            e_mean = ((energies["e_low"] + energies["e_high"]) / 2).value
            systematic_err_percentage = np.select(
                [e_mean < 7, (e_mean < 10) & (e_mean >= 7), e_mean >= 10],
                [0.07, 0.05, 0.03],
            )

            total = np.nansum(counts, axis=sum_axes, keepdims=True)

            # Divisor taken from counts_var, not counts: these are the axes that
            # actually get quadrature-summed downstream. On the bkgsub path
            # counts_var's pixel axis is already collapsed while counts' is not,
            # so sizing off counts would mis-scale by sqrt(n_pix). Spreading
            # p*total over n_elem slots as p*total/sqrt(n_elem) means the
            # downstream quadrature sum returns exactly p*total.
            n_elem = int(np.prod([counts_var.shape[a] for a in sum_axes]))
            sys_err_elem = (systematic_err_percentage * total) / np.sqrt(n_elem)

            counts_var = (
                np.sqrt(counts_var.value**2 + np.broadcast_to(sys_err_elem.value, counts_var.shape) ** 2) * u.ct
            )

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
                    livefrac = np.vstack(
                        [np.mean(livefrac[tl : th + 1, ...], axis=0, keepdims=True) for tl, th in time_indices]
                    )

                if livefrac_error is not None:
                    livefrac_error = np.vstack(
                        [
                            np.sqrt(np.mean(livefrac_error[tl : th + 1, ...] ** 2, axis=0, keepdims=True))
                            for tl, th in time_indices
                        ]
                    )

                counts_var = np.vstack(
                    [
                        np.sqrt(np.sum(counts_var[tl : th + 1, ...] ** 2, axis=0, keepdims=True))
                        for tl, th in time_indices
                    ]
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

    @staticmethod
    def _bkg_sub(
        product,
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
        - `sunkit_spex_detector_sum` True: by the mean livetime fraction of the
          selected detectors;
        - `sunkit_spex_detector_sum` False with flat detector indices: by an
          effective livetime per detector and pixel, with negative counts set to
          zero;
        - `sunkit_spex_detector_sum` False with [start, end] ranges: by one
          count-weighted effective livetime per range.

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
        detector_indices : numpy.ndarray
            Science detector selection, flat or [start, end] pairs.
        pixel_indices : numpy.ndarray
            Science pixel selection, flat or [start, end] pairs.
        sunkit_spex_detector_sum : bool
            Whether detectors will be summed into one spectrum downstream. Sets how
            the livetime correction is undone (see above).
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

        Raises
        ------
        ValueError
            If a requested science pixel is not in the background product.
        """

        e_norm = product.dE
        counts = ScienceData._full_layout(product, "counts")
        shape = counts.shape

        try:
            counts_var = (ScienceData._full_layout(product, "counts_comp_err").value ** 2) * u.ct
        except KeyError:
            counts_var = (ScienceData._full_layout(product, "counts_comp_comp_err").value ** 2) * u.ct

        counts_bkg = ScienceData._full_layout(bkg, "counts")

        try:
            counts_var_bkg = (ScienceData._full_layout(bkg, "counts_comp_err").value ** 2) * u.ct
        except KeyError:
            counts_var_bkg = (ScienceData._full_layout(bkg, "counts_comp_comp_err").value ** 2) * u.ct

        counts_var_bkg = np.sqrt(counts_bkg + counts_var_bkg)

        # counts_bkg = counts_bkg[:, detector_indices_bkg, :, :]
        # counts_bkg = counts_bkg[:, :, pixel_indices_bkg, :]
        # counts_bkg = counts_bkg[:, :, :, energy_indices_bkg]

        # counts_var_bkg = counts_var_bkg[:, detector_indices_bkg, :, :]
        # counts_var_bkg = counts_var_bkg[:, :, pixel_indices_bkg, :]
        # counts_var_bkg = counts_var_bkg[:, :, :, energy_indices_bkg]

        # livefrac_error_bkg = livefrac_error_bkg[:, detector_indices_bkg, :, :]
        # livefrac_error_bkg = livefrac_error_bkg[:, :, pixel_indices_bkg, :]

        # if elut_cor_fac is None:
        #     livefrac_error_bkg = livefrac_error_bkg[:, :, :, energy_indices_bkg]

        # livefrac_bkg = livefrac_bkg[:, detector_indices_bkg, :, :]

        # if len(shape) == 4:
        #     pix = np.asarray(pixel_indices)
        #     if pix.ndim == 2:
        #         pix = np.asarray(ScienceData._indices_expand_ranges(pix, nest=False))
        #     pix = np.asarray(pix, dtype=int)

        #     # counts_var_bkg has already been sliced to pixel_indices_bkg, so map the
        #     # requested pixels onto positions within that subset.
        #     # pix_bkg_pos = np.searchsorted(np.asarray(pixel_indices_bkg), pix)

        #     pix_bkg_pos = np.flatnonzero(np.isin(pixel_indices_bkg, pix))
        #     if pix_bkg_pos.size != len(pix):
        #         missing = np.setdiff1d(pix, pixel_indices_bkg)
        #         raise ValueError(f"pixels {missing.tolist()}")

        #     counts_var_bkg = counts_var_bkg[:, :, pix_bkg_pos, :]
        #     if livefrac_error_bkg.shape[2] != 1:
        #         livefrac_error_bkg = livefrac_error_bkg[:, :, pix_bkg_pos, :]

        # Both products are on the full 32 x 12 grid (_full_layout), so detector and pixel
        # numbers index the background directly: keep the full grid, so its shape matches the
        # science counts, and only select the matching energy bins.
        counts_bkg = counts_bkg[:, :, :, energy_indices_bkg]
        counts_var_bkg = counts_var_bkg[:, :, :, energy_indices_bkg]

        if elut_cor_fac is None:
            livefrac_error_bkg = livefrac_error_bkg[:, :, :, energy_indices_bkg]

        if len(shape) == 4:
            pix = np.asarray(pixel_indices)
            if pix.ndim == 2:
                pix = np.asarray(ScienceData._indices_expand_ranges(pix, nest=False))
            pix = np.asarray(pix, dtype=int)

            dets = np.asarray(detector_indices)
            if dets.ndim == 2:
                dets = np.asarray(ScienceData._indices_expand_ranges(dets, nest=False))

            # every selected detector and pixel must be on in both files
            missing_dets = np.setdiff1d(dets, detector_indices_bkg)
            missing_pix = np.setdiff1d(pix, pixel_indices_bkg)
            if missing_dets.size or missing_pix.size:
                raise ValueError(
                    f"Detectors {missing_dets.tolist()} and pixels {missing_pix.tolist()} are selected but are not "
                    "on in both the science and background files, so they can not be background subtracted. "
                    f"Detectors on in both: {list(detector_indices_bkg)}, pixels on in both: {list(pixel_indices_bkg)}."
                )

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

        counts_var_bkg = ScienceData._livetime_uncertainty(counts_var_bkg, livefrac_error_bkg, livefrac_bkg)

        if len(shape) < 4:
            counts = counts.reshape(shape[0], 1, 1, shape[-1])
            counts_var = counts_var.reshape(shape[0], 1, 1, shape[-1])

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

        if len(shape) == 4:
            counts_var = counts_var[:, :, pix, :]

            if livefrac_error.shape[2] != 1:
                livefrac_error = livefrac_error[:, :, pix, :]

        if elut_cor_fac is not None:
            counts_var = counts_var * elut_cor_fac

        counts_var = ScienceData._livetime_uncertainty(counts_var, livefrac_error, livefrac)

        t_norm_bkg = bkg.data["timedel"]
        t_norm = t_norm.to(u.s)
        t_norm_bkg = t_norm_bkg.to(u.s)

        if elut_cor_fac is not None:
            counts_uncorr = counts * elut_cor_fac
            counts_lvtcorr = (counts * elut_cor_fac) / livefrac
        else:
            counts_uncorr = counts
            counts_lvtcorr = (counts) / livefrac

        if elut_cor_fac is not None:
            counts_uncorr_bkg = counts_bkg * elut_cor_fac
            counts_lvtcorr_bkg = (counts_bkg / livefrac_bkg) * elut_cor_fac
        else:
            counts_uncorr_bkg = counts_bkg
            counts_lvtcorr_bkg = counts_bkg / livefrac_bkg

        count_rate_uncorr_bkg = counts_uncorr_bkg / t_norm_bkg.mean()
        count_uncorr_scaled_bkg = t_norm.reshape(len(t_norm), 1, 1, 1) * count_rate_uncorr_bkg

        count_rate_lvtcorr_bkg = counts_lvtcorr_bkg / t_norm_bkg.mean()
        count_lvtcorr_scaled_bkg = t_norm.reshape(len(t_norm), 1, 1, 1) * count_rate_lvtcorr_bkg

        counts_var_lvtcorr = counts_var
        counts_var_lvtcorr_bkg = counts_var_bkg

        counts_var_lvtcorr_scaled_bkg = (counts_var_lvtcorr_bkg / t_norm_bkg.mean()) * t_norm.reshape(
            len(t_norm), 1, 1, 1
        )

        spec_in_corr = counts_lvtcorr - count_lvtcorr_scaled_bkg
        spec_in = counts_uncorr - count_uncorr_scaled_bkg

        spec_in_err = np.sqrt((counts_var_lvtcorr**2) + (counts_var_lvtcorr_scaled_bkg**2))

        spec_in_corr_lvt = counts_lvtcorr
        spec_in_lvt = counts_uncorr

        if energies["e_low"][0].value == 0:
            spec_in = spec_in[..., 1:]
            spec_in_lvt = spec_in_lvt[..., 1:]
            spec_in_corr_lvt = spec_in_corr_lvt[..., 1:]
            spec_in_corr = spec_in_corr[..., 1:]
            spec_in_err = spec_in_err[..., 1:]
            e_norm = e_norm[1:]
            livefrac_error = livefrac_error[..., 1:]
            if elut_cor_fac is not None:
                elut_cor_fac = elut_cor_fac[..., 1:]

        if np.isnan(energies["e_high"][-1].value):
            spec_in = spec_in[..., :-1]
            spec_in_corr = spec_in_corr[..., :-1]
            spec_in_lvt = spec_in_lvt[..., :-1]
            spec_in_corr_lvt = spec_in_corr_lvt[..., :-1]
            spec_in_err = spec_in_err[..., :-1]
            e_norm = e_norm[:-1]
            livefrac_error = livefrac_error[..., :-1]
            if elut_cor_fac is not None:
                elut_cor_fac = elut_cor_fac[..., :-1]

        if len(shape) < 4:
            spec_in_final = spec_in_corr * livefrac
            spec_in_err_final = spec_in_err * livefrac

            counts = spec_in_final
            counts_var = spec_in_err_final

        else:
            detector_groups = None
            if np.asarray(detector_indices).ndim == 2:
                detector_groups = ScienceData._indices_expand_ranges(
                    detector_indices, nest=True
                )  # list of per-group arrays
                detector_indices = np.concatenate(detector_groups)  # flat — identical to nest=False

            if np.asarray(pixel_indices).ndim == 2:
                pixel_indices = ScienceData._indices_expand_ranges(pixel_indices, nest=False)

            if sunkit_spex_detector_sum:
                eff_livefrac = np.nanmean(livefrac[:, detector_indices, :, :], axis=1, keepdims=True)

                spec_in_final = spec_in_corr * eff_livefrac
                spec_in_err_final = spec_in_err * eff_livefrac

                counts = spec_in_final

                counts_var = spec_in_err_final

                livefrac = np.broadcast_to(eff_livefrac, counts.shape)

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
                        group_eff = np.nansum(
                            spec_in_lvt[:, gidx[0], gidx[1], :], axis=(1, 2, 3), keepdims=True
                        ) / np.nansum(spec_in_corr_lvt[:, gidx[0], gidx[1], :], axis=(1, 2, 3), keepdims=True)

                        # write the group's single ratio onto every detector in the group
                        # (all pixels), so _data_select's later per-group mean returns it unchanged
                        eff_livefrac_full[:, group_dets, :, :] = group_eff

                        spec_in_final[:, group_dets, :, :] = spec_in_corr[:, group_dets, :, :] * group_eff
                        spec_in_err_final[:, group_dets, :, :] = spec_in_err[:, group_dets, :, :] * group_eff

                    # counts = np.where(spec_in_final < 0, 0, spec_in_final)
                    counts = spec_in_final
                    counts_var = spec_in_err_final
                    livefrac = eff_livefrac_full

        return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr

    @staticmethod
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

    @staticmethod
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

        pixel_indices_full = np.where(product.pixel_masks.masks == 1)[1]
        pixel_indices_full_bkg = np.where(bkg.pixel_masks.masks == 1)[1]
        pixel_indices = [d for i, d in enumerate(pixel_indices_full) if d in pixel_indices_full_bkg]

        detector_indices_full = np.where(product.detector_masks.masks == 1)[1]
        detector_indices_full_bkg = np.where(bkg.detector_masks.masks == 1)[1]
        detector_indices = [d for i, d in enumerate(detector_indices_full) if d in detector_indices_full_bkg]

        return pixel_indices, detector_indices

    @staticmethod
    def _livefrac(product, elut_cor_fac, pixel_indices, energy_indices=None):
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
        elut_cor_fac : numpy.ndarray or None
            ELUT correction factor applied to the counts before the uncertainty is
            computed. None for no ELUT correction.
        pixel_indices : numpy.ndarray or None
            Pixels to compute the uncertainty over, flat or [start, end] pairs, or
            None for all pixels. Ignored for spectrogram products.
        energy_indices : numpy.ndarray, optional
            Energy bins of the counts to keep before applying `elut_cor_fac`, for
            when the factor was computed for another product's bins (the
            background case). Only used if `elut_cor_fac` is given.

        Returns
        -------
        livefrac : numpy.ndarray
            Livetime fraction, shape (time, detector, 1, 1) for pixel data or
            (time, 1, 1, 1) for spectrogram products.
        livefrac_error : astropy.units.Quantity
            Livetime uncertainty in counts, with the same shape as the counts
            after any energy selection.
        """

        trigger_to_detector = STIX_INSTRUMENT.subcol_adc_mapping
        shape = product.data["counts"].shape

        if len(shape) < 4:
            counts = product.data["counts"].reshape(shape[0], 1, 1, shape[-1])

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
            counts = ScienceData._full_layout(product, "counts")

            triggers = product.data["triggers"][:, trigger_to_detector].astype(float)[...]

            triggers_error = product.data["triggers_comp_err"][:, trigger_to_detector].astype(float)[...]

            triggers_error = np.sqrt(triggers_error**2 + triggers)

            triggers_lower = np.floor(
                np.maximum(triggers - triggers_error, 0)
            )  # This brings in line with IDL precision, if removed then the ratio at livefrac of ~0.5 goes to 0.0002, rather than ~1e-7.
            triggers_upper = np.floor(
                triggers + triggers_error
            )  # This brings in line with IDL precision, if removed then the ratio at livefrac of ~0.5 goes to 0.0002, rather than ~1e-7.
            # triggers_lower = np.maximum(triggers - triggers_error, 0)
            # triggers_upper = triggers + triggers_error

            livefrac, _, _ = get_livetime_fraction(triggers / product.data["timedel"].to("s").reshape(-1, 1))
            livefrac_lower, _, _ = get_livetime_fraction(
                triggers_lower / product.data["timedel"].to("s").reshape(-1, 1)
            )
            livefrac_upper, _, _ = get_livetime_fraction(
                triggers_upper / product.data["timedel"].to("s").reshape(-1, 1)
            )

            livefrac = livefrac.reshape(livefrac.shape + (1, 1))
            livefrac_lower = livefrac_lower.reshape(livefrac_lower.shape + (1, 1))
            livefrac_upper = livefrac_upper.reshape(livefrac_upper.shape + (1, 1))

        if elut_cor_fac is not None:
            if energy_indices is not None:
                counts = counts[..., energy_indices] * elut_cor_fac
            else:
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
                    pix = np.asarray(ScienceData._indices_expand_ranges(pix, nest=False))
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

    @staticmethod
    def _return_spec_object(case, sci_data, flare_angle, distance, srm_dict, bkg):
        """
        Build one `sunkit_spex` `Spectrum` from a slice of the selected data.

        Counts are summed over every axis except energy, and uncertainties in
        quadrature. The exposure time is the integration time weighted by the
        livetime fraction, averaged over detectors and summed over time. The
        spectral response matrix from `srm_dict` is multiplied by the count energy
        bin widths.

        Parameters
        ----------
        case : str
            Which slice this is: 'spec_1D_detector_collapse',
            'spec_sequence_detector_collapse', 'spec_1D_detector_expand' or
            'spec_sequence_detector_expand'. Sets which axes are summed. For every
            case except 'spec_1D_detector_collapse', negative summed counts are set
            to zero.
        sci_data : tuple
            The 10-element tuple from `_data_select`, or a single time bin or
            detector of it, as sliced by `_get_sunkit_spex_spectrum`.
        flare_angle : astropy.units.Quantity or None
            Flare angle, stored in the metadata.
        distance : astropy.units.Quantity
            Spacecraft-Sun distance, stored in the metadata.
        srm_dict : dict
            Output of `get_masked_srm`, with keys 'srm', 'ph_axis' and 'geo_area'.
        bkg : bool
            Not used.

        Returns
        -------
        sunkit_spex.spectrum.Spectrum
            Spectrum with the counts, their uncertainty, and the count energy edges
            as the spectral axis. Its metadata holds 'exposure_time', 'geo_area',
            'angle', 'distance', 'srm', 'ph_axis' and 'time_range'.
        """

        counts, counts_uncertainity, t_norm, _, livefrac, _, elut_cor_fac, times_full, energies, _ = sci_data

        t_norm = t_norm.to(u.s)

        counts_axis = np.concatenate([energies["e_low"], [energies["e_high"][-1]]])

        if case == "spec_1D_detector_collapse":
            counts_final = np.nansum(counts, axis=(0, 1, 2))
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2, axis=(0, 1, 2)))

            t_norm = t_norm[:, None, None, None] * livefrac
            t_norm = np.nanmean(t_norm, axis=(1, 2, 3))

        elif case == "spec_sequence_detector_collapse" or case == "spec_1D_detector_expand":
            counts_final = np.nansum(counts, axis=(0, 1))
            counts_final[counts_final < 0] = 0
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2, axis=(0, 1)))

            t_norm = t_norm * livefrac
            t_norm = np.nanmean(t_norm, axis=(0, 1, 2))

        elif case == "spec_sequence_detector_expand":
            counts_final = np.nansum(counts, axis=(0))
            counts_final[counts_final < 0] = 0
            counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2, axis=(0)))

            t_norm = t_norm * livefrac
            t_norm = np.nanmean(t_norm, axis=(0))

        counts_uncertainity_pu = PoissonUncertainty(counts_uncertainity_final)

        counts_spectral_axis = SpectralAxis(counts_axis, bin_specification="edges")

        meta = NDMeta()

        time_range_actual = Time([(times_full - 0.5 * t_norm).value, (times_full + 0.5 * t_norm).value])

        ct_de = np.diff(counts_axis.value)

        srm = srm_dict["srm"] * ct_de[None, :]

        meta.add("exposure_time", np.sum(t_norm))
        meta.add("geo_area", srm_dict["geo_area"])
        meta.add("angle", flare_angle)
        meta.add("distance", distance)
        meta.add("srm", srm)
        meta.add("ph_axis", srm_dict["ph_axis"] * u.keV)
        meta.add("time_range", time_range_actual)

        spec_1d = Spectrum(
            data=counts_final, uncertainty=counts_uncertainity_pu, spectral_axis=counts_spectral_axis, meta=meta
        )

        return spec_1d

    @staticmethod
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

    @staticmethod
    def _srm_format_flat_or_ranges(indices, case):
        """
        Put detector or pixel indices into the form `get_masked_srm` expects.

        Parameters
        ----------
        indices : list, numpy.ndarray or None
            Flat indices, or a list of [start, end] pairs.
        case : str
            Spectrum case (see `_return_spec_object`). For the two
            '..._detector_collapse' cases ranges are expanded into one flat list;
            for the two '..._detector_expand' cases, into one array per range.

        Returns
        -------
        list or None
            An empty list if `indices` is None, `indices` unchanged if they are
            flat, otherwise the expanded ranges. None if the ranges are given as
            tuples or `case` is not recognised.
        """

        if indices is None:
            return []

        elif isinstance(indices[0], (int, np.integer)):
            return indices

        elif isinstance(indices[0], (list, np.ndarray)):
            indices = ScienceData._indices_expand_ranges(indices)

            if case in ("spec_1D_detector_collapse", "spec_sequence_detector_collapse"):
                return [idx for ls in indices for idx in ls]

            elif case in ("spec_1D_detector_expand", "spec_sequence_detector_expand"):
                return indices

    @staticmethod
    def _srm_det_pix_indices_format(detector_indices, pixel_indices, case):
        """
        Format detector and pixel indices for `get_masked_srm`.

        Applies `_srm_format_flat_or_ranges` to each.

        Parameters
        ----------
        detector_indices : list, numpy.ndarray or None
            Flat detector indices, or a list of [start, end] pairs.
        pixel_indices : list, numpy.ndarray or None
            Flat pixel indices, or a list of [start, end] pairs.
        case : str
            Spectrum case (see `_return_spec_object`).

        Returns
        -------
        detector_indices : list
            Formatted detector indices.
        pixel_indices : list
            Formatted pixel indices.
        """

        det_formatted = ScienceData._srm_format_flat_or_ranges(detector_indices, case)
        pix_formatted = ScienceData._srm_format_flat_or_ranges(pixel_indices, case)

        return det_formatted, pix_formatted

    @staticmethod
    def _get_sunkit_spex_spectrum(
        product,
        detector_indices,
        pixel_indices,
        sci_data,
        flare_location,
        flare_angle,
        systematic,
        detector_sum=True,
        rcr=None,
        bkg=False,
        srm_e_min=3.5 * u.keV,
    ):
        """
        Turn the selected data into `sunkit_spex` spectra.

        What is returned depends on the number of time bins and on `detector_sum`:

        =========  ============  ===============================================
        time bins  detector_sum  returns
        =========  ============  ===============================================
        1          True          one `Spectrum`
        several    True          `NDCubeSequence` of spectra, one per time bin
        1          False         `NDCollection` of spectra, keyed by detector
        several    False         `NDCollection` of `NDCubeSequence`, keyed by
                                 detector
        =========  ============  ===============================================

        A spectral response matrix is computed with `product.get_masked_srm` for
        each detector (or for the summed detectors), once per RCR state present.
        Spectrogram products are always treated as ``detector_sum=True``, with the
        detectors and pixels taken from the product's masks.

        Parameters
        ----------
        product : ScienceData
            Product the data came from, used for the spectral response, the flare
            angle and the spacecraft distance (``meta["DSUN_OBS"]``).
        detector_indices : numpy.ndarray
            Detector selection, flat or [start, end] pairs. With
            ``detector_sum=False`` each flat index or range gives one spectrum.
        pixel_indices : numpy.ndarray
            Pixel selection, flat or [start, end] pairs.
        sci_data : tuple
            The 10-element tuple from `_data_select`.
        flare_location : dict or None
            Flare position with keys 'stx' (STIX-frame Tx/Ty) and 'hpc'. If None,
            no flare angle is computed and `get_masked_srm` gets no flare position.
        flare_angle : astropy.units.Quantity or None
            Flare angle. If None and `flare_location` is given, it is computed with
            `_flare_angle`.
        systematic : bool
            Not used; the systematic uncertainty is added in `_data_select`.
        detector_sum : bool, optional
            Sum detectors into one spectrum (True, default), or make one spectrum
            per detector or detector range (False).
        rcr : array_like, optional
            Not used; the RCR states are taken from `sci_data`.
        bkg : bool, optional
            Passed on to `_return_spec_object`, which does not use it.
        srm_e_min : astropy.units.Quantity or None, optional
            Lower energy limit passed to `get_masked_srm`.

        Returns
        -------
        sunkit_spex.spectrum.Spectrum, ndcube.NDCubeSequence or ndcube.NDCollection
            See the table above.
        """

        counts, counts_uncertainity, t_norm, e_norm, livefrac, _, elut_cor_fac, times_full, energies, rcr = sci_data

        if flare_location is not None:
            flare_location_stx = np.array([flare_location["stx"].Tx.value, flare_location["stx"].Ty.value])
            if flare_angle is None:
                flare_angle = product._flare_angle(product, flare_location)
        else:
            flare_location_stx = None
            flare_angle = None

        distance = (product.meta["DSUN_OBS"] * u.m).to(u.AU)
        rcr_unique = np.unique(rcr)

        shape = np.shape(product.data["counts"])

        if len(shape) < 4:
            detector_indices = np.where(product.detector_masks.masks == 1)[1]
            pixel_indices = np.where(product.pixel_masks.masks == 1)[1]
            detector_sum = True

        if detector_sum:
            if np.shape(counts)[0] == 1:
                case = "spec_1D_detector_collapse"

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(
                    detector_indices, pixel_indices, case
                )

                srm_dict = product.get_masked_srm(
                    flare_location=flare_location_stx,
                    detector_indices_input=detector_indices_srm,
                    pixel_indices_input=pixel_indices_srm,
                    rcr=rcr_unique[0],
                    srm_e_min=srm_e_min,
                )

                return ScienceData._return_spec_object(case, sci_data, flare_angle, distance, srm_dict, bkg)

            else:
                case = "spec_sequence_detector_collapse"

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(
                    detector_indices, pixel_indices, case
                )

                rcr_unique = np.unique(rcr)

                srm_dict_by_rcr = {
                    rcr_val: product.get_masked_srm(
                        flare_location=flare_location_stx,
                        detector_indices_input=detector_indices_srm,
                        pixel_indices_input=pixel_indices_srm,
                        rcr=rcr_val,
                        srm_e_min=srm_e_min,
                    )
                    for rcr_val in rcr_unique
                }

                spec_list_working = []

                for i in range(np.shape(counts)[0]):
                    (
                        counts,
                        counts_uncertainity,
                        t_norm,
                        e_norm,
                        livefrac,
                        _,
                        elut_cor_fac,
                        times_full,
                        energies,
                        rcr,
                    ) = sci_data

                    sci_data_indexed = (
                        counts[i, ...],
                        counts_uncertainity[i, ...],
                        t_norm[i, ...],
                        e_norm,
                        livefrac[i, ...],
                        _,
                        elut_cor_fac,
                        times_full[i, ...],
                        energies,
                        rcr,
                    )

                    spec_1d = ScienceData._return_spec_object(
                        case, sci_data_indexed, flare_angle, distance, srm_dict_by_rcr[int(rcr[i][0])], bkg
                    )

                    spec_list_working.append(spec_1d)

                spec_sequence = NDCubeSequence(
                    spec_list_working, meta={"detector": "det1", "instrument": "STIX"}, common_axis=0
                )

                return spec_sequence

        else:
            if np.shape(counts)[0] == 1:
                case = "spec_1D_detector_expand"

                spec_list_working = []

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(
                    detector_indices, pixel_indices, case
                )

                for i in range(np.shape(counts)[1]):
                    srm_dict = product.get_masked_srm(
                        flare_location=flare_location_stx,
                        detector_indices_input=detector_indices_srm[i],
                        pixel_indices_input=pixel_indices_srm,
                        rcr=rcr_unique,
                        srm_e_min=srm_e_min,
                    )

                    (
                        counts,
                        counts_uncertainity,
                        t_norm,
                        e_norm,
                        livefrac,
                        _,
                        elut_cor_fac,
                        times_full,
                        energies,
                        rcr,
                    ) = sci_data

                    sci_data_indexed = (
                        counts[:, i, ...],
                        counts_uncertainity[:, i, ...],
                        t_norm,
                        e_norm,
                        livefrac[:, i, ...],
                        _,
                        elut_cor_fac,
                        times_full,
                        energies,
                        rcr,
                    )

                    spec_1d = ScienceData._return_spec_object(
                        case, sci_data_indexed, flare_angle, distance, srm_dict, bkg
                    )

                    spec_list_working.append((f"{detector_indices[i]}", spec_1d))

                spec_collection = NDCollection(spec_list_working, aligned_axes="all")

                return spec_collection

            else:
                spec_list_collection_working = []

                case = "spec_sequence_detector_expand"

                detector_indices_srm, pixel_indices_srm = ScienceData._srm_det_pix_indices_format(
                    detector_indices, pixel_indices, case
                )

                for i in range(np.shape(counts)[1]):
                    rcr_unique = np.unique(rcr)

                    srm_dict_by_rcr = {
                        rcr_val: product.get_masked_srm(
                            flare_location=flare_location_stx,
                            detector_indices_input=detector_indices_srm[i],
                            pixel_indices_input=pixel_indices_srm,
                            rcr=rcr_val,
                            srm_e_min=srm_e_min,
                        )
                        for rcr_val in rcr_unique
                    }

                    (
                        counts,
                        counts_uncertainity,
                        t_norm,
                        e_norm,
                        livefrac,
                        _,
                        elut_cor_fac,
                        times_full,
                        energies,
                        rcr,
                    ) = sci_data

                    counts = counts[:, i, ...]
                    counts_uncertainity = counts_uncertainity[:, i, ...]
                    livefrac = livefrac[:, i, ...]

                    spec_list_sequence_working = []

                    for j in range(np.shape(counts)[0]):
                        sci_data_indexed = (
                            counts[j, ...],
                            counts_uncertainity[j, ...],
                            t_norm[j, ...],
                            e_norm,
                            livefrac[j, ...],
                            _,
                            elut_cor_fac,
                            times_full[j, ...],
                            energies,
                            rcr,
                        )

                        spec_1d = ScienceData._return_spec_object(
                            case, sci_data_indexed, flare_angle, distance, srm_dict_by_rcr[int(rcr[j][0])], bkg
                        )

                        spec_list_sequence_working.append(spec_1d)

                    spec_sequence = NDCubeSequence(
                        spec_list_sequence_working,
                        meta={"detector": "det1", "instrument": "STIX"},  # sequence-level
                        common_axis=0,
                    )

                    spec_list_collection_working.append((f"{detector_indices[i]}", spec_sequence))

                spec_collection = NDCollection(spec_list_collection_working, aligned_axes="all")

                return spec_collection

    @staticmethod
    def _flare_angle(product, flare_location):
        """
        Compute the flare angle for the product.

        Solar Orbiter's position is taken at the start of the product's time range
        and passed, with the flare position, to `flare_spacecraft_angle`.

        Parameters
        ----------
        product : ScienceData
            Product whose time range sets the spacecraft position.
        flare_location : dict
            Flare position, with key 'hpc' holding its helioprojective coordinate.

        Returns
        -------
        astropy.units.Quantity
            The angle returned by `flare_spacecraft_angle`.
        """

        _, solo_xyz, _ = get_hpc_info(product.time_range.start, product.time_range.start)

        solo = HeliographicStonyhurst(*solo_xyz, obstime=product.time_range.center, representation_type="cartesian")

        flare_angle = flare_spacecraft_angle(solo, flare_location["hpc"])

        return flare_angle

    # @staticmethod
    # def _check_shadowing(product, detector_indices):
    #     """
    #     Warn if the top and bottom pixel rows disagree, a sign of pixel shadowing.

    #     Compares the counts in the top pixel row with the bottom row for the given
    #     detectors, over the first 25 energy bins, and warns if either exceeds the
    #     other by 5% or more. The check only runs if both rows are in the product's
    #     pixel mask.

    #     Parameters
    #     ----------
    #     product : ScienceData
    #         Product to check.
    #     detector_indices : array_like
    #         Detectors to include.

    #     Warns
    #     -----
    #     UserWarning
    #         If the top/bottom or bottom/top ratio is 1.05 or more.
    #     """

    #     tolerance = 1.05

    #     pixels_top = np.arange(0, 4)
    #     pixels_bot = np.arange(4, 9)

    #     pixels_top_bot = np.concatenate([pixels_top, pixels_bot])

    #     pixel_indices_full = np.where(product.pixel_masks.masks == 1)[1]

    #     counts = product.data["counts"]
    #     counts = counts[:, detector_indices, ...]

    #     if set(pixels_top_bot).issubset(set(pixel_indices_full)):
    #         rat_top_bot = counts[:, :, pixels_top, 0:25] / counts[:, :, pixels_bot, 0:25]
    #         rat_bot_top = counts[:, :, pixels_bot, 0:25] / counts[:, :, pixels_top, 0:25]

    #         if rat_top_bot >= tolerance:
    #             warnings.warn(
    #                 f"Top pixel total 5% higher than bottom row with a ratio of {np.round(rat_top_bot, 2)}. Possible pixel shadowing. Recommend using only top pixels for analysis."
    #             )

    #         elif rat_bot_top >= tolerance:
    #             warnings.warn(
    #                 f"Bottom pixel total 5% higher than top row with a ratio of {np.round(rat_bot_top, 2)}. Possible pixel shadowing. Recommend using only top pixels for analysis."
    #             )

    @staticmethod
    def _time_indices_format(time_indices, times, dt, rcr):
        """
        Turn a time selection into integer indices, checking it against the file
        and the RCR states.

        Accepted forms:

        - flat integer indices, e.g. ``[0, 2, 5]``, returned unchanged;
        - flat strings or `~astropy.time.Time`, taken as consecutive range edges:
          ``["2023-01-01T10:00", "2023-01-01T10:05", "2023-01-01T10:10"]`` gives
          two ranges;
        - [start, end] pairs of strings or `~astropy.time.Time`;
        - [start, end] pairs of integer indices.

        Indices and times are checked against the file first. Times are then
        resolved to the data bins that lie wholly inside each range. Flat indices
        warn if the RCR state varies across them; for pairs, an RCR change inside a
        pair raises and a difference between pairs warns. Numpy integers and arrays
        are accepted as well as Python ints and lists.

        Parameters
        ----------
        time_indices : list or numpy.ndarray
            The time selection, in one of the forms above.
        times : astropy.time.Time
            Centre time of each data bin.
        dt : astropy.units.Quantity
            Duration of each data bin.
        rcr : array_like
            RCR state of each data bin.

        Returns
        -------
        list or numpy.ndarray
            Flat integer indices, or a list of [start, end] integer pairs.

        Raises
        ------
        ValueError
            If an index or time is outside the file, if the RCR state changes
            inside a pair, or if the format is not recognised.
        IndexError
            If a time range contains no complete data bin.

        Warns
        -----
        UserWarning
            If the RCR state varies across flat indices or between pairs.
        """

        first = time_indices[0]

        # file limits: start of the first bin to the end of the last bin
        file_start = times[0] - 0.5 * dt[0]
        file_end = times[-1] + 0.5 * dt[-1]

        if isinstance(first, (int, np.integer)):
            ScienceData._check_index_limits(time_indices, len(times))
            ScienceData._rcr_warning(time_indices, rcr)
            return time_indices

        if isinstance(first, (str, Time)):
            bins = [[time_indices[i], time_indices[i + 1]] for i in range(len(time_indices) - 1)]
            ScienceData._check_time_limits(bins, file_start, file_end)
            result = ScienceData._handle_datetime_strings(bins, times, dt)
            ScienceData._handle_nested_pairs(result, rcr)

            return result

        if isinstance(first, (list, tuple, np.ndarray)):
            if isinstance(first[0], (str, Time)):
                ScienceData._check_time_limits(time_indices, file_start, file_end)
                result = ScienceData._handle_datetime_strings(time_indices, times, dt)
                ScienceData._handle_nested_pairs(result, rcr)
                return result
            if len(first) == 2 and all(isinstance(v, (int, np.integer)) for v in first):
                ScienceData._check_index_limits(time_indices, len(times))
                ScienceData._handle_nested_pairs(time_indices, rcr)
                return time_indices
            raise ValueError(f"Nested lists must be [start, end] integer or time pairs, got: {first}")

        raise ValueError(f"Cannot determine format from first element: {first!r}")

    @staticmethod
    def _check_time_limits(bins, file_start, file_end):
        """
        Check that time ranges lie within the file.

        A tolerance of 1 ms absorbs floating-point error in the file limits, so a
        range that starts or ends exactly on the file edge is accepted.

        Parameters
        ----------
        bins : list
            [start, end] pairs of strings, `~astropy.time.Time`, or anything else
            `~astropy.time.Time` accepts.
        file_start : astropy.time.Time
            Start of the first data bin.
        file_end : astropy.time.Time
            End of the last data bin.

        Raises
        ------
        ValueError
            If any start is before `file_start` or any end is after `file_end`.
        """
        tol = 1 * u.ms  # absorbs floating-point error in file_start / file_end
        for start, end in bins:
            if Time(start) < file_start - tol or Time(end) > file_end + tol:
                raise ValueError(
                    f"Requested times [{Time(start).isot} - {Time(end).isot}] fall outside the "
                    f"file time range [{file_start.isot} - {file_end.isot}]."
                )

    @staticmethod
    def _check_index_limits(indices, n_times):
        """
        Check that time indices lie within the file.

        Parameters
        ----------
        indices : array_like
            Flat integer indices or [start, end] pairs.
        n_times : int
            Number of time bins in the file.

        Raises
        ------
        ValueError
            If any index is negative or not less than `n_times`.
        """
        indices = np.asarray(indices)
        bad = indices[(indices < 0) | (indices >= n_times)]
        if bad.size > 0:
            raise ValueError(
                f"Time indices {np.unique(bad).tolist()} are outside the file, which has time indices 0-{n_times - 1}."
            )

    @staticmethod
    def _rcr_warning(time_indices, rcr):
        """
        Warn if the RCR state varies across flat time indices.

        Parameters
        ----------
        time_indices : list of int
            Time indices to check.
        rcr : array_like
            RCR state of each data bin.

        Warns
        -----
        UserWarning
            Once for each index whose RCR state differs from that of the first
            index.
        """

        first_rcr = rcr[time_indices[0]]
        for i in time_indices[1:]:
            if rcr[i] != first_rcr:
                warnings.warn(
                    f"RCR state change detected "
                    f"index {time_indices[0]} has RCR={first_rcr!r}, "
                    f"index {i} has RCR={rcr[i]!r}."
                    f"Use with caution!",
                    stacklevel=4,
                )
        return None

    @staticmethod
    def _rcr_shift(rcr, counts):
        """
        Align the RCR state boundaries with the jumps in the counts.

        The recorded RCR changes can be a few time bins away from where the counts
        actually change. This finds the jumps in the counts and moves the RCR
        boundaries onto them.

        A jump is a change of more than 1e4 between consecutive time bins in the
        counts of energy channel index 2, summed over detectors and pixels.
        Adjacent jump indices are merged into one. The time axis is then split at
        the jumps and each segment is given the next RCR state in order.

        Parameters
        ----------
        rcr : array_like
            Recorded RCR state of each time bin.
        counts : astropy.units.Quantity
            Counts, shape (time, detector, pixel, energy) or (time, energy).

        Returns
        -------
        numpy.ndarray
            The aligned RCR state of each time bin, or `rcr` unchanged if it has
            only one state.

        Raises
        ------
        IndexError
            If the RCR state changes but no jump is found in the counts, or if more
            jumps are found than there are RCR changes.
        """

        if np.unique(rcr).size > 1:
            rcr = np.asarray(rcr)

            diffs = rcr[1:] - rcr[:-1]
            q = np.where(diffs != 0)[0]

            index = np.concatenate(([0], q + 1))
            state = rcr[index]

            shape = counts.shape

            if len(shape) < 4:
                counts = counts.reshape(shape[0], 1, 1, shape[-1])

            cts_collapse = np.nansum(counts[:, :, :, 2], axis=(1, 2)).astype(np.int64)

            inds = []

            for i in range(len(cts_collapse) - 1):
                if abs(cts_collapse[i] - cts_collapse[i + 1]).value > 1e4:
                    inds.append(i + 1)

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
                rg = np.full(segment_lengths[i], state[i])
                rcr_shift_lists.append(rg)

            rcr_shifted = np.concatenate(rcr_shift_lists)

            return rcr_shifted

        else:
            return rcr

    @staticmethod
    def _rcr_error(indices, rcr):
        """
        Raise if the RCR state changes across the given time indices.

        Parameters
        ----------
        indices : list of int
            Time indices to check. An empty list passes.
        rcr : array_like
            RCR state of each data bin.

        Raises
        ------
        ValueError
            If any index has a different RCR state from the first.
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
    def _handle_datetime_strings(time_bin: list[list[str | Time]], times: list[str | Time], dt) -> list[list[int]]:
        """
        Resolve time ranges to the data bins that lie wholly inside them.

        Parameters
        ----------
        time_bin : list
            [start, end] pairs of strings, `~astropy.time.Time`, or anything else
            `~astropy.time.Time` accepts.
        times : astropy.time.Time
            Centre time of each data bin.
        dt : astropy.units.Quantity
            Duration of each data bin.

        Returns
        -------
        list of list of int
            For each range, the [first, last] index of the data bins that start at
            or after its start and end at or before its end.

        Raises
        ------
        ValueError
            If a range does not have exactly two elements.
        IndexError
            If a range contains no complete data bin.
        """

        data_bin_start = times - (0.5 * dt)
        data_bin_end = times + (0.5 * dt)

        results = []
        for n, pair in enumerate(time_bin):
            if len(bin) != 2:
                raise ValueError(
                    f"Each time bin must have exactly 2 elements [start, end], got {len(pair)} at index {n}."
                )

            bin_start = Time(pair[0])
            bin_end = Time(pair[1])

            matched = [
                i for i, t in enumerate(times) if (bin_start <= data_bin_start[i]) and (data_bin_end[i] <= bin_end)
            ]

            results.append([matched[0], matched[-1]])

        return results

    @staticmethod
    def _handle_nested_pairs(time_indices: list[list[int]], rcr: list) -> list[list[int]]:
        """
        Check [start, end] time index pairs for RCR changes.

        Parameters
        ----------
        time_indices : list of list of int
            [start, end] time index pairs.
        rcr : array_like
            RCR state of each data bin.

        Returns
        -------
        list of list of int
            `time_indices`, unchanged.

        Raises
        ------
        ValueError
            If the RCR state changes inside any pair.

        Warns
        -----
        UserWarning
            If the pairs are in different RCR states from each other.
        """
        # Check within each pair
        for pair in time_indices:
            indices_in_pair = list(range(pair[0], pair[1] + 1))
            ScienceData._rcr_error(indices_in_pair, rcr)

        # Warn if RCR state differs across pairs
        pair_representatives = [rcr[pair[0]] for pair in time_indices]
        if len(set(pair_representatives)) > 1:
            warnings.warn(
                f"RCR state differs across nested pairs: "
                f"{[f'pair {n}={r!r}' for n, r in enumerate(pair_representatives)]}.",
                stacklevel=4,
            )

        return time_indices

    @staticmethod
    def _find_bin_index(start, end, e_low, e_high):
        """
        Find the energy bins whose centres lie in an energy range.

        A bin belongs to the range its centre, ``(e_low + e_high) / 2``, falls in,
        inclusive at both ends. The open top bin has a NaN upper edge and so is
        never selected.

        Parameters
        ----------
        start : float
            Lower edge of the range, in keV.
        end : float
            Upper edge of the range, in keV.
        e_low : numpy.ndarray
            Lower edge of each bin, in keV.
        e_high : numpy.ndarray
            Upper edge of each bin, in keV.

        Returns
        -------
        list of int
            ``[first, last]`` index of the selected bins.

        Raises
        ------
        ValueError
            If no bin centre lies in the range.
        """

        e_centre = (e_low + e_high) / 2

        matches = np.where((e_centre >= start) & (e_centre <= end))[0]

        if matches.size == 0:
            raise ValueError(
                f"Energy range [{start} - {end}] keV does not contain the centre of any product energy bin."
            )

        return [np.min(matches), np.max(matches)]

    @staticmethod
    def _energy_indices_from_flat_edges(values, e_low, e_high):
        """
        Turn a flat list of energy edges into [start, end] bin index pairs.

        N edges give N - 1 consecutive ranges: ``[5, 10, 25]`` gives 5-10 keV and
        10-25 keV. Each range is resolved with `_find_bin_index`.

        Parameters
        ----------
        values : numpy.ndarray
            Energy edges, in keV.
        e_low : numpy.ndarray
            Lower edge of each bin, in keV.
        e_high : numpy.ndarray
            Upper edge of each bin, in keV.

        Returns
        -------
        list of list of int
            One [start, end] bin index pair per range.
        """
        pairs = []
        for i in range(len(values) - 1):
            idx = ScienceData._find_bin_index(values[i], values[i + 1], e_low, e_high)
            pairs.append(idx)
        return pairs

    @staticmethod
    def _energy_indices_from_range_pairs(values, e_low, e_high):
        """
        Turn [start, end] energy ranges into [start, end] bin index pairs.

        Each range is resolved with `_find_bin_index`.

        Parameters
        ----------
        values : numpy.ndarray
            [start, end] energy ranges, in keV, e.g. ``[[5, 10], [15, 25]]``.
        e_low : numpy.ndarray
            Lower edge of each bin, in keV.
        e_high : numpy.ndarray
            Upper edge of each bin, in keV.

        Returns
        -------
        list of list of int
            One [start, end] bin index pair per range.
        """
        pairs = []
        for start_val, end_val in values:
            idx = ScienceData._find_bin_index(start_val, end_val, e_low, e_high)
            pairs.append(idx)
        return pairs

    @staticmethod
    def _energy_indices_format(energy_indices, energies):
        """
        Turn an energy selection given in energy units into bin index pairs.

        Anything that is not an `~astropy.units.Quantity`, including None, is
        returned unchanged and treated as bin indices already. A Quantity is
        converted to keV and can be:

        - 1D, a flat list of edges: ``[4, 10, 28] * u.keV`` gives 4-10 and
          10-28 keV;
        - 2D, a list of [start, end] ranges: ``[[4, 10], [15, 28]] * u.keV``.

        Each range takes the bins whose centres lie inside it (see
        `_find_bin_index`).

        Parameters
        ----------
        energy_indices : astropy.units.Quantity, list, numpy.ndarray or None
            The energy selection.
        energies : astropy.table.QTable
            Energy table with 'e_low' and 'e_high' columns.

        Returns
        -------
        list of list of int
            [start, end] bin index pairs, numbering the rows of `energies`, or
            `energy_indices` unchanged if it is not a Quantity.

        Raises
        ------
        ValueError
            If a requested energy is outside the file's energy range, if a range
            contains no bin centre, or if the Quantity is neither 1D nor 2D.
        """

        if not isinstance(energy_indices, u.Quantity):
            return energy_indices

        energy_indices = energy_indices.to(u.keV)

        e_low = energies["e_low"].value
        e_high = energies["e_high"].value

        # The requested energies must lie within the range covered by the file. nanmin/nanmax
        # skip the NaN upper edge of the open top bin present in a full energy table.
        e_file_min = np.nanmin(e_low)
        e_file_max = np.nanmax(e_high)

        e_requested_min = np.nanmin(energy_indices.value)
        e_requested_max = np.nanmax(energy_indices.value)

        if e_requested_min < e_file_min or e_requested_max > e_file_max:
            raise ValueError(
                f"Requested energies [{e_requested_min} - {e_requested_max}] keV fall outside the "
                f"energy range of the file [{e_file_min} - {e_file_max}] keV."
            )

        if energy_indices.ndim == 1:
            return ScienceData._energy_indices_from_flat_edges(energy_indices.value, e_low, e_high)

        elif energy_indices.ndim == 2:
            return ScienceData._energy_indices_from_range_pairs(energy_indices.value, e_low, e_high)

        else:
            raise ValueError(
                "energy_indices given as a Quantity must be either 1D "
                "(flat list of bin edges) or 2D (list of [start, end] pairs)."
            )

    @staticmethod
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

    @staticmethod
    def _elut_correction_sort(bins, bins_actual, sunkit_spex_detector_sum, pixel_indices, detector_indices, spec_file):
        """
        Compute the ELUT correction factor for the selected detectors and pixels.

        `bins` and `bins_actual` are averaged over the selected pixels, and the
        factor is ``bins / bins_actual``. Detectors are handled as follows:

        - spectrogram products (`spec_file` True): averaged over the detectors
          given, giving one factor;
        - `sunkit_spex_detector_sum` True: averaged over the selected detectors,
          giving one factor;
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
        sunkit_spex_detector_sum : bool
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
                pixel_indices = ScienceData._indices_expand_ranges(pixel_indices, nest=False)

            bins = np.nanmean(bins[:, :, pixel_indices, :], axis=2, keepdims=True)
            bins_actual = np.nanmean(bins_actual[:, :, pixel_indices, :], axis=2, keepdims=True)

            if sunkit_spex_detector_sum:
                if detector_indices.ndim == 2:
                    detector_indices = ScienceData._indices_expand_ranges(detector_indices, nest=False)

                bins = np.nanmean(bins[:, detector_indices, :, :], axis=1, keepdims=True)

                bins_actual = np.nanmean(bins_actual[:, detector_indices, :, :], axis=1, keepdims=True)

                elut_cor_fac = bins / bins_actual

            elif detector_indices.ndim == 1:
                elut_cor_fac = bins / bins_actual

            elif detector_indices.ndim == 2:
                detector_indices = ScienceData._indices_expand_ranges(detector_indices, nest=True)

                elut_cor_fac = ScienceData._normalize_elut_by_group_detector_mean(bins, bins_actual, detector_indices)

        return elut_cor_fac

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
        sunkit_spex_detector_sum=True,
        srm_e_min=3.5 * u.keV,
    ):
        """
        Return the selected data, with optional livetime, ELUT and background
        corrections, as arrays or as `sunkit_spex` spectra.

        Parameters
        ----------
        vtype : {'c', 'cr', 'dcr'}, optional
            Normalisation of the returned counts and uncertainties: counts ('c'),
            count rate in ct/s ('cr'), or differential count rate in ct/(s keV)
            ('dcr', default). With `livetime_correction`, the rates are also divided
            by the livetime fraction. Ignored if `sunkit_spex_spectrum` is True.
        time_indices : list, numpy.ndarray or astropy.time.Time, optional
            Flat integer indices keep those time bins, and [start, end] integer
            pairs sum each range. Strings or `~astropy.time.Time` are accepted as
            consecutive range edges or as [start, end] pairs, and resolve to the
            data bins that lie wholly inside each range. See
            `_time_indices_format`.
        energy_indices : list, numpy.ndarray or astropy.units.Quantity, optional
            Flat integer indices keep those energy bins, and [start, end] pairs sum
            each range. Indices number the rows of the product's energy table
            (`energies`), which only holds the bins in the energy mask. Energies can
            also be given as a Quantity, either flat edges (``[4, 10, 28] * u.keV``)
            or [start, end] ranges (``[[4, 10], [15, 28]] * u.keV``); each range
            takes the bins whose centres lie inside it. Ignored, with a warning, if
            `sunkit_spex_spectrum` is True.
        detector_indices : list, numpy.ndarray or str, optional
            Flat indices keep those detectors, and [start, end] pairs sum each
            range. Two labels are accepted: "top24", the 24 imaging detectors used
            for spectroscopy in STIX-GSW, and "bkg", the background detector
            (index 9) with its small-aperture pixels [2, 5]. None (default) uses
            every detector in the product. Ignored for spectrogram products. With
            `sunkit_spex_spectrum`, the CFL detector (index 8) can not be used, and
            the BKG detector only on its own.
        pixel_indices : list or numpy.ndarray, optional
            Flat indices keep those pixels, and [start, end] pairs sum each range.
            None (default) uses every pixel in the product, or [2, 5] with
            ``detector_indices="bkg"``. Ignored for spectrogram products.
        sum_all_times : bool, optional
            If True and `time_indices` gives [start, end] ranges, sum the ranges
            into one time bin. Default False.
        livetime_correction : bool, optional
            Apply the livetime correction. Default True; forced to True when `bkg`
            is given.
        elut_correction : bool, optional
            Apply the ELUT correction. Default True.
        sunkit_spex_spectrum : bool, optional
            Return `sunkit_spex` spectra instead of arrays. The data are then
            returned as counts regardless of `vtype`. Default False.
        flare_location : dict, optional
            Flare position with keys 'stx' and 'hpc', used for the spectral
            response when `sunkit_spex_spectrum` is True.
        flare_angle : astropy.units.Quantity, optional
            Flare angle. If None and `flare_location` is given, it is computed.
        bkg : ScienceData, optional
            Background product to subtract.
        sunkit_spex_systematic_error : bool, optional
            Add an energy-dependent systematic uncertainty: 7% below 7 keV, 5% from
            7 to 10 keV and 3% from 10 keV. Default False.
        sunkit_spex_detector_sum : bool, optional
            Sum detectors into one spectrum (True, default), or keep one per
            detector or detector range (False). Also sets how the ELUT factor and
            the background livetime are averaged.
        srm_e_min : astropy.units.Quantity, float, bool or None, optional
            Lower energy limit passed to `get_masked_srm`. A float is taken as keV,
            True means 3.5 keV and False means None. Default 3.5 keV.

        Returns
        -------
        tuple
            If `sunkit_spex_spectrum` is False: ``(counts, counts_var, t_norm,
            e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies,
            rcr)``, with `counts` and `counts_var` normalised according to
            `vtype`. `counts_var` holds the 1-sigma uncertainty, not the variance.
        sunkit_spex.spectrum.Spectrum, ndcube.NDCubeSequence or ndcube.NDCollection
            If `sunkit_spex_spectrum` is True; see `_get_sunkit_spex_spectrum`.

        Raises
        ------
        ValueError
            If a selection is outside the file or the product's energy table, if a
            detector label is not recognised, if ``detector_indices="bkg"`` is given
            with pixels other than [2, 5], if a time range spans an RCR change, or
            if `vtype` is not 'c', 'cr' or 'dcr'. With `sunkit_spex_spectrum`, also
            if the CFL detector is selected or the BKG detector is combined with
            other detectors.
        """

        rcr = self.rcr_shifted

        if isinstance(srm_e_min, bool):
            srm_e_min = 3.5 * u.keV if srm_e_min else None
        elif isinstance(srm_e_min, u.Quantity):
            srm_e_min = srm_e_min.to(u.keV)
        elif srm_e_min is not None:
            srm_e_min = float(srm_e_min) * u.keV

        if energy_indices is not None:
            if sunkit_spex_spectrum:
                energy_indices = None
                warnings.warn(
                    "sunkit_spex_spectrum == True and so energy_indices set to None",
                    stacklevel=2,
                )
            else:
                energy_indices = self._energy_indices_format(energy_indices, self.energies)

        if time_indices is not None:
            time_indices = self._time_indices_format(time_indices, self.times, self.durations, rcr)

        detector_indices, pixel_indices, energy_indices = self._indices_check(
            self, detector_indices, pixel_indices, energy_indices
        )

        if elut_correction:
            _, _, bins, bins_actual = get_elut_correction(np.array(self.energies["channel"]), self)

            if len(self.data["counts"].shape) < 4:
                detector_indices_elut = np.where(self.detector_masks.masks == 1)[1]
                pixel_indices_elut = np.where(self.pixel_masks.masks == 1)[1]
                spec_file = True
            else:
                detector_indices_elut = detector_indices
                pixel_indices_elut = pixel_indices
                spec_file = False

            elut_cor_fac = ScienceData._elut_correction_sort(
                bins, bins_actual, sunkit_spex_detector_sum, pixel_indices_elut, detector_indices_elut, spec_file
            )

            warnings.warn(
                "ELUT correction factor is always averaged over the used pixels"
                "but can be given detector-wise or detector averaged.",
                stacklevel=2,
            )

        else:
            elut_cor_fac = None

        if bkg:
            livetime_correction = True

            energy_indices_bkg = self._energies_bkg_sub(self, bkg)

            pixel_indices_bkg, detector_indices_bkg = self._bkg_indices_check(self, bkg)

        if livetime_correction:
            warnings.warn(
                "If livetime_correction=True livetime is applied avergaed across detectors to be consistent with IDL approach.",
                stacklevel=2,
            )

            livefraction_sci, livefraction_sci_error = self._livefrac(self, elut_cor_fac, pixel_indices)

            if bkg and isinstance(bkg, ScienceData):
                livefraction_bkg, livefraction_bkg_error = self._livefrac(
                    bkg, elut_cor_fac, pixel_indices_bkg, energy_indices_bkg
                )

        else:
            livefraction_sci = None
            livefraction_sci_error = None

        if not bkg:
            background_boolean = False

            sci_data = self._data_select(
                self,
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
                bkg=background_boolean,
            )

        else:
            background_boolean = True
            warnings.warn(
                "For background subtraction livetime_correction set as True.",
                stacklevel=2,
            )

            sci_data_all = self._bkg_sub(
                self,
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
                rcr,
            )

            sci_data = self._data_select(
                sci_data_all,
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
                bkg=background_boolean,
            )

        if sunkit_spex_spectrum:
            warnings.warn(
                "As sunkit_spex_spectrum = True, all data will be output as counts."
                "Normalisation selection (vtype) will not be taken into account.",
                stacklevel=2,
            )

            sunkit_spex_spectrum = self._get_sunkit_spex_spectrum(
                self,
                detector_indices,
                pixel_indices,
                sci_data,
                flare_location,
                flare_angle,
                systematic=sunkit_spex_systematic_error,
                detector_sum=sunkit_spex_detector_sum,
                rcr=rcr,
                bkg=background_boolean,
                srm_e_min=srm_e_min,
            )

            return sunkit_spex_spectrum

        else:
            counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr = sci_data

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

            counts = counts * norm
            counts_var = counts_var * norm

            return counts, counts_var, t_norm, e_norm, livefrac, livefrac_error, elut_cor_fac, times, energies, rcr

    @staticmethod
    def _match_idl_grid(drm, ph_full, ct_full, e_edges, epsilon=1e-4):
        """
        Put the saved DRM on the photon grid IDL builds for this product.

        IDL's grid is ``get_uniq([transmission_grid, ct_edges], epsilon=1e-4)``. The saved grid is the
        transmission grid plus all STIX count edges (``ct_full``), so dropping the ones this product
        doesn't have gives IDL's grid exactly, and each new bin is a run of whole saved bins.

        Parameters
        ----------
        drm : numpy.ndarray
            Saved DRM, (photon, count), counts / keV / photon.
        ph_full : numpy.ndarray
            Bin edges of the saved DRM in keV (same for both axes).
        ct_full : numpy.ndarray
            STIX count edges the saved grid contains, in keV.
        e_edges : numpy.ndarray
            This product's count edges, in keV.

        Returns
        -------
        drm_new : numpy.ndarray
            DRM on IDL's grid in counts per count bin per photon: count bins summed, photon bins
            averaged (weighted by width).
        ph_edges : numpy.ndarray
            IDL's photon grid for this product.
        """
        missing = ~np.isclose(e_edges[:, None], ph_full[None, :], atol=epsilon).any(axis=1)
        if missing.any():
            raise ValueError(f"Energy edges {e_edges[missing]} are not on the saved DRM grid.")

        is_count_edge = np.isclose(ph_full[:, None], ct_full[None, :], atol=epsilon).any(axis=1)
        in_product = np.isclose(ph_full[:, None], e_edges[None, :], atol=epsilon).any(axis=1)

        keep = ~is_count_edge | in_product  # = IDL's get_uniq([transmission_grid, e_edges])
        ph_edges = ph_full[keep]
        starts = np.flatnonzero(keep)[:-1]  # first saved bin in each new bin

        widths = np.diff(ph_full)
        summed = np.add.reduceat(drm * widths[:, None] * widths[None, :], starts, axis=0)
        summed = np.add.reduceat(summed, starts, axis=1)

        return summed / np.diff(ph_edges)[:, None], ph_edges

    def get_masked_srm(self, flare_location, detector_indices_input, pixel_indices_input, rcr, srm_e_min=3.5 * u.keV):
        """
        Build the spectral response matrix (SRM) for a set of detectors and pixels.

        The detector response matrix (DRM) is read from the calibration file
        ``stx_detector_response_matrix.fits.gz``: ``drm.smatrix`` from STIX-GSW
        ``stx_build_drm``, without hole tailing, on a photon grid made of the STIX
        transmission-file grid plus all 31 STIX count edges, together with its bin
        edges, the count edges and the CdTe cross sections needed for the tailing.

        The steps follow STIX-GSW ``stx_build_pixel_drm``:

        1. The count edges are taken from the product's energy table, without the
           0 keV lower edge and the open top bin.
        2. The DRM is put on the photon grid IDL builds for these edges (see
           `_match_idl_grid`).
        3. The fine count bins are summed into the product's count bins.
        4. Hole tailing is applied along the photon axis on this grid (see
           `stixpy.calibration.detector.tailing_matrix`).
        5. Each photon row is multiplied by the detector transmission at the bin
           centre, averaged over the detectors, with the attenuator in if `rcr` is
           not 0, and by the grid transmission for `flare_location`, averaged over
           the detectors. For the background detector (index 9) on its own, the
           grid transmission is the mean over the selected pixels from
           ``real_bkg_grid_transmission.txt``.
        6. The result is divided by the count bin widths.

        Parameters
        ----------
        flare_location : array_like or None
            Flare position ``[Tx, Ty]`` in the STIX frame, passed to
            `get_grid_transmission`.
        detector_indices_input : int or array_like
            Detectors the response is for.
        pixel_indices_input : int or array_like
            Pixels the response is for.
        rcr : int or array_like
            RCR state, 0-7, as a number or a one-element array. Sets whether the
            attenuator is in and the fraction of the pixel area that is active.
        srm_e_min : astropy.units.Quantity, bool or None, optional
            Photon energies below this are cut from the SRM and the photon axis.
            True means 3.5 keV, and False or None keeps every photon energy.
            Default 3.5 keV.

        Returns
        -------
        dict
            With keys:

            - 'srm': the response matrix, one row per photon bin and one column per
              count bin, in counts per keV of count energy per photon.
            - 'ph_axis': photon bin edges in keV, as a plain array.
            - 'geo_area': geometric area in cm^2, the number of detectors times the
              area of the selected pixels, scaled by the fraction of the pixel area
              active in this RCR state.

        Raises
        ------
        ValueError
            If the CFL detector (index 8) is selected, if the BKG detector (index 9)
            is combined with other detectors, or if a count edge of the product is
            not on the DRM grid.
        """

        HERE = Path(__file__).parent
        ROOT = HERE.parent.parent
        PATH_DRM = ROOT / "config" / "data" / "detector" / "stx_detector_response_matrix.fits.gz"
        PATH_BKG_TRANS = ROOT / "config" / "data" / "grid" / "real_bkg_grid_transmission.txt"

        drm = np.array(Table.read(PATH_DRM, hdu=1)["DRM"])
        ph_energies = np.array(Table.read(PATH_DRM, hdu=2)["DRM"])
        ct_energies = np.array(Table.read(PATH_DRM, hdu=3)["DRM"])
        xsec = Table.read(PATH_DRM, hdu=4)

        detector_indices_input = np.atleast_1d(detector_indices_input)
        pixel_indices_input = np.atleast_1d(pixel_indices_input)
        rcr = int(np.asarray(rcr).item())  # callers may pass a 1-element array

        # As in STIX-GSW stx_convert_spectrogram2ospex: the CFL (index 8) has no grid transmission,
        # and the BKG detector (index 9) has its own, so it can only be used on its own.
        if np.isin(8, detector_indices_input):
            raise ValueError("The CFL detector (index 8) can not be selected for spectral fitting.")
        if np.isin(9, detector_indices_input) and detector_indices_input.size > 1:
            raise ValueError(
                "The BKG detector (index 9) can not be selected together with imaging detectors for spectral fitting."
            )

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

        drm_clipped, ph_energies_clipped = self._match_idl_grid(drm, ph_energies, ct_energies, e_edges)

        pixel_areas_full = STIX_INSTRUMENT.pixel_config["Area"].to("cm2")

        pixel_areas = pixel_areas_full[pixel_indices_input].value

        area_scale = np.size(detector_indices_input) * np.sum(pixel_areas)

        energy_widths = np.diff(ph_energies_clipped)

        e_mids = ph_energies_clipped[:-1] + (energy_widths / 2)

        trans = Transmission()

        if rcr == 0:
            tot_trans = trans.get_transmission(energies=e_mids * u.keV)
        else:
            tot_trans = trans.get_transmission(energies=e_mids * u.keV, attenuator=True)

        rcr_state_all = np.array([0.8096, 0.80961, 0.4048, 0.2024, 0.1012, 0.0396, 0.0198, 0.0099])
        pixel_indices_input_rcr = np.arange(0, 12, 1)

        rcr_state = rcr_state_all[int(rcr)]
        rcr_factor = rcr_state / np.sum(pixel_areas_full[pixel_indices_input_rcr].value)

        attenuation = np.mean([np.asarray(tot_trans[f"det-{det}"]) for det in detector_indices_input], axis=0)

        drm_new = []

        for j in range(np.shape(drm_clipped)[0]):
            working = []

            for i in range(len(e_edges) - 1):
                indices_sum = np.where((ph_energies_clipped >= e_edges[i]) & (ph_energies_clipped < e_edges[i + 1]))[0]

                tot = drm_clipped[j, indices_sum].sum(axis=0)

                working.append(tot)

            drm_new.append(working)

        drm_new = np.array(drm_new)

        tailing = tailing_matrix(ph_energies_clipped, np.array(xsec["ENERGY"]), np.array(xsec["XSEC"]))
        drm_new = (tailing.T @ drm_new) * attenuation[:, None]

        grid_transmission = get_grid_transmission(e_mids, detector_indices_input, flare_location)

        if detector_indices_input.size == 1 and detector_indices_input[0] == 9:
            bkg_transmission = Table.read(PATH_BKG_TRANS, format="ascii.no_header", comment="[;~]")["col1"]
            bkg_transmission_mean = np.nanmean(bkg_transmission[pixel_indices_input])
            grid_transmission = np.broadcast_to(bkg_transmission_mean, (np.shape(grid_transmission)[0], 1))

        grid_transmission = grid_transmission.mean(axis=1)

        srm = (drm_new * grid_transmission[:, None]) / ct_e_diff[None, :]

        if srm_e_min is True:
            srm_e_min = 3.5 * u.keV
        elif srm_e_min is False:
            srm_e_min = None

        if srm_e_min is not None:
            # ph_energies_clipped holds bin EDGES (one longer than the SRM's row
            # count), so find the cut on the lower edges and apply to both.
            lower = ph_energies_clipped[:-1] if ph_energies_clipped.size == srm.shape[0] + 1 else ph_energies_clipped
            i0 = int(np.searchsorted(lower, srm_e_min.value, side="left"))
            srm = srm[i0:]
            ph_energies_clipped = ph_energies_clipped[i0:]

        return {"srm": srm, "ph_axis": ph_energies_clipped, "geo_area": area_scale * rcr_factor}

    @staticmethod
    def _merge_removed_edges(drm, ph_energies, indices_to_remove):
        """Remove grid edges by merging the bins either side, instead of dropping a row/column."""
        widths = np.diff(ph_energies)
        keep = np.ones(ph_energies.size, bool)
        keep[indices_to_remove] = False
        keep[[0, -1]] = True  # never drop the outer grid edges
        new_edges = ph_energies[keep]
        n = new_edges.size - 1

        group = np.searchsorted(new_edges, ph_energies[:-1], side="right") - 1

        # columns (count energy): per-keV -> counts, then add fine bins together
        cols = np.zeros((drm.shape[0], n))
        np.add.at(cols.T, group, (drm * widths[None, :]).T)

        # rows (photon energy): width-weighted average of fine bins
        rows = np.zeros((n, n))
        np.add.at(rows, group, cols * widths[:, None])
        wsum = np.zeros(n)
        np.add.at(wsum, group, widths)

        return rows / wsum[:, None], new_edges

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
