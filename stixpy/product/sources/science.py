from collections import defaultdict
from itertools import product

from pathlib import Path

import astropy.units as u
import numpy as np

from astropy.io import fits
from astropy.table import QTable, vstack
from astropy.time import Time, TimeDelta
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import date2num, HourLocator, DateFormatter
from matplotlib.widgets import Slider
from sunpy.time.timerange import TimeRange

__all__ = ['ScienceData', 'RawPixelData', 'CompressedPixelData', 'SummedCompressedPixelData',
           'Visibility', 'Spectrogram', 'TimesSeriesPlotMixin', 'SpectrogramPlotMixin',
           'PixelPlotMixin', 'PPrintMixin', 'IndexMasks', 'DetectorMasks', 'PixelMasks',
           'EnergyMasks']

from stixpy.product.product import L1Product

quantity_support()


class PPrintMixin:
    """
    Provides pretty printing for index masks.
    """
    @staticmethod
    def _pprint_indices(indices):
        groups = np.split(np.r_[:len(indices)], np.where(np.diff(indices) != 1)[0] + 1)
        out = ''
        for group in groups:
            if group.size < 3:
                out += f'{indices[group]}'
            else:
                out += f'[{indices[group[0]]}...{indices[group[-1]]}]'

        return out


class IndexMasks(PPrintMixin):
    """

    Attributes
    ----------
    masks : 'numpy.ndarray`
        The mask arrays
    indices : `numpy.ndarray'
        The indices the mask/s applies to

    """
    def __init__(self, mask_array):
        masks = np.unique(mask_array, axis=0)
        indices = [np.argwhere(np.all(mask_array == mask, axis=1)).reshape(-1)
                   for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f'{self.__class__.__name__}\n'
        for m, i in zip(self.masks, self.indices):
            text += f'    {self._pprint_indices(i)}: ' \
                    f'[{",".join(np.where(m, np.arange(m.size), np.full(m.size, "_")))}]\n'
        return text


class DetectorMasks(IndexMasks):
    """
    Detector Index Masks
    """
    pass


class EnergyMasks(IndexMasks):
    """
    Energy Index Mask
    """
    pass


class PixelMasks(PPrintMixin):
    """
    Pixel Index Masks
    """
    def __init__(self, pixel_masks):
        masks = np.unique(pixel_masks, axis=0)
        indices = []
        if masks.ndim == 2:
            indices = [np.argwhere(np.all(pixel_masks == mask, axis=1)).reshape(-1)
                       for mask in masks]
        elif masks.ndim == 3:
            indices = [np.argwhere(np.all(pixel_masks == mask, axis=(1, 2))).reshape(-1)
                       for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f'{self.__class__.__name__}\n'
        for m, i in zip(self.masks, self.indices):
            text += f'    {self._pprint_indices(i)}: ' \
                    f'[{str(np.where(m.shape[0], m, np.full(m.shape, "_")))}]\n'
        return text


class SpectrogramPlotMixin:
    """
    Spectrogram plot mixin providing spectrogram plotting for pixel data.
    """
    def plot_spectrogram(self, axes=None, time_indices=None, energy_indices=None,
                         detector_indices='all', pixel_indices='all', **plot_kwargs):
        """
        Plot a spectrogram for the selected time and energies.

        Parameters
        ----------
        axes : optional `matplotlib.axes`
            The axes the plot the spectrogram.
        time_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
        **plot_kwargs : `dict`
            Any additional arguments are passed to :meth:`~matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        `matplotlib.axes`
        """
        counts_shape = self.data['counts'].shape
        if len(counts_shape) != 4:
            # if spectrogram can't do anything with pixel or detector indices
            if detector_indices != 'all' or pixel_indices != 'all':
                raise ValueError('Detector and or pixel indices have can not be used with spectrogram')

            pid = None
            did = None
        else:

            if detector_indices == 'all':
                did = [[0, 31]]
            else:
                det_idx_arr = np.array(detector_indices)
                if det_idx_arr.ndim == 1 and det_idx_arr.size != 1:
                    raise ValueError('Spectrogram plots can only show data from a single '
                                     'detector or summed over a number of detectors')
                elif det_idx_arr.ndim == 2 and det_idx_arr.shape[0] != 1:
                    raise ValueError('Spectrogram plots can only one sum '
                                     'detector or summed over a number of detectors')
                did = detector_indices

            if pixel_indices == 'all':
                pid = [[0, 11]]
            else:
                pix_idx_arr = np.array(pixel_indices)
                if pix_idx_arr.ndim == 1 and pix_idx_arr.size != 1:
                    raise ValueError('Spectrogram plots can only show data from a single '
                                     'detector or summed over a number of detectors')
                elif pix_idx_arr.ndim == 2 and pix_idx_arr.shape[0] != 1:
                    raise ValueError('Spectrogram plots can only one sum '
                                     'detector or summed over a number of detectors')
                pid = pixel_indices

        counts, errors, times, timedeltas, energies = self.get_data(detector_indices=did,
                                                                    pixel_indices=pid,
                                                                    time_indices=time_indices,
                                                                    energy_indices=energy_indices)

        e_edges = np.hstack([energies['e_low'], energies['e_high'][-1]])
        t_edges = Time(np.concatenate(
            [times - timedeltas.reshape(-1) / 2, times[-1] + timedeltas.reshape(-1)[-1:] / 2]))

        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = axes.get_figure()

        pcolor_kwargs = {'norm':LogNorm(), 'shading':'flat'}
        pcolor_kwargs.update(plot_kwargs)
        im = axes.pcolormesh(t_edges.datetime, e_edges[1:-1], counts[:, 0, 0, 1:-1].T.value,
                             **pcolor_kwargs)

        #axes.colorbar(im).set_label(format(counts.unit))
        axes.xaxis_date()
        # axes.set_yticks(range(y_lims[0], y_lims[1] + 1))
        # axes.set_yticklabels(labels)
        minor_loc = HourLocator()
        axes.xaxis.set_minor_locator(minor_loc)
        axes.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
        # fig.autofmt_xdate()
        # fig.tight_layout()

        return axes


class TimesSeriesPlotMixin:
    """
    TimesSeries plot mixin providing timeseries plotting for pixel data.
    """
    def plot_timeseries(self, time_indices=None, energy_indices=None, detector_indices='all',
                        pixel_indices='all', axes=None, error_bar=False, **plot_kwarg):
        """
        Plot a times series of the selected times and energies.

        Parameters
        ----------
        axes : optional `matplotlib.axes`
            The axes the plot the spectrogram.
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
        else:
            fig = axes.get_figure()

        if detector_indices == 'all':
            detector_indices = [[0, 31]]

        if pixel_indices == 'all':
            pixel_indices = [[0, 11]]

        counts, errors, times, timedeltas, energies \
            = self.get_data(detector_indices=detector_indices,
                            pixel_indices=pixel_indices,
                            time_indices=time_indices, energy_indices=energy_indices)

        labels = [f'{el.value} - {eh.value}' for el, eh in energies['e_low', 'e_high']]

        n_time, n_det, n_pix, n_energy = counts.shape

        for did, pid, eid in product(range(n_det), range(n_pix), range(n_energy)):
            if error_bar:
                lines = axes.errorbar(times.to_datetime(), counts[:, did, pid, eid], yerr=errors[:, did, pid, eid],
                                      label=labels[eid], **plot_kwarg)
            else:
                lines = axes.plot(times.to_datetime(), counts[:, did, pid, eid],
                                  label=labels[eid], **plot_kwarg)

        axes.set_yscale('log')
        axes.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()

        return lines


class PixelPlotMixin:
    """
    Pixel plot mixin providing pixel plotting for pixel data.
    """
    def plot_pixels(self, time_indices=None, energy_indices=None, fig=None):
        """
        Plot individual pixel data for each detector.

        Parameters
        ----------
        time_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
        fig : optional `matplotlib.figure`
            The figure where to which the pixel plot will be added.

        Returns
        -------
        `matplotlib.figure`
            The figure
        """

        if fig:
            axes = fig.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(10, 5))
        else:
            fig, axes = plt.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(10, 5))

        counts, count_err, times, dt, energies = self.get_data(time_indices=time_indices,
                                                               energy_indices=energy_indices)

        def timeval(val):
            return times[val].isot

        def energyval(val):
            return f"{energies[val]['e_low'].value}-{energies[val]['e_high']}"

        axcolor = 'lightgoldenrodyellow'
        axenergy = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor)
        senergy = SliderCustomValue(axenergy, 'Energy', 0, len(energies)-1, format_func=energyval,
                                    valinit=0, valstep=1)
        axetime = plt.axes([0.15, 0.01, 0.55, 0.03], facecolor=axcolor)
        stime = SliderCustomValue(axetime, 'Time', 0, counts.shape[0]-1, format_func=timeval,
                                  valinit=0, valstep=1)

        pixel_ids = [slice(0, 4), slice(4, 8), slice(8, 12)]
        if counts.shape[2] == 4:
            pixel_ids = [slice(0, 4)]

        containers = defaultdict(list)

        for detector_id in range(32):
            row, col = divmod(detector_id, 8)
            for pixel_id in pixel_ids:
                errbar_cont = axes[row, col].errorbar((0.5, 1.5, 2.5, 3.5),
                                                      counts[0, detector_id, pixel_id, 0],
                                                      yerr=count_err[0, detector_id, pixel_id, 0],
                                                      xerr=0.5, ls='')

                containers[row, col].append(errbar_cont)
                axes[row, col].set_xlim(0, 4)
                axes[row, col].set_title(f'Det {detector_id}')
                axes[row, col].set_xticks([])
                if col != 0:
                    # axs[row, col].set_yticks([])
                    axes[row, col].set_ylabel('')

        def update(_):
            energy_index = senergy.val
            time_index = stime.val
            pids_ = [slice(0, 4), slice(4, 8), slice(8, 12)]
            if counts.shape[2] == 4:
                pids_ = [slice(0, 4)]

            imaging_mask = np.ones(32, bool)
            imaging_mask[8:10] = False

            for did in range(32):
                r, c = divmod(did, 8)
                axes[r, c].set_ylim(0, counts[time_index, imaging_mask, :,
                                              energy_index].max()*1.2)

                for i, pid in enumerate(pids_):
                    lines, caps, bars = containers[r, c][i]
                    lines.set_ydata(counts[time_index, did, pid, energy_index])

                    # horizontal bars at value
                    segs = np.array(bars[0].get_segments())
                    segs[:, 0, 0] = [0., 1., 2., 3.]
                    segs[:, 1, 0] = [1., 2., 3., 4.]
                    segs[:, 0, 1] = counts[time_index, did, pid, energy_index]
                    segs[:, 1, 1] = counts[time_index, did, pid, energy_index]
                    bars[0].set_segments(segs)
                    # vertical bars at +/- error
                    segs = np.array(bars[1].get_segments())
                    segs[:, 0, 0] = [0.5, 1.5, 2.5, 3.5]
                    segs[:, 1, 0] = [0.5, 1.5, 2.5, 3.5]
                    segs[:, 0, 1] = counts[time_index, did, pid, energy_index] \
                        - count_err[time_index, did, pid, energy_index]
                    segs[:, 1, 1] = counts[time_index, did, pid, energy_index] \
                        + count_err[time_index, did, pid, energy_index]
                    bars[1].set_segments(segs)

        senergy.on_changed(update)
        stime.on_changed(update)


class ScienceData(L1Product):
    """
    Basic science data class
    """
    def __init__(self, *, meta, control, data, energies, idb_versions=None):
        """

        Parameters
        ----------
        header : `astropy.fits.Header`
            Fits header
        control : `astropy.table.QTable`
            Fits file control extension
        data :` astropy.table.QTable`
            Fits file data extension
        energies : `astropy.table.QTable`
            Fits file energy extension
        """
        super().__init__(meta=meta, control=control, data=data,
                         energies=energies, idb_versions=idb_versions)

        self.count_type = 'rate'
        if 'detector_masks' in self.data.colnames:
            self.detector_masks = DetectorMasks(self.data['detector_masks'])
        if 'pixel_masks' in self.data.colnames:
            self.pixel_masks = PixelMasks(self.data['pixel_masks'])
        if 'energy_bin_edge_mask' in self.control.colnames:
            self.energy_masks = EnergyMasks(self.control['energy_bin_edge_mask'])
            self.dE = energies['e_high'] - energies['e_low']

    @property
    def time_range(self):
        """
        A `sunpy.time.TimeRange` for the data.
        """
        return TimeRange(self.data['time'][0] - self.data['timedel'][0]/2,
                         self.data['time'][-1] + self.data['timedel'][-1]/2)

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
        return self.data['time']

    @property
    def durtaion(self):
        """
        An `astropy.units.Quantiy` array giving the duration or integration time
        """
        return self.data['timedel']

    def get_data(self, time_indices=None, energy_indices=None, detector_indices=None,
                 pixel_indices=None, sum_all_times=False):
        """
        Return the counts, errors, times, durations and energies for selected data.

        Optionally summing in time and or energy.

        Parameters
        ----------
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
        counts = self.data['counts']
        try:
            counts_var = self.data['counts_comp_err']**2
        except KeyError:
            counts_var = self.data['counts_comp_comp_err'] ** 2
        shape = counts.shape
        if len(shape) < 4:
            counts = counts.reshape(shape[0], 1, 1, shape[-1])
            counts_var = counts_var.reshape(shape[0], 1, 1, shape[-1])

        energies = self.energies[:]
        times = self.times

        if detector_indices is not None:
            detecor_indices = np.asarray(detector_indices)
            if detecor_indices.ndim == 1:
                detector_mask = np.full(32, False)
                detector_mask[detecor_indices] = True
                counts = counts[:, detector_mask, ...]
                counts_var = counts_var[:, detector_mask, ...]
            elif detecor_indices.ndim == 2:
                counts = np.hstack([np.sum(counts[:, dl:dh + 1, ...], axis=1, keepdims=True)
                                    for dl, dh in detecor_indices])

                counts_var = np.concatenate(
                    [np.sum(counts_var[:, dl:dh + 1, ...], axis=1, keepdims=True)
                     for dl, dh in detecor_indices], axis=1)

        if pixel_indices is not None:
            pixel_indices = np.asarray(pixel_indices)
            if pixel_indices.ndim == 1:
                pixel_mask = np.full(12, False)
                pixel_mask[pixel_indices] = True
                counts = counts[..., pixel_mask, :]
                counts_var = counts_var[..., pixel_mask, :]
            elif pixel_indices.ndim == 2:
                counts = np.concatenate([np.sum(counts[..., pl:ph + 1, :], axis=2, keepdims=True)
                                         for pl, ph in pixel_indices], axis=2)

                counts_var = np.concatenate(
                    [np.sum(counts_var[..., pl:ph + 1, :], axis=2, keepdims=True)
                     for pl, ph in pixel_indices], axis=2)

        e_norm = self.dE
        if energy_indices is not None:
            energy_indices = np.asarray(energy_indices)
            if energy_indices.ndim == 1:
                energy_mask = np.full(shape[-1], False)
                energy_mask[energy_indices] = True
                counts = counts[..., energy_mask]
                counts_var = counts_var[..., energy_mask]
                e_norm = self.dE[energy_mask]
                energies = self.energies[energy_mask]
            elif energy_indices.ndim == 2:
                counts = np.concatenate([np.sum(counts[..., el:eh+1], axis=-1, keepdims=True)
                                         for el, eh in energy_indices], axis=-1)

                counts_var = np.concatenate(
                    [np.sum(counts_var[..., el:eh+1], axis=-1, keepdims=True)
                     for el, eh in energy_indices], axis=-1)

                e_norm = np.hstack([(energies['e_high'][eh] - energies['e_low'][el])
                                    for el, eh in energy_indices])

                energies = np.atleast_2d(
                    [(self._energies['e_low'][el].value, self._energies['e_high'][eh].value)
                     for el, eh in energy_indices])
                energies = QTable(energies*u.keV, names=['e_low', 'e_high'])

        t_norm = self.data['timedel']
        if time_indices is not None:
            time_indices = np.asarray(time_indices)
            if time_indices.ndim == 1:
                time_mask = np.full(times.shape, False)
                time_mask[time_indices] = True
                counts = counts[time_mask, ...]
                counts_var = counts_var[time_mask, ...]
                t_norm = self.data['timedel'][time_mask]
                times = times[time_mask]
                # dT = self.data['timedel'][time_mask]
            elif time_indices.ndim == 2:
                new_times = []
                dt = []
                for tl, th in time_indices:
                    ts = times[tl] - self.data['timedel'][tl] * 0.5
                    te = times[th] + self.data['timedel'][th] * 0.5
                    td = te - ts
                    tc = ts + (td * 0.5)
                    dt.append(td.to('s'))
                    new_times.append(tc)
                dt = np.hstack(dt)
                times = Time(new_times)
                counts = np.vstack([np.sum(counts[tl:th+1, ...], axis=0, keepdims=True) for tl, th
                                    in time_indices])

                counts_var = np.vstack([np.sum(counts_var[tl:th+1, ...], axis=0, keepdims=True)
                                        for tl, th in time_indices])
                t_norm = dt

                if sum_all_times and len(new_times) > 1:
                    counts = np.sum(counts, axis=0, keepdims=True)
                    counts_var = np.sum(counts_var, axis=0, keepdims=True)
                    t_norm = np.sum(dt)

        if e_norm.size != 1:
            e_norm = e_norm.reshape(1, 1, 1, -1)

        if t_norm.size != 1:
            t_norm = t_norm.reshape(-1, 1, 1, 1)

        counts_err = np.sqrt(counts*u.ct + counts_var)/(e_norm * t_norm)
        counts = counts / (e_norm * t_norm)

        return counts, counts_err, times, t_norm, energies

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
                self_control_ind_max = data['control_index'].max() + 1
                other.control['index'] = other.control['index'] + self_control_ind_max
                other.data['control_index'] = other.data['control_index'] + self_control_ind_max

                try:
                    [(table.meta.pop('DATASUM'), table.meta.pop('CHECKSUM'))
                     for table in [control, other.control, data, other.data]]
                except KeyError:
                    pass

                control = vstack([control, other.control])
                data = vstack([data, other.data])

            return type(self)(meta=self.meta, control=control, data=data,
                              energies=self.energies, idb_version=self.idb_versions)

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'{self.time_range}' \
               f'    {self.detector_masks}\n' \
               f'    {self.pixel_masks}\n' \
               f'    {self.energy_masks}'


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
        EnergyMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    <BLANKLINE>
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (21, 6, 20) and level == 'L1':
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
        EnergyMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (21, 6, 21) and level == 'L1':
            return True


class SummedCompressedPixelData(ScienceData, PixelPlotMixin, TimesSeriesPlotMixin,
                                SpectrogramPlotMixin):
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
        EnergyMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    """
    pass

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (21, 6, 22) and level == 'L1':
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
    #     EnergyMasks
    #     [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    # <BLANKLINE>
    """
    def __init__(self, *, header, control, data, energies):
        super().__init__(meta=header, control=control, data=data, energies=energies)
        self.pixel_masks = PixelMasks(self.pixels)

    @property
    def pixels(self):
        return np.vstack([self.data[f'pixel_mask{i}'][0] for i in range(1, 6)])

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (21, 6, 23) and level == 'L1':
            return True


class Spectrogram(ScienceData, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Spectrogram from selected pixels, detectors and energies.

    Parameters
    ----------
    header : `astropy.fits.Header`
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
        EnergyMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    <BLANKLINE>
    """
    def __init__(self, *, meta, control, data, energies, idb_versions):
        """

        Parameters
        ----------
        header : astropy.fits.Header
        control : astropy.table.QTable
        data : astropy.table.QTable
        energies : astropy.table.QTable
        """
        super().__init__(meta=meta, control=control, data=data,
                         energies=energies, idb_versions=idb_versions)
        self.count_type = 'rate'
        self.detector_masks = DetectorMasks(self.control['detector_masks'])
        self.pixel_masks = PixelMasks(self.data['pixel_masks'])
        # self.energy_masks = EnergyMasks(self.control['energy_bin_edge_mask'])
        # self.dE = (energies['e_high'] - energies['e_low'])[self.energy_masks.masks[0] == 1]
        # self.dE = np.hstack([[1], np.diff(energies['e_low'][1:]).value, [1]]) * u.keV

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (21, 6, 24) and level == 'L1':
            return True


class SliderCustomValue(Slider):
    """
    A slider with a customisable formatter
    """
    def __init__(self, *args, format_func=None, **kwargs):
        if format_func is not None:
            self._format = format_func
        super().__init__(*args, **kwargs)
