from pathlib import Path

import astropy.units as u
import numpy as np

from astropy.io import fits
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import date2num, HourLocator, DateFormatter
from sunpy.time.timerange import TimeRange
from sunpy.visualization.visualization import peek_show

__all__ = ['ScienceData', 'XrayLeveL0', 'XrayLeveL1', 'XrayLeveL2', 'XrayVisibility',
           'XraySpectrogram']

quantity_support()


class PPrintMixin:
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
    def __init__(self, detector_masks):
        masks = np.unique(detector_masks, axis=0)
        indices = [np.argwhere(np.all(detector_masks == mask, axis=1)).reshape(-1)
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
    pass


class PixelMasks(PPrintMixin):
    def __init__(self, detector_masks):
        masks = np.unique(detector_masks, axis=0)
        if masks.ndim == 2:
            indices = [np.argwhere(np.all(detector_masks == masks[0], axis=1)).reshape(-1)
                       for mask in masks]
        elif masks.ndim == 3:
            indices = [np.argwhere(np.all(detector_masks == masks[0], axis=(1, 2))).reshape(-1)
                       for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f'{self.__class__.__name__}\n'
        for m, i in zip(self.masks, self.indices):
            text += f'    {self._pprint_indices(i)}: ' \
                    f'[{str(np.where(m.shape[0], np.eye(*m.shape), np.full(m.shape, "_")))}]\n'
        return text


class EnergyMasks(IndexMasks):
    pass


class ScienceData:
    """
    Basic science data class
    """
    def __init__(self, *, header, control, data, energies):
        """

        Parameters
        ----------
        header : astropy.fits.Header
        control : astropy.table.QTable
        data : astropy.table.QTable
        energies : astropy.table.QTable
        """
        self.count_type = 'rate'
        self.header = header
        self.control = control
        self.data = data
        self.energies_ = energies
        self.detector_masks = DetectorMasks(self.data['detector_masks'])
        self.pixel_masks = PixelMasks(self.data['pixel_masks'])
        self.energy_masks = EnergyMasks(self.control['energy_bin_mask'])
        self.dE = (energies['e_high'] - energies['e_low'])
        self.dE[[0, -1]] = 1 * u.keV

    @property
    def time_range(self):
        return TimeRange(self.data['time'].min(), self.data['time'].max())

    @property
    def pixels(self):
        return self.pixel_masks

    @property
    def detectors(self):
        return self.detector_masks

    @property
    def energies(self):
        return self.energies_

    @property
    def times(self):
        return self.data['time']

    @property
    def counts(self):
        if self.count_type == 'raw':
            return self.data['counts']
        elif self.count_type == 'count':
            return self.data['counts'] / self.dE
        elif self.count_type == 'rate':
            return self.data['counts'] / self.dE  # (self.data['timedel'].reshape(-1, 1) * self.dE)

    @property
    def var(self):
        if self.count_type == 'raw':
            return self.counts
        elif self.count_type == 'count':
            return self.counts / self.dE
        elif self.count_type == 'rate':
            return self.counts / (self.data['timedel'].reshape(-1, 1) * self.dE)

    def get_data(self, time_indices=None, energy_indices=None, detector_indices=None,
                 pixel_indices=None, sum_all_times=False):
        """
        Return the counts, errors, times, durations and energies for selected data

        optionally summing in time and or energy.

        Parameters
        ----------
        time_indices : `list` or `array-like`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `array-like`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
        detector_indices : `list` or `array-like`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `detector_indices=[0, 2, 5]` would return only the first, third and
            sixth detectors while `detector_indices=[[0, 2],[3, 5]]` would sum the data between.
        pixel_indices : `list` or `array-like`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `pixel_indices=[0, 2, 5]` would return only the first, third and
            sixth pixels while `pixel_indices=[[0, 2],[3, 5]]` would sum the data between.
        sum_all_times : `bool`
            Flag to sum all give time intervals into one

        Returns
        -------
        counts
            Counts
        times
            Times
        energies
            Energies

        """
        counts = self.data['counts']
        counts_var = self.data['counts_err']**2
        shape = counts.shape
        if len(shape) < 4:
            counts = counts.reshape(shape[0], 1, 1, shape[-1])
            counts_var = counts_var.reshape(shape[0], 1, 1, shape[-1])

        energies = self.energies_
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
                energy_mask = np.full(32, False)
                energy_mask[energy_indices] = True
                counts = counts[..., energy_mask]
                counts_var = counts_var[..., energy_mask]
                e_norm = self.dE[energy_mask]
                energies = self.energies_[energy_mask]
            elif energy_indices.ndim == 2:
                counts = np.concatenate([np.sum(counts[..., el:eh+1], axis=-1, keepdims=True)
                                         for el, eh in energy_indices], axis=-1)

                counts_var = np.concatenate(
                    [np.sum(counts_var[..., el:eh+1], axis=-1, keepdims=True)
                     for el, eh in energy_indices], axis=-1)

                e_norm = np.hstack([(energies['e_high'][eh] - energies['e_low'][el])
                                    for el, eh in energy_indices])

                energies = np.atleast_2d(
                    [(self.energies_['e_low'][el].value, self.energies_['e_high'][eh].value)
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
                dT = self.data['timedel'][time_mask]
            elif time_indices.ndim == 2:
                new_times = []
                dT = []
                for tl, th in time_indices:
                    ts = times[tl] - self.data['timedel'][tl] * 0.5
                    te = times[th] + self.data['timedel'][th] * 0.5
                    td = te - ts
                    tc = ts + (td * 0.5)
                    dT.append(td.to('s'))
                    new_times.append(tc)
                dT = np.hstack(dT)
                times = Time(new_times)
                counts = np.vstack([np.sum(counts[tl:th+1, ...], axis=0, keepdims=True) for tl, th
                                    in time_indices])

                counts_var = np.vstack([np.sum(counts_var[tl:th+1, ...], axis=0, keepdims=True)
                                        for tl, th in time_indices])
                t_norm = dT

                if sum_all_times and len(new_times) > 1:
                    counts = np.sum(counts, axis=0, keepdims=True)
                    counts_var = np.sum(counts_var, axis=0, keepdims=True)
                    t_norm = np.sum(dT)

        if e_norm.size != 1:
            e_norm = e_norm.reshape(1, 1, 1, -1)

        if t_norm.size != 1:
            t_norm = t_norm.reshape(-1, 1, 1, 1)

        counts_err = np.sqrt(counts*u.ct + counts_var)/(e_norm * t_norm)
        counts = counts / (e_norm * t_norm)

        return counts, counts_err, times, t_norm, energies

    def concatenate(self, others):
        """
        Concatenate two science products

        Parameters
        ----------
        others

        Returns
        -------

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
                except KeyError as e:
                    pass

                control = vstack([control, other.control])
                data = vstack([data, other.data])

            return type(self)(header=self.header, control=control, data=data,
                              energies=self.energies)

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'{self.time_range}' \
               f'    {self.detector_masks}\n' \
               f'    {self.pixel_masks}\n' \
               f'    {self.energy_masks}'

    @classmethod
    def from_fits(cls, file):
        """
        Parses STIX FITS data files to create.

        Parameters
        ----------
        file : `str` or `pathlib.Path`
            The path to the file you want to parse.
        """

        file = Path(file)
        header = fits.getheader(file)
        control = QTable.read(file, hdu=1)
        data = QTable.read(file, hdu=2)
        data['time'] = Time(header['date_obs']) + data['time']
        energies = QTable.read(file, hdu=3)

        filename = file.name
        if 'xray-l0' in filename:
            return XrayLeveL0(header=header, control=control, data=data, energies=energies)
        elif 'xray-l1' in filename:
            return XrayLeveL1(header=header, control=control, data=data, energies=energies)
        elif 'xray-l2' in filename:
            return XrayLeveL2(header=header, control=control, data=data, energies=energies)
        elif 'xray-l3' in filename:
            return XrayVisibility(header=header, control=control, data=data, energies=energies)
        elif 'spectrogram' in filename:
            return XraySpectrogram(header=header, control=control, data=data, energies=energies)
        else:
            raise TypeError(f'File {file} does not contain pixel data')

    def plot(self):
        counts, count_err, times, dt, energies = self.get_data(time_indices=[[0, -1]])
        fig, axs = plt.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(10, 5))

        for did in range(32):
            row, col = divmod(did, 8)
            axs[row, col].err((0, 1, 2, 3), counts[0, did, 0:4, :].sum(axis=-1), ds='steps-mid')
            axs[row, col].plot((0, 1, 2, 3), counts[0, did, 4:8, :].sum(axis=-1), ds='steps-mid')
            axs[row, col].plot((0, 1, 2, 3), counts[0, did, 8:, :].sum(axis=-1), ds='steps-mid')
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_ylabel('')

    def _update(self, time, energy):
        pass


class XrayLeveL0(ScienceData):
    """
    Uncompressed count data form selected pixels, detectors and energies.
    """
    pass


class XrayLeveL1(ScienceData):
    """
    Compressed count data form selected pixels, detectors and energies.
    """
    pass


class XrayLeveL2(ScienceData):
    """
    Summed, compressed count data form selected pixels, detectors and energies.
    """
    pass


class XrayVisibility(ScienceData):
    """
    Compressed visibilities from selected pixels, detectors and energies.
    """
    def __init__(self, *, header, control, data, energies):
        self.count_type = 'rate'
        self.header = header
        self.control = control
        self.data = data
        self.energies_ = energies
        self.detector_masks = DetectorMasks(self.data['detector_masks'])
        self.pixel_masks = PixelMasks(self.pixels)
        self.energy_masks = EnergyMasks(self.control['energy_bin_mask'])
        self.dE = (energies['e_high'] - energies['e_low'])
        self.dE[[0, -1]] = 1 * u.keV

    @property
    def pixels(self):
        return np.vstack([self.data[f'pixel_mask{i}'][0] for i in range(1, 6)])


class XraySpectrogram(ScienceData):
    """
    Spectrogram selected pixels, detectors and energies.
    """
    def __init__(self, *, header, control, data, energies):
        """

        Parameters
        ----------
        header : astropy.fits.Header
        control : astropy.table.QTable
        data : astropy.table.QTable
        energies : astropy.table.QTable
        """

        self.count_type = 'rate'
        self.header = header
        self.control = control
        self.data = data
        self.energies_ = energies
        self.detector_masks = DetectorMasks(self.control['detector_mask'])
        self.pixel_masks = PixelMasks(self.control['pixel_mask'])
        self.energy_masks = EnergyMasks(self.control['energy_bin_mask'])
        self.dE = np.hstack([[1], np.diff(energies['e_low'][1:]).value, [1]]) * u.keV

    @peek_show
    def peek(self, **kwargs):
        # Now make the plot
        figure = plt.figure()
        self.plot(**kwargs)

        return figure

    def plot(self, axes=None, time_indices=None, energy_indices=None, plot_type='spectrogram'):
        counts, errors, times, energies, dT = self.get_data(time_indices=time_indices,
                                                            energy_indices=energy_indices)

        if axes is None:
            axes = plt.gca()

        fig = axes.get_figure()
        labels = [f'{el.value} - {eh.value}' for el, eh in energies['e_low', 'e_high']]

        if plot_type == 'spectrogram':
            x_lims = date2num(times.to_datetime())
            y_lims = [0, len(energies)-1]

            im = axes.imshow(counts.T.value,
                             extent=[x_lims[0], x_lims[-1], y_lims[0], y_lims[1]],
                             origin='lower', aspect='auto', interpolation='none', norm=LogNorm())

            fig.colorbar(im).set_label(format(counts.unit))
            axes.xaxis_date()
            axes.set_yticks(range(y_lims[0], y_lims[1]+1))
            axes.set_yticklabels(labels)
            minor_loc = HourLocator()
            axes.xaxis.set_minor_locator(minor_loc)
            axes.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
            fig.autofmt_xdate()
            fig.tight_layout()

        elif plot_type == 'timeseries':
            var, *_ = self.get_variance(time_indices=time_indices, energy_indices=energy_indices)
            var = np.sqrt(var) * u.ct**0.5
            [axes.errorbar(times.to_datetime(), counts[:, ech], yerr=var[:, ech],
                           fmt='.', label=labels[ech]) for ech in range(len(energies))]
            axes.set_yscale('log')
            axes.legend()
            axes.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
            fig.autofmt_xdate()
            fig.tight_layout()

        return fig
