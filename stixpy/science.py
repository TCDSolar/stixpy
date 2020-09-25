import astropy.units as u
import numpy as np

from astropy.io import fits
from astropy.table import QTable, vstack
from astropy.time import Time, TimeDelta
from sunpy.time.timerange import TimeRange

__all__ = ['ScienceData', 'XrayLeveL0', 'XrayLeveL1', 'XrayLeveL2', 'XrayVisibity',
           'XraySpectrogram']


class IndexMasks:
    def __init__(self, detector_masks):
        masks = np.unique(detector_masks, axis=0)
        indices = [np.argwhere(np.all(detector_masks == mask, axis=1)).reshape(-1)
                   for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f'{self.__class__.__name__}\n'
        for m, i in zip(self.masks, self.indices):
            text += f'    {self._pprint_indices(i)}: [{",".join(np.where(m, np.arange(m.size), np.full(m.size, "_")))}]\n'
        return text

    @staticmethod
    def _pprint_indices(indices):
        groups = np.split(np.r_[:len(indices)], np.where(np.diff(indices) != 1)[0]+1)
        out = ''
        for group in groups:
            if group.size < 3:
                out += f'{indices[group]}'
            else:
                out += f'[{indices[group[0]]}...{indices[group[-1]]}]'

        return out


class DetectorMasks(IndexMasks):
    pass


class PixelMasks(IndexMasks):
    pass


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
        self.header = header
        self.control = control
        self.data = data
        self.energies_ = energies
        self.detector_masks = DetectorMasks(self.data['detector_masks'])
        self.pixel_masks = PixelMasks(self.data['pixel_masks'])
        self.energy_masks = EnergyMasks(self.control['energy_bin_edge_mask'])

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

    def concatenate(self, others):
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
        Parses STIX FITS data files to create .

        Parameters
        ----------
        filepath : `str` or `pathlib.Path`
            The path to the file you want to parse.
        """

        header = fits.getheader(file)
        control = QTable.read(file, hdu=1)
        data = QTable.read(file, hdu=2)
        data['time'] = Time(header['date_obs']) + data['time']
        energies = QTable.read(file, hdu=3)

        if 'xray-l0' in file:
            return XrayLeveL0(header=header, control=control, data=data, energies=energies)
        elif 'xray-l1' in file:
            return XrayLeveL1(header=header, control=control, data=data, energies=energies)
        elif 'xray-l2' in file:
            return XrayLeveL2(header=header, control=control, data=data, energies=energies)
        elif 'xray-l3' in file:
            return XrayVisibity(header=header, control=control, data=data, energies=energies)
        elif 'spectrogram' in file:
            return XraySpectrogram(header=header, control=control, data=data, energies=energies)
        else:
            raise TypeError(f'File {file} does not contain pixel data')


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


class XrayVisibity(ScienceData):
    """
    Compressed visibilities selected pixels, detectors and energies.
    """
    @property
    def pixels(self):
        return np.vstack([self.data[f'pixel_mask{i}'][0] for i in range(1,6)])


class XraySpectrogram(ScienceData):
    """
    Spectrogram selected pixels, detectors and energies.
    """
    pass
