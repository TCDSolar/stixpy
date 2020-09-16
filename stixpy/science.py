import astropy.units as u
import numpy as np

from astropy.io import fits
from astropy.table.table import QTable
from astropy.time import Time, TimeDelta
from sunpy.time.timerange import TimeRange


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

    @property
    def time_range(self):
        times = Time(self.header['date_obs']) + TimeDelta(self.data['time'])
        return TimeRange(times.min(), times.max())

    @property
    def pixels(self):
        return self.data['pixel_masks'][0]

    @property
    def detectors(self):
        return self.data['detector_masks'][0]

    @property
    def energies(self):
        return self.energies_

    def __repr__(self):

        energy = ','.join([f'{el.value}-{eh.value}' for el, eh in self.energies['e_low', 'e_high']])
        return f'{self.__class__.__name__}' \
               f'{self.time_range}' \
               f'    Detectors: [{",".join(np.where(self.detectors, np.arange(32), np.full(32, "_")))}]\n' \
               f'    Pixels:    [{",".join(np.where(self.pixels, np.arange(12), np.full(12, "_")))}]\n' \
               f'    Energies:  [{",".join([str(ch) for ch in self.energies["channel"]])}]'

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
    Compreesed visibities selected pixels, detectors and energies.
    """
    pass

class XraySpectrogram(ScienceData):
    """
    Spectrogram selected pixels, detectors and energies.
    """
    pass