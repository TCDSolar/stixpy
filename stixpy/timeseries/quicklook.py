from collections import OrderedDict

import astropy.units as u
import matplotlib.dates

from astropy.table import Table, QTable
from astropy.io import fits
from astropy.time.core import Time, TimeDelta
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt

from sunpy.visualization import peek_show
from sunpy.timeseries.timeseriesbase import GenericTimeSeries

__all__ = ['QLLightCurve', 'QLBackground', 'QLVariance']


class QLLightCurve(GenericTimeSeries):
    """
    Quicklook X-ray time series.

    Nominally in 5 energy bands from 4 - 150 kev
    """
    _source = 'stix'

    @peek_show
    def peek(self, **kwargs):
        self._validate_data_for_plotting()
        quantity_support()
        figure = plt.figure()
        axes = plt.gca()

        dates = matplotlib.dates.date2num(self.to_dataframe().index)

        labels = [f'{col} keV' for col in self.columns[4:]]

        [axes.plot_date(dates, self.to_dataframe().iloc[:, 4+i], '-', label=labels[i], **kwargs)
         for i in range(5)]

        axes.legend()

        axes.set_yscale("log")

        axes.set_title('STIX Quick Look ')
        axes.set_ylabel('count s$^{-1}$ keV$^{-1}$')

        axes.yaxis.grid(True, 'major')
        axes.xaxis.grid(False, 'major')
        axes.legend()

        # TODO: display better tick labels for date range (e.g. 06/01 - 06/05)
        formatter = matplotlib.dates.DateFormatter('%d %H:%M')
        axes.xaxis.set_major_formatter(formatter)

        axes.fmt_xdata = matplotlib.dates.DateFormatter('%d %H:%M')
        figure.autofmt_xdate()

        return figure

    @classmethod
    def _parse_file(cls, filepath):
        """
        Parses a GOES/XRS FITS file.

        Parameters
        ----------
        filepath : `str`
            The path to the file you want to parse.
        """
        hdus = fits.open(filepath)
        return cls._parse_hdus(hdus)

    @classmethod
    def _parse_hdus(cls, hdulist):
        """
        Parses STIX FITS data files to create TimeSeries.
        Parameters
        ----------
        hdulist :
        """
        header = hdulist[0].header
        control = QTable(hdulist[1].data)
        data = QTable(hdulist[2].data)
        energies = QTable(hdulist[3].data)
        energy_delta = energies['e_high'] - energies['e_low'] << u.keV
        data['counts'] = data['counts'] * u.ct
        data['timdel'] = data['timdel'] * u.s
        data['counts'] = data['counts'] / (data['timdel'].reshape(-1, 1) * energy_delta)

        names = [f'{energies["e_low"][i]}-{energies["e_high"][i]}' for i in range(5)]

        [data.add_column(data['counts'][:, i], name=names[i]) for i in range(5)]
        data.remove_column('counts')
        data['time'] = Time(header['date_obs']) + TimeDelta(data['time'] * u.s)

        # [f'{energies[i]["e_low"]} - {energies[i]["e_high"]} keV' for i in range(5)]

        units = OrderedDict([('control_index', None),
                             ('timedel', u.s),
                             ('triggers', None),
                             ('rcr', None),
                             (names[0], u.ct/(u.keV * u.s)),
                             (names[1], u.ct/(u.keV * u.s)),
                             (names[2], u.ct/(u.keV * u.s)),
                             (names[3], u.ct/(u.keV * u.s)),
                             (names[4], u.ct/(u.keV * u.s))])

        data_df = data.to_pandas()
        data_df.index = data_df['time']
        data_df.drop(columns='time', inplace=True)

        return data_df, header, units

    @classmethod
    def is_datasource_for(cls, **kwargs):
        """
        Determines if the file corresponds to a STIX QL LightCurve
        `~sunpy.timeseries.TimeSeries`.
        """
        # Check if source is explicitly assigned
        if 'source' in kwargs.keys():
            if kwargs.get('source', ''):
                return kwargs.get('source', '').lower().startswith(cls._source)
        # Check if HDU defines the source instrument
        if 'meta' in kwargs.keys():
            return (kwargs['meta'].get('telescop', '') == 'SOLO/STIX'
                    and 'ql-lightcurve' in kwargs['meta']['filename'])


class QLBackground(GenericTimeSeries):
    """
    Quicklook X-ray background detector time series.
    """

    @classmethod
    def _parse_file(cls, filepath):
        """
        Parses a GOES/XRS FITS file.

        Parameters
        ----------
        filepath : `str`
            The path to the file you want to parse.
        """
        hdus = fits.open(filepath)
        return cls._parse_hdus(hdus)

    @classmethod
    def _parse_hdus(cls, hdulist):
        """
        Parses STIX FITS data files to create TimeSeries.
        Parameters
        ----------
        filepath : `str` or `pathlib.Path`
            The path to the file you want to parse.
        """
        header = hdulist[0].header
        control = Table(hdulist[1].data)
        data = Table(hdulist[2].data)
        energies = Table(hdulist[3].data)

        [data.add_column(data['background'].data[:, i] * u.count, name=f'cts{i}') for i in range(5)]
        data.remove_column('background')
        data['time'] = Time(header['date_obs']) + TimeDelta(data['time'] * u.s)

        # [f'{energies[i]["e_low"]} - {energies[i]["e_high"]} keV' for i in range(5)]

        units = OrderedDict([('control_index', None),
                             ('timdel', u.s),
                             ('triggers', None),
                             ('rcr', None),
                             ('cts0', u.count),
                             ('cts1', u.count),
                             ('cts2', u.count),
                             ('cts3', u.count),
                             ('cts4', u.count)])

        data_df = data.to_pandas()
        data_df.index = data_df['time']
        data_df.drop(columns='time', inplace=True)

        return data_df, header, units

    @classmethod
    def is_datasource_for(cls, **kwargs):
        """
        Determines if the file corresponds to a STIX QL LightCurve
        `~sunpy.timeseries.TimeSeries`.
        """
        # Check if source is explicitly assigned
        if 'source' in kwargs.keys():
            if kwargs.get('source', ''):
                return kwargs.get('source', '').lower().startswith(cls._source)
        # Check if HDU defines the source instrument
        if 'meta' in kwargs.keys():
            return (kwargs['meta'].get('telescop', '') == 'SOLO/STIX'
                    and 'ql-background' in kwargs['meta']['filename'])


class QLVariance(GenericTimeSeries):
    """
    Quicklook X-ray background detector time series.
    """

    @classmethod
    def _parse_file(cls, filepath):
        """
        Parses a GOES/XRS FITS file.

        Parameters
        ----------
        filepath : `str`
            The path to the file you want to parse.
        """
        hdus = fits.open(filepath)
        return cls._parse_hdus(hdus)

    @classmethod
    def _parse_hdus(cls, hdulist):
        """
        Parses STIX FITS data files to create TimeSeries.
        Parameters
        ----------
        filepath : `str` or `pathlib.Path`
            The path to the file you want to parse.
        """
        header = hdulist[0].header
        control = Table(hdulist[1].data)
        data = Table(hdulist[2].data)
        energies = Table(hdulist[3].data)

        # [f'{energies[i]["e_low"]} - {energies[i]["e_high"]} keV' for i in range(5)]
        data['time'] = Time(header['date_obs']) + TimeDelta(data['time'] * u.s)
        units = OrderedDict([('control_index', None),
                             ('timdel', u.s),
                             ('variance', u.count**2)])

        data_df = data.to_pandas()
        data_df.index = data_df['time']
        data_df.drop(columns='time', inplace=True)

        return data_df, header, units

    @classmethod
    def is_datasource_for(cls, **kwargs):
        """
        Determines if the file corresponds to a STIX QL LightCurve
        `~sunpy.timeseries.TimeSeries`.
        """
        # Check if source is explicitly assigned
        if 'source' in kwargs.keys():
            if kwargs.get('source', ''):
                return kwargs.get('source', '').lower().startswith(cls._source)
        # Check if HDU defines the source instrument
        if 'meta' in kwargs.keys():
            return (kwargs['meta'].get('telescop', '') == 'SOLO/STIX'
                    and 'ql-variance' in kwargs['meta']['filename'])


# class QLVariance(GenericTimeSeries):
#     """
#     Quicklook X-ray background detector time series.
#     """
#
#     @classmethod
#     def _parse_file(cls, filepath):
#         """
#         Parses a GOES/XRS FITS file.
#
#         Parameters
#         ----------
#         filepath : `str`
#             The path to the file you want to parse.
#         """
#         hdus = fits.open(filepath)
#         return cls._parse_hdus(hdus)
#
#
#     @classmethod
#     def _parse_hdus(cls, hdulist):
#         """
#         Parses STIX FITS data files to create TimeSeries.
#         Parameters
#         ----------
#         filepath : `str` or `pathlib.Path`
#             The path to the file you want to parse.
#         """
#         header = hdulist[0].header
#         control = Table(hdulist[1].data)
#         data = Table(hdulist[2].data)
#         energies = Table(hdulist[3].data)
#
#         # [f'{energies[i]["e_low"]} - {energies[i]["e_high"]} keV' for i in range(5)]
#         data['time'] = Time(header['date_obs']) + TimeDelta(data['time'] * u.s)
#         units = OrderedDict([('control_index', None),
#                              ('timdel', u.s),
#                              ('variance', u.count**2)])
#
#         data_df = data.to_pandas()
#         data_df.index = data_df['time']
#         data_df.drop(columns='time', inplace=True)
#
#         return data_df, header, units
#
#     @classmethod
#     def is_datasource_for(cls, **kwargs):
#         """
#         Determines if the file corresponds to a STIX QL LightCurve
#         `~sunpy.timeseries.TimeSeries`.
#         """
#         # Check if source is explicitly assigned
#         if 'source' in kwargs.keys():
#             if kwargs.get('source', ''):
#                 return kwargs.get('source', '').lower().startswith(cls._source)
#         # Check if HDU defines the source instrument
#         if 'meta' in kwargs.keys():
#             return (kwargs['meta'].get('telescop', '') == 'SOLO/STIX'
#                     and 'ql-variance' in kwargs['meta']['filename'])
