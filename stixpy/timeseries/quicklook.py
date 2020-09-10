from collections import OrderedDict

import astropy.units as u

from astropy.table import Table
from astropy.io import fits
from astropy.time.core import Time, TimeDelta

from sunpy.timeseries.timeseriesbase import GenericTimeSeries


class QLLightCurve(GenericTimeSeries):
    """
    Quicklook X-ray time series.

    Nominally in 5 energy bands from
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

        [data.add_column(data['counts'].data[:, i] * u.count, name=f'cts{i}') for i in range(5)]
        data.remove_column('counts')
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