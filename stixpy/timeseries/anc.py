import re
from collections import OrderedDict

import astropy.units as u
from astropy.io import fits
from astropy.io.fits import BinTableHDU, Header
from astropy.io.fits.connect import read_table_fits
from astropy.table import QTable
from astropy.time.core import Time, TimeDelta
from sunpy.timeseries.timeseriesbase import GenericTimeSeries

from stixpy.calibration.livetime import get_livetime_fraction

__all__ = ["ANCAspect", ]


def _hdu_to_qtable(hdupair):
    r"""
    Given a HDU pair, convert it to a QTable

    Parameters
    ----------
    hdupair
        HDU, header pair

    Returns
    -------
    QTable
    """
    header = hdupair.header
    # TODO remove when python 3.9 and astropy 5.3.4 are dropped weird non-ascii error
    header.pop("keycomments")
    header.pop("comment")
    bintable = BinTableHDU(hdupair.data, header=Header(cards=header))
    table = read_table_fits(bintable)
    qtable = QTable(table)
    return qtable


class ANCAspect(GenericTimeSeries):
    r"""
    Aspect solution time series

    tbd

    Examples
    --------
    >>> from stixpy.data import test
    >>> from sunpy.timeseries import TimeSeries
    >>> import stixpy.timeseries
    >>> asp = TimeSeries("https://pub099.cs.technik.fhnw.ch/data/fits/ANC/2022/03/14/ASP/solo_ANC_stix-asp-ephemeris_20220314_V02.fits") # doctest: +REMOTE_DATA
    >>> asp # doctest: +SKIP
    
    ANCAspect
        <sunpy.time.timerange.TimeRange object at 0x2b3d0a96890>
        Start: 2022-03-14 00:00:23
        End:   2022-03-14 23:59:18
        Center:2022-03-14 11:59:51
        Duration:0.999252662037037 days or
            23.982063888888888 hours or
            1438.9238333333333 minutes or
            86335.43 seconds
    <BLANKLINE>
    """

    _source = "stix"

    def plot(self, axes=None, columns=None, **plot_args):
        r"""
        Show a plot of the data.

        Parameters
        ----------
        axes : `~matplotlib.axes.Axes`, optional
            If provided the image will be plotted on the given axes.
            Defaults to `None`, so the current axes will be used.
        columns : `list`, optional
            Columns to plot, defaults to 'counts'.
        **plot_args : `dict`, optional
            Additional plot keyword arguments that are handed to
            :meth:`pandas.DataFrame.plot`.

        Returns
        -------
        axes : `~matplotlib.axes.Axes`
            The plot axes.
        """
        
        import matplotlib.pyplot as plt
        import numpy as np
        from sunpy.net import attrs as a
        
        if columns is None:
            columns = ['y_srf', 'z_srf']
            
            axes, columns = self._setup_axes_columns(axes, columns)  
            
            pam = self.data['sas_ok']
            
            sas_ok_values = self.data['sas_ok']
            
            has_false = (sas_ok_values == False).any()
            
            if has_false:
                
                # copy data to destroy reference and keep original data 
                y_srf_cp_t = self.to_dataframe()['y_srf'].copy().to_numpy()
                z_srf_cp_t = self.to_dataframe()['z_srf'].copy().to_numpy()
                # overwrite False data with nan such that there is a break in the plot when results are invalid
                y_srf_cp_t[self.data['sas_ok']==False] = np.nan
                z_srf_cp_t[self.data['sas_ok']==False] = np.nan

                # copy data to destroy reference and keep original data
                y_srf_cp_f = self.to_dataframe()['y_srf'].copy().to_numpy()
                z_srf_cp_f = self.to_dataframe()['z_srf'].copy().to_numpy()   
                # overwrite True data with nan such that there is a break in the plot when results are valid
                y_srf_cp_f[self.data['sas_ok']==True] = np.nan
                z_srf_cp_f[self.data['sas_ok']==True] = np.nan           
                
                # plot graphs where sas_ok is True with a solid line
                a = axes.plot(self._data.index, y_srf_cp_t, linestyle='solid', label=f'sun y = valid', **plot_args)
                b = axes.plot(self._data.index, z_srf_cp_t, linestyle='solid', label=f'sun z = valid', **plot_args)
            
                a = a[0].get_color()
                b = b[0].get_color()
                
                # plot graphs where sas_ok is False with a dashed line
                axes.plot(self._data.index, y_srf_cp_f, linestyle='dashed', color=a, label=f'sun y = invalid', **plot_args)
                axes.plot(self._data.index, z_srf_cp_f, linestyle='dashed', color=b, label=f'sun z = invalid', **plot_args)
                
            else: 
                axes = self._data[columns].plot(ax=axes, **plot_args)

        else:
            axes, columns = self._setup_axes_columns(axes, columns)
            axes = self.data[columns].plot(ax=axes, **plot_args)
        
        units = set([self.units[col] for col in columns])
        if len(units) == 1 and list(units)[0] is not None:
            # If units of all columns being plotted are the same, add a unit
            # label to the y-axis.
            unit = u.Unit(list(units)[0])
            axes.set_ylabel(unit.to_string())
            
        axes.set_title("STIX Aspect solutions")
        axes.legend()
        self._setup_x_axis(axes)
        plt.xlabel("UTC")
        return axes

    @classmethod
    def _parse_file(cls, filepath):
        r"""
        Parses a STIX ANC Aspect file.

        Parameters
        ----------
        filepath : `str`
            The path to the file you want to parse.
        """
        hdus = fits.open(filepath)
        return cls._parse_hdus(hdus)

    @classmethod
    def _parse_hdus(cls, hdulist):
        r"""
        Parses STIX FITS data files to create TimeSeries.

        Parameters
        ----------
        hdulist :
        """
        header = hdulist[0].header
        # control = _hdu_to_qtable(hdulist[1])
        data = _hdu_to_qtable(hdulist[2])

        try:
            data["time"] = Time(header["date_obs"]) + data["time"]
        except KeyError:
            data["time"] = Time(header["date-obs"]) + TimeDelta(data["time"])

        units = OrderedDict((c.info.name, c.unit) for c in data.itercols() if c.info.name != "time")

        names = [name for name in data.colnames if len(data[name].shape) <= 1]

        data_df = data[names].to_pandas()
        data_df.index = data_df["time"]
        data_df.drop(columns="time", inplace=True)

        return data_df, header, units

    @classmethod
    def is_datasource_for(cls, **kwargs):
        """
        Determines if the file corresponds to a STIX QL LightCurve
        `~sunpy.timeseries.TimeSeries`.
        """
        # Check if source is explicitly assigned
        if "source" in kwargs.keys():
            if kwargs.get("source", ""):
                return kwargs.get("source", "").lower().startswith(cls._source)
        # Check if HDU defines the source instrument
        if "meta" in kwargs.keys():
            return kwargs["meta"].get("telescop", "") == "SOLO/STIX" and "asp-ephemeris" in kwargs["meta"]["filename"]

    def __repr__(self):
        return f"{self.__class__.__name__}\n" f"    {self.time_range}"
 