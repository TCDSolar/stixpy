import re
from collections import OrderedDict

import astropy.units as u
from astropy.io import fits
from astropy.io.fits import BinTableHDU, Header
from astropy.io.fits.connect import read_table_fits
from astropy.table import QTable, Table
from astropy.time.core import Time, TimeDelta
from sunpy.timeseries.timeseriesbase import GenericTimeSeries

from stixpy.calibration.livetime import get_livetime_fraction

__all__ = ["QLLightCurve", "QLBackground", "QLVariance", "HKMaxi"]


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
    header.pop('keycomments')
    header.pop('comment')
    bintable = BinTableHDU(hdupair.data, header=Header(cards=header))
    table = read_table_fits(bintable)
    qtable = QTable(table)
    return qtable


class QLLightCurve(GenericTimeSeries):
    r"""
    Quicklook X-ray time series.

    Nominally in five energy bands from 4-150 keV which are obtained by summing counts for all masked detectors and
    pixels into five predefined energy ranges. They are double buffered with default integration time of 4s and depth
    of 32 bits. Maximum rate is approximately 1MHz with one summed live time counter for the appropriate detectors.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from sunpy.timeseries import TimeSeries
    >>> from stixpy.timeseries.quicklook import QLLightCurve
    >>> ql_lc = TimeSeries("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/solo_L1_stix-ql-lightcurve_20200506_V02.fits") # doctest: +REMOTE_DATA
    >>> ql_lc # doctest: +SKIP
    QLLightCurve
           <sunpy.time.timerange.TimeRange object at ...>
        Start: 2020-05-06 00:00:01
        End:   2020-05-06 23:59:57
        Center:2020-05-06 11:59:59
        Duration:0.9999538194444444 days or
               23.998891666666665 hours or
               1439.9334999999999 minutes or
               86396.01 seconds
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
        if columns is None:
            count_re = re.compile(r"\d+-\d+ keV$")
            columns = [column for column in self.columns if count_re.match(column)]

        axes, columns = self._setup_axes_columns(axes, columns)

        axes = self._data[columns].plot(ax=axes, **plot_args)

        units = set([self.units[col] for col in columns])
        if len(units) == 1 and list(units)[0] is not None:
            # If units of all columns being plotted are the same, add a unit
            # label to the y-axis.
            unit = u.Unit(list(units)[0])
            axes.set_ylabel(unit.to_string())
            if unit == u.ct/(u.s * u.keV):
                axes.set_yscale("log")

        axes.set_title("STIX QL Light Curve")
        axes.legend()
        self._setup_x_axis(axes)
        return axes

    @classmethod
    def _parse_file(cls, filepath):
        r"""
        Parses a STIX file.

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
        energies = _hdu_to_qtable(hdulist[4])
        energy_delta = energies["e_high"] - energies["e_low"]

        live_frac, *_ = get_livetime_fraction(data["triggers"].reshape(-1) / (16 * data["timedel"]))

        data["counts"] = data["counts"] / ((data['timedel'].to(u.s) * live_frac).reshape(-1, 1) * energy_delta)

        names = [
            f"{energies['e_low'][i].value.astype(int)}-{energies['e_high'][i].value.astype(int)} {energies['e_high'].unit}" for i in range(5)]

        [data.add_column(data["counts"][:, i], name=names[i]) for i in range(5)]
        data.remove_column("counts")
        [data.add_column(data["counts_comp_err"][:, i], name=f"{names[i]}_comp_err") for i in range(5)]
        data.remove_column("counts_comp_err")

        try:
            data["time"] = Time(header["date_obs"]) + data["time"]
        except KeyError:
            data["time"] = Time(header["date-obs"]) + TimeDelta(data["time"])

        units = OrderedDict((c.info.name, c.unit) for c in data.itercols() if c.info.name != 'time')
        units.update([(f"{name}_comp_err", units[name]) for name in names])

        data["triggers"] = data["triggers"].reshape(-1)
        data["triggers_comp_err"] = data["triggers_comp_err"].reshape(-1)

        data_df = data.to_pandas()
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
            return kwargs["meta"].get("telescop", "") == "SOLO/STIX" and "ql-lightcurve" in kwargs["meta"]["filename"]

    def __repr__(self):
        return f"{self.__class__.__name__}\n" f"    {self.time_range}"


class QLBackground(GenericTimeSeries):
    r"""
    Quicklook X-ray time series from the background detector.

    Background monitoring is done in such way that counts from background detector are summed in five specified energy
    ranges. These QL data are double buffered into accumulators of 32bit depth. Maximum rate is approximately 30kHz and
    one live time counter is available. Integration time is parameter with default value of 32s

    Examples
    --------
    >>> from stixpy.data import test
    >>> from sunpy.timeseries import TimeSeries
    >>> from stixpy.timeseries.quicklook import QLLightCurve
    >>> ql_bg = TimeSeries("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/"
    ...                    "solo_L1_stix-ql-background_20200506_V02.fits") # doctest: +REMOTE_DATA
    >>> ql_bg # doctest: +SKIP
    <stixpy.timeseries.quicklook.QLBackground object at ...
    SunPy TimeSeries
    ----------------
     Observatory:		 SOLO/STIX
     Instrument:		 STIX
     Channel(s):		 control_index<br>timedel<br>triggers<br>triggers_err<br>4.0-10.0<br>10.0-15.0<br>15.0-25.0<br>25.0-50.0<br>50.0-150.0<br>4.0-10.0_err<br>10.0-15.0_err<br>15.0-25.0_err<br>25.0-50.0_err<br>50.0-150.0_err
     Start Date:		 2020-05-06 00:00:04
     End Date:		 2020-05-06 23:59:56
     Center Date:		 2020-05-06 11:59:59
     Resolution:		 8.03 s
     Samples per Channel:		 10758
     Data Range(s):		 control_index    1.60E+01<br>timedel          0.00E+00<br>triggers         1.10E+03<br>triggers_err     3.23E+01<br>4.0-10.0         7.72E+01<br>10.0-15.0        1.34E+02<br>15.0-25.0        6.07E+01<br>25.0-50.0        2.17E+01<br>50.0-150.0       4.95E+00<br>4.0-10.0_err     9.25E+00<br>10.0-15.0_err    1.85E+01<br>15.0-25.0_err    1.85E+01<br>25.0-50.0_err    1.85E+01<br>50.0-150.0_err   9.25E+00
     Units:		 None<br>ct<br>s
                              control_index  timedel  ...  25.0-50.0_err  50.0-150.0_err
     time                                             ...
     2020-05-06 00:00:03.531              3      800  ...       2.345208        0.707107
     2020-05-06 00:00:11.531              3      800  ...       2.345208        0.707107
     2020-05-06 00:00:19.531              3      800  ...       2.345208        0.000000
     2020-05-06 00:00:27.531              3      800  ...       2.345208        0.000000
     2020-05-06 00:00:35.531              3      800  ...       2.345208        0.000000
     ...                                ...      ...  ...            ...             ...
     2020-05-06 23:59:23.531              2      800  ...       0.707107        1.224745
     2020-05-06 23:59:31.531              2      800  ...       0.000000        2.345208
     2020-05-06 23:59:39.531              2      800  ...       0.000000        4.636809
     2020-05-06 23:59:47.531              2      800  ...       0.000000        1.224745
     2020-05-06 23:59:55.531              2      800  ...       0.707107        2.345208
     <BLANKLINE>
     [10758 rows x 14 columns]

    """

    def plot(self, axes=None, columns=None, **plot_args):
        r"""
        Show a plot of the data.

        Parameters
        ----------
        axes : `~matplotlib.axes.Axes`, optional
            If provided the image will be plotted on the given axes.
            Defaults to `None`, so the current axes will be used.
        columns : `str`, optional
            Columns to plot, defaults to 'counts'.
        **plot_args : `dict`, optional
            Additional plot keyword arguments that are handed to
            :meth:`pandas.DataFrame.plot`.

        Returns
        -------
        axes : `~matplotlib.axes.Axes`
            The plot axes.
        """
        if columns is None:
            count_re = re.compile(r"\d+-\d+ keV$")
            columns = [column for column in self.columns if count_re.match(column)]

        axes, columns = self._setup_axes_columns(axes, columns)

        axes = self._data[columns].plot(ax=axes, **plot_args)

        units = set([self.units[col] for col in columns])
        if len(units) == 1 and list(units)[0] is not None:
            # If units of all columns being plotted are the same, add a unit
            # label to the y-axis.
            unit = u.Unit(list(units)[0])
            axes.set_ylabel(unit.to_string())
            axes.set_yscale("log")

        axes.set_title("STIX QL Background")
        axes.legend()
        self._setup_x_axis(axes)
        return axes

    @classmethod
    def _parse_file(cls, filepath):
        """
        Parses a STIX FITS file.

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
            The path to the file you want to parse.
        """
        header = hdulist[0].header
        # control = _hdu_to_qtable(hdulist[1])
        data = _hdu_to_qtable(hdulist[2])
        energies = _hdu_to_qtable(hdulist[4])
        energy_delta = energies["e_high"] - energies["e_low"]

        live_frac, *_ = get_livetime_fraction(data["triggers"].reshape(-1) / data["timedel"])

        data["counts"] = data["counts"] / ((data['timedel'].to(u.s) * live_frac).reshape(-1, 1) * energy_delta)

        names = [f'{energies["e_low"][i].value.astype(int)}-{energies["e_high"][i].value.astype(int)} {energies["e_high"].unit}' for i in range(5)]

        [data.add_column(data["counts"][:, i], name=names[i]) for i in range(5)]
        data.remove_column("counts")

        [data.add_column(data["counts_comp_err"][:, i], name=f"{names[i]}_comp_err") for i in range(5)]
        data.remove_column("counts_comp_err")

        try:
            data["time"] = Time(header["date_obs"]) + TimeDelta(data["time"])
        except KeyError:
            data["time"] = Time(header["date-obs"]) + TimeDelta(data["time"])

        units = OrderedDict((c.info.name, c.unit) for c in data.itercols() if c.info.name != 'time')
        units.update([(f"{name}_comp_err", units[name]) for name in names])

        data["triggers"] = data["triggers"].reshape(-1)
        data["triggers_comp_err"] = data["triggers_comp_err"].reshape(-1)

        data_df = data.to_pandas()
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
            return kwargs["meta"].get("telescop", "") == "SOLO/STIX" and "ql-background" in kwargs["meta"]["filename"]


class QLVariance(GenericTimeSeries):
    """
    Quicklook variance time series

    Variance of counts is calculated for one energy range over counts summed from selected detectors and pixels in 40
    accumulators, each accumulating for 0.1s. These accumulators are double buffered with depth of 32bits and
    approximate data rate of 1 Mhz.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from sunpy.timeseries import TimeSeries
    >>> from stixpy.timeseries.quicklook import QLLightCurve
    >>> ql_var = TimeSeries("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/QL/"
    ...                     "solo_L1_stix-ql-variance_20200506_V02.fits") # doctest: +REMOTE_DATA
    >>> ql_var # doctest: +SKIP
    <stixpy.timeseries.quicklook.QLVariance object at ...>
    SunPy TimeSeries
    ----------------
    Observatory:		 SOLO/STIX
    Instrument:		     STIX
    Channel(s):		     timedel<br>control_index<br>4.0-20.0<br>4.0-20.0_err
    Start Date:		     2020-05-06 00:00:02
    End Date:		  2020-05-06 23:59:58
    Center Date:		 2020-05-06 11:59:59
    Resolution:		 4.015 s
    Samples per Channel:		 21516
    Data Range(s):		 timedel         0.00E+00<br>control_index   6.00E+00<br>4.0-20.0        4.93E+00<br>4.0-20.0_err    5.87E+02
    Units:		 ct<br>s<br>None
                             timedel  control_index  4.0-20.0  4.0-20.0_err
    time
    2020-05-06 00:00:01.531      400              2  0.419844     73.901962
    2020-05-06 00:00:05.531      400              2  0.169844     36.952671
    2020-05-06 00:00:09.531      400              2  0.154844     18.479719
    2020-05-06 00:00:13.531      400              2  1.359844    295.603607
    2020-05-06 00:00:17.531      400              2  0.114844     18.479719
    ...                          ...            ...       ...           ...
    2020-05-06 23:59:41.531      400              1  0.919844    147.802231
    2020-05-06 23:59:45.531      400              1  0.919844    147.802231
    2020-05-06 23:59:49.531      400              1  2.159844    295.603607
    2020-05-06 23:59:53.531      400              1  0.094844     18.479719
    2020-05-06 23:59:57.531      400              1  0.154844     18.479719
    <BLANKLINE>
    [21516 rows x 4 columns]
    """

    def plot(self, axes=None, columns=None, **plot_args):
        """
        Show a plot of the data.

        Parameters
        ----------

        axes : `~matplotlib.axes.Axes`, optional
            If provided the image will be plotted on the given axes.
            Defaults to `None`, so the current axes will be used.
        columns :
            Columns to plot, defaults to 'variance'.
        **plot_args : `dict`, optional
            Additional plot keyword arguments that are handed to
            :meth:`pandas.DataFrame.plot`.

        Returns
        -------
        axes : `~matplotlib.axes.Axes`
            The plot axes.
        """
        if columns is None:
            count_re = re.compile(r"\d+-\d+ keV$")
            columns = [column for column in self.columns if count_re.match(column)]

        axes, columns = self._setup_axes_columns(axes, columns)
        axes = self._data[columns].plot(ax=axes, **plot_args)

        units = set([self.units[col] for col in columns])
        if len(units) == 1 and list(units)[0] is not None:
            # If units of all columns being plotted are the same, add a unit
            # label to the y-axis.
            unit = u.Unit(list(units)[0])
            axes.set_ylabel(unit.to_string())
            axes.set_yscale("log")

        axes.set_title("STIX QL Variance")
        axes.legend()
        self._setup_x_axis(axes)
        return axes

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
            The path to the file you want to parse.
        """
        header = hdulist[0].header
        control = _hdu_to_qtable(hdulist[1])
        data = _hdu_to_qtable(hdulist[2])
        energies = _hdu_to_qtable(hdulist[4])
        delta_energy = (
            energies[control["energy_bin_mask"][0]]["e_high"][-1] - energies[control["energy_bin_mask"][0]]["e_low"][0]
        )
        name = (
            f'{energies[control["energy_bin_mask"][0]]["e_low"][0].value.astype(int)}'
            f'-{energies[control["energy_bin_mask"][0]]["e_high"][-1].value.astype(int)} {energies["e_high"].unit}'
        )

        try:
            data["time"] = Time(header["date_obs"]) + TimeDelta(data["time"])
        except KeyError:
            data["time"] = Time(header["date-obs"]) + TimeDelta(data["time"])

        data["variance"] = data["variance"].reshape(-1) * u.ct / (delta_energy * data["timedel"].to(u.s))

        data.add_column(data["variance"], name=name)
        data.remove_column("variance")
        data.add_column(data["variance_comp_err"], name=f"{name}_comp_err")
        data.remove_column("variance_comp_err")

        units = OrderedDict((c.info.name, c.unit) for c in data.itercols() if c.info.name != 'time')
        units[f"{name}_comp_err"]  = u.ct/(u.s * u.keV)

        data[name] = data[name].reshape(-1)
        data[f"{name}_comp_err"] = data[f"{name}_comp_err"].reshape(-1)

        data_df = data.to_pandas()
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
            return kwargs["meta"].get("telescop", "") == "SOLO/STIX" and "ql-variance" in kwargs["meta"]["filename"]


class HKMaxi(GenericTimeSeries):
    """
    Housekeeping Maxi Report.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from sunpy.timeseries import TimeSeries
    >>> from stixpy.timeseries.quicklook import HKMaxi
    >>> hk = TimeSeries("https://pub099.cs.technik.fhnw.ch/fits/L1/2020/05/06/HK/"
    ...                 "solo_L1_stix-hk-maxi_20200506_V02U.fits") # doctest: +REMOTE_DATA
    >>> hk  # doctest: +SKIP
    <stixpy.timeseries.quicklook.HKMaxi ...>
    """

    # def plot(self, axes=None, **plot_args):
    #     """
    #     Show a plot of the data.
    #
    #     Parameters
    #     ----------
    #     axes : `~matplotlib.axes.Axes`, optional
    #         If provided the image will be plotted on the given axes.
    #         Defaults to `None`, so the current axes will be used.
    #     **plot_args : `dict`, optional
    #         Additional plot keyword arguments that are handed to
    #         :meth:`pandas.DataFrame.plot`.
    #
    #     Returns
    #     -------
    #     axes : `~matplotlib.axes.Axes`
    #         The plot axes.
    #     """
    #     import matplotlib.pyplot as plt
    #     # Get current axes
    #     if axes is None:
    #         fig, axes = plt.subplots()
    #
    #     self._validate_data_for_plotting()
    #     quantity_support()
    #
    #     dates = matplotlib.dates.date2num(self.to_dataframe().index)
    #
    #     label = f'{self.columns[2]} keV'
    #
    #     axes.plot_date(dates, self.to_dataframe().iloc[:, 2], '-', label=label) #, **plot_args)
    #
    #     axes.legend(loc='upper right')
    #
    #     axes.set_yscale("log")
    #
    #     axes.set_title('STIX Quick Look Variance')
    #     axes.set_ylabel('count s$^{-1}$ keV$^{-1}$')
    #
    #     axes.yaxis.grid(True, 'major')
    #     axes.xaxis.grid(False, 'major')
    #     axes.legend()
    #
    #     # TODO: display better tick labels for date range (e.g. 06/01 - 06/05)
    #     formatter = matplotlib.dates.DateFormatter('%d %H:%M')
    #     axes.xaxis.set_major_formatter(formatter)
    #
    #     axes.fmt_xdata = matplotlib.dates.DateFormatter('%d %H:%M')
    #     fig.autofmt_xdate()
    #
    #     return fig

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
            The path to the file you want to parse.
        """
        header = hdulist[0].header
        control = Table(hdulist[1].data)
        header["control"] = control
        data = Table(hdulist[2].data)

        try:
            data["time"] = Time(header["date_obs"]) + TimeDelta(data["time"] * u.cs)
        except KeyError:
            data["time"] = Time(header["date-obs"]) + TimeDelta(data["time"] * u.cs)

        data_df = data.to_pandas()
        data_df.index = data_df["time"]
        data_df.drop(columns="time", inplace=True)

        return data_df, header, None

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
            return kwargs["meta"].get("telescop", "") == "SOLO/STIX" and "hk-maxi" in kwargs["meta"]["filename"]
