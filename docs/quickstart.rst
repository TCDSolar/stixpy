Quickstart
==========

This guide will show how to search, download and plot basic STIX observations for more detailed
information on the instrument and observations please see :ref:`stix`.

Installation
------------

.. code-block::

   pip install git+https://github.com/samaloney/stixpy


Data Search and Download
------------------------

Data search download is built on Sunpy's Fido and provides easy search and download of data.
To search for some quicklook X-ray light curve data run the following commands

.. code-block:: python

   from sunpy.net import Fido, attrs as a
   from stixpy.net.client import STIXClient

   query = Fido.search(a.Time('2020-06-05', '2020-06-07'), a.Instrument.stix,
                       a.stix.DataProduct.ql_lightcurve)

Downloading the data is then a simple as

.. code-block:: python

   files = Fido.fetch(query)



Plotting data
-------------

One we have the downloaded we can create time series from the fits files, following on from the above code.

.. code-block:: python

   from sunpy.timeseries import TimeSeries
   from stixpy import timeseries

   ql_lightcurves = TimeSeries(files)

   ql_lightcurves[0].peek()


.. plot::

   from sunpy.net import Fido, attrs as a
   from sunpy.timeseries import TimeSeries
   from stixpy.net.client import STIXClient
   from stixpy.timeseries import quicklook

   query = Fido.search(a.Time('2020-06-05', '2020-06-07'), a.Instrument.stix,
                       a.stix.DataProduct.ql_lightcurve)
   files = Fido.fetch(query)
   ql_lightcurves = TimeSeries(files)
   ql_lightcurves[0].peek()


The three separate time series can easily be concatenated to plot the entire time range

.. code-block:: python

   combined_ts = ql_lightcurves[0]
   for lc in ql_lightcurves[1:]:
       combined_ts = combined_ts.concatenate(lc)

   combined_ts.peek()

.. plot::

   from sunpy.net import Fido, attrs as a
   from sunpy.timeseries import TimeSeries
   from stixpy.net.client import STIXClient
   from stixpy.timeseries import quicklook

   query = Fido.search(a.Time('2020-06-05', '2020-06-07'), a.Instrument.stix,
                       a.stix.DataProduct.ql_lightcurve)
   files = Fido.fetch(query)
   ql_lightcurves = TimeSeries(files)

   combined_ts = ql_lightcurves[0]
   for lc in ql_lightcurves[1:]:
       combined_ts = combined_ts.concatenate(lc)

   combined_ts.peek()
