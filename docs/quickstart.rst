Quickstart
==========

This guide will show how to search, download and plot basic STIX observations for more detailed
information on the instrument and observations please see xxx

Data Search and Download
------------------------

Data search download is built on Sunpy's Fido and provides easy search and download of data. To
search for some quicklook X-ray light curve data run the following commands

.. code-block:: python

   from sunpy.net import Fido, attrs as a
   from stixpy.net.client import STIXClient

   query = Fido.search(a.Time('2020-06-05', '2020-06-07'), a.Instrument.stix, a.stix.DataProduct.ql_lightcurve)
   print(query)


Downloading the data is then a simple as

.. code-block:: python

   files = Fido.fetch(query)
   print(files)

Plotting data
-------------

One we have the downloaded we can create timeseries from the fits files, following on from the above
code.

.. code-block:: python

   from sunpy.timeseries import TimeSeries
   from stiy import timeseries

   ql_lightcurves = TimeSeries(files)

   ql_lightcurves[0].peek()

The three seperate time siers can esasily be concatentated and plot the entire time range

.. code-block:: python

   combined_ts = ql_lightcurves[0]
   for lc in ql_lightcurves[1:]:
       combined_ts = combind_ts.concatenate(lc)

   combined_ts.peek()
