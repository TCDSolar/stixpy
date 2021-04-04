.. _stix:

Instrument
==========

Introduction
------------

STIX is an imaging X-ray spectrometer and essentially consists of 384 independent CdTe X-ray detectors which are arranged into 32 collimators each with a pixelised detector with 12 pixels.
Thirty of these are used for normal observations, the other two serve as a background monitor and a coarse flare locator.
Additionally STIX contains and aspect system that can determine the location of the Sun in the STIX FOV the time and spatial accuracy necessary for X-ray imaging.

At a high level the most granular X-ray data record is pixel data (PD), that is for each pixel of
each detector the X-ray counts for each time and energy bin are recorded. While the detectors
count individual photons (time and energy) these are summed into 32 pre-selected energy bins (4-150)
and rate dependant time intervals (1 - 20 s)

.. note::
    As SolarOrbiter is a deep space mission many of the design choices of the instrument have been
    made to reduce telemetry rates while still meeting the scientific goals of the mission.
    It is for this reason that the detected photons are binned in both time and energy during the
    processing chain.

Observations
------------
In normal operations STIX continuously records data into and large internal rotating memory store.
Quick look (QL) or summary data are frequently sent to ground based on these and other
observations science data requests are created and send to the instrument. The instrument processes
these request and essentially send down slices of the internal rotating memory store as science
data request.

Data Products
-------------

STIX produces data in four broad categories:

* House Keeping (HK)
* Quick Look (QL)
* Science (SCI)
* Diagnostics and other

House Keeping (HK)
~~~~~~~~~~~~~~~~~~

.. note::
   Not currently supported in stixpy

Quick Look (QL)
~~~~~~~~~~~~~~~

Quick look or summary data products are summed and reduced in cadence on board to

QLLightCurves
"""""""""""""
Counts from all pixels in the 30 detectors (excluding background and CFL) as summed together to for a
single data set which is further reduced by summing energy and time bins.

Background
""""""""""

Counts from all the pixels in the background detector are summed further reduced by summing energy
and time bins.

Variance
""""""""


Spectra
"""""""
Counts from each detector are summed and combined in time to produce per-detector energy spectra at
routine intervals.

EnergyCalibration
"""""""""""""""""
Energy calibration data obtained during quite (non-flaring) times of the on-board calibration X-ray
sources for each pixel of each detector.

Science (SCI)
~~~~~~~~~~~~~

The science data products are different combination of summing and compression of the basic
pixel date, each science request can selects subsets of of energies, pixels and detectors.


XrayLevel0
""""""""""
This is the least processed data and contains uncompressed counts for the selected energies, pixels
and detectors.

XrayLevel1
""""""""""
This is essentially the same as XrayLevel0 but the counts are compressed on-board before being sent to
ground.

XrayLevel2
""""""""""
For this product the counts from the 12 pixel are summed down to 4 before compression.

Visibility
""""""""""""""
This product further reduces the data by combine the the 4 summed pixel counts into a complex
visibility which is also compressed.

Spectrogram
"""""""""""""""
All spatial information is removed by summing all pixels for each detector, resulting in
per-detector spectrograms.
