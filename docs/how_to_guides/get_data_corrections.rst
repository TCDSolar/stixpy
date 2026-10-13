.. _get-data-corrections:

*************************************************
Select and correct counts with ``get_data``
*************************************************

`~stixpy.product.sources.science.ScienceData.get_data` selects counts from a STIX science
product (pixel data or a spectrogram) and applies the corrections needed before the counts
can be compared with other data or fitted. This guide shows the common options; the
docstring has the full details. The examples use the pixel data and background files from
the gallery examples:

.. code-block:: python

    >>> import astropy.units as u
    >>> from stixpy.product import Product
    >>> cpd_sci = Product(
    ...     "http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/"
    ...     "solo_L1_stix-sci-xray-cpd_20210923T152015-20210923T152639_V02_2109230030-62447.fits"
    ... )  # doctest: +REMOTE_DATA
    >>> cpd_bkg = Product(
    ...     "http://pub099.cs.technik.fhnw.ch/fits/L1/2021/09/23/SCI/"
    ...     "solo_L1_stix-sci-xray-cpd_20210923T095923-20210923T113523_V02_2109230083-57078.fits"
    ... )  # doctest: +REMOTE_DATA

The returned data
=================

``get_data`` returns a `~stixpy.product.sources.science.SelectedData` named tuple. Access
its fields by name, as more may be added later:

.. code-block:: python

    >>> data = cpd_sci.get_data()  # doctest: +REMOTE_DATA
    >>> data.counts.shape  # doctest: +REMOTE_DATA
    (698, 32, 1, 30)
    >>> data.counts.unit  # doctest: +REMOTE_DATA
    Unit("ct / (keV s)")

``counts`` and ``counts_err`` have the shape ``(time, detector, pixel, energy)``. The other
fields hold the time bin durations (``t_norm``), the energy bin widths (``e_norm``), the
livetime fraction and its uncertainty, the ELUT correction factor, the time bin centres,
the energy table and the attenuator (RCR) state of each time bin.

By default the counts are a differential count rate (``vtype="dcr"``, ct / (s keV)). Use
``vtype="cr"`` for a count rate or ``vtype="c"`` for counts.

Selecting times, energies, detectors and pixels
===============================================

Each of ``time_indices``, ``energy_indices``, ``detector_indices`` and ``pixel_indices``
takes flat indices, which keep those bins, or ``[start, end]`` pairs, which sum each range.
Times can also be given as strings or `~astropy.time.Time`, and energies as a
`~astropy.units.Quantity` in keV. ``detector_indices`` also accepts ``"top24"``, the 24
imaging detectors used for spectroscopy, and ``"bkg"``, the background detector.

.. code-block:: python

    >>> data = cpd_sci.get_data(
    ...     time_indices=["2021-09-23T15:20:30", "2021-09-23T15:23:30"],
    ...     energy_indices=[[4, 10], [15, 28]] * u.keV,
    ...     detector_indices="top24",
    ...     vtype="cr",
    ... )  # doctest: +REMOTE_DATA
    >>> data.counts.shape  # doctest: +REMOTE_DATA
    (1, 24, 1, 2)

Requested times must lie inside the file; times or indices outside it raise a
`ValueError`.

Livetime and ELUT corrections
=============================

Two corrections are applied by default:

* ``livetime_correction=True`` corrects for the time the detectors were busy and could not
  record counts. As in STIX-GSW, the pixels are summed, so the pixel axis has length 1, and
  the livetime uncertainty is added to ``counts_err``.
* ``elut_correction=True`` corrects each energy bin for the difference between its nominal
  science width and its calibrated width from the energy look-up table (ELUT).

Turn both off and use ``vtype="c"`` to get the raw counts:

.. code-block:: python

    >>> raw = cpd_sci.get_data(vtype="c", livetime_correction=False, elut_correction=False)  # doctest: +REMOTE_DATA

Background subtraction
======================

Pass a background product as ``bkg`` to subtract it. It must be pixel data with a single
time bin, such as a STIX background (BKG) file, and cover every energy bin, detector and
pixel used. The background is converted to a count rate, scaled to each science time bin
and subtracted, and its uncertainty is added to ``counts_err``. Background subtraction needs
the livetime correction.

.. code-block:: python

    >>> data = cpd_sci.get_data(
    ...     time_indices=["2021-09-23T15:20:30", "2021-09-23T15:23:30"],
    ...     detector_indices="top24",
    ...     bkg=cpd_bkg,
    ... )  # doctest: +REMOTE_DATA

Spectra for fitting with sunkit-spex
====================================

With ``sunkit_spex_spectrum=True``, ``get_data`` returns `sunkit_spex` spectra, with the
spectral response matrix (SRM) for the flare location and attenuator state in their
metadata, instead of a `~stixpy.product.sources.science.SelectedData`. This needs the
optional ``spectroscopy`` dependencies:

.. code-block:: bash

    $ pip install 'stixpy[spectroscopy]'

The SRM itself can also be made directly with `~stixpy.calibration.srm.get_srm`. See the
:ref:`sphx_glr_generated_gallery_spectroscopy_demo.py` example for a complete walk through.
