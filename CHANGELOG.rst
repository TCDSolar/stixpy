0.3.0 (2026-06-05)
==================

Breaking Changes
----------------

- The `~stixpy.product.sources.science.ScienceData` method :meth:`~stixpy.product.sources.science.ScienceData.get_data` is now keyword only. (`#196 <https://github.com/TCDSolar/stixpy/pull/196>`__)
- Increased minimum versions for these dependencies:

  - Python >= 3.12
  - astropy >= 6.1.0
  - matplotlib >= 3.8.0
  - sunpy >= 7.0
  - xrayvisim >= 0.2.1 (`#212 <https://github.com/TCDSolar/stixpy/pull/212>`__)


Removals
--------

- Remove deprecated ``stixpy.science`` module use `stixpy.product.Product` instead. (`#196 <https://github.com/TCDSolar/stixpy/pull/196>`__)


New Features
------------

- Add ability to control the data normalisation to `~stixpy.product.sources.science.ScienceData.get_data` via the `vtype` keyword:

  - 'c' - count [c]
  - 'cr' - count rate [c/s]
  - 'dcr' - differential count rate [c/(s keV)] (`#196 <https://github.com/TCDSolar/stixpy/pull/196>`__)


Bug Fixes
---------

- Fixed version attributes not being exported in stixpy.net.attrs module. All version attributes (Version, VersionU, MinVersion, MinVersionU, MaxVersion, MaxVersionU) are now properly included in the module's ``__all__`` list and can be imported directly from ``stixpy.net.attrs``. (`#209 <https://github.com/TCDSolar/stixpy/pull/209>`__)
- Fix :ref:`sphx_glr_generated_gallery_imaging_demo.py` to correctly use center of STIX field of view for initial grid correction and flare position estimation. (`#216 <https://github.com/TCDSolar/stixpy/pull/216>`__)


Internal Changes
----------------

- Update `~stixpy.net.client.STIXClient` Fido client to use format keyword and ``parse`` syntax, internal changes only. (`#177 <https://github.com/TCDSolar/stixpy/pull/177>`__)
- Update package with sunpy template and enable updates. (`#219 <https://github.com/TCDSolar/stixpy/pull/219>`__)


Stixpy 0.2.1 (2026-03-10)
=========================

Bug Fixes
---------

- Update sunpy dependency to <7.0 as code not updated for new Fido client API. (`#211 <https://github.com/TCDSolar/stixpy/pull/211>`__)


Stixpy 0.2.0 (2026-01-18)
=========================

Backwards Incompatible Changes
------------------------------

- Update minimum dependencies of Python 3.11 and sunpy to 6.0.0 inline with `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_. (`#174 <https://github.com/TCDSolar/stixpy/pull/174>`__)


Features
--------

- Update `Fido` client to optionally take root path or url. Also add attributes to enable searching for specific versions for example latest version available. (`#143 <https://github.com/TCDSolar/stixpy/pull/143>`__)
- Enable `stixpy.calibration.visibility.create_meta_pixels` to include only top or bottom row of big pixels. (`#148 <https://github.com/TCDSolar/stixpy/pull/148>`__)
- Update `Fido` client for change in aspect/ephemeris file name and path. (`#153 <https://github.com/TCDSolar/stixpy/pull/153>`__)
- Add the pixel area normalisation from `~stixpy.calibration.visibility.create_meta_pixels` to dictionary returned. (`#162 <https://github.com/TCDSolar/stixpy/pull/162>`__)
- Add new timeseries source for STIX ANC aspect data `~stixpy.timeseries.anc.ANCAspect`. (`#169 <https://github.com/TCDSolar/stixpy/pull/169>`__)
- Add a instrument configuration constant `~stixpy.config.instrument.STIX_INSTRUMENT` to bundle instrument constants like grid configurations, pixel sizes .... (`#171 <https://github.com/TCDSolar/stixpy/pull/171>`__)
- Update imaging demo to use Sun center as location for making initial full disk back projection to determine flare location. (`#173 <https://github.com/TCDSolar/stixpy/pull/173>`__)
- Add enhanced pixel plot ~`stixpy.product.sources.science.PixelPlotMixin` improved layout now shows detector quadrant and labels. Pixel plots now support three options a normal bar plot an error bar plot and configuration plot. (`#178 <https://github.com/TCDSolar/stixpy/pull/178>`__)
- Created new `stixpy.visualisation.plotters.PixelPlotter` class to create pixel plots by moving and refactored code from current mixin so current API is maintained. This new approach simplifies the `~stixpy.product.sources.science` module and programmatic access to the plot controls enabling animations. (`#181 <https://github.com/TCDSolar/stixpy/pull/181>`__)


Bug Fixes
---------

- Fixed a bug in stixpy.calibration.visibility.create_meta_pixels to correctly determine which time bins to include in the sum based on their overlap with the specified time range. (`#138 <https://github.com/TCDSolar/stixpy/pull/138>`__)
- Fix bug in orientation of grids for internal shadowing. (`#146 <https://github.com/TCDSolar/stixpy/pull/146>`__)
- Fix bug where pointing information could be improperly interpolated in `~stixpy.coordinates.transforms.get_hpc_info`. (`#152 <https://github.com/TCDSolar/stixpy/pull/152>`__)
- Fix typo in `~stixpy.calibration.visibility.create_meta_pixels` which cased error if grid shadow correction was enabled. Also update grid calculations to use `Skycoord` objects. (`#156 <https://github.com/TCDSolar/stixpy/pull/156>`__)
- Fix error in `~stixpy.net.client.StixQueryResponse.filter_for_latest_version` if the result table was empty. (`#160 <https://github.com/TCDSolar/stixpy/pull/160>`__)
- Fix bug in STIX Fido client which caused data that spanned midnight to be missed. (`#176 <https://github.com/TCDSolar/stixpy/pull/176>`__)
- Fix apparent bug in `~stixpy.calibration.visibility.create_meta_pixels` by improving naming and docstrings. (`#194 <https://github.com/TCDSolar/stixpy/pull/194>`__)
- Fix incomplete conversion of `~stixpy.calibration.visibility.calibrate_visibility` to use `~astropy.coordinates.SkyCoord`. (`#194 <https://github.com/TCDSolar/stixpy/pull/194>`__)
- Fix ELUT reader `~stixpy.io.readers.read_elut` for column name change in upstream files. (`#194 <https://github.com/TCDSolar/stixpy/pull/194>`__)
- Restore mistakenly dropped updateds to the energies used in `~stixpy.calibration.transmission.generate_transmission_tables`.: (`#197 <https://github.com/TCDSolar/stixpy/pull/197>`__)
- Fix bug in `~stixpy.product.sources.science.ScienceData.get_data` when specifying `pixel_indices` with data containing less than 12 pixels. (`#200 <https://github.com/TCDSolar/stixpy/pull/200>`__)


Improved Documentation
----------------------

- Convert reconstructions to proper sunpy maps in ref`:ref:`sphx_glr_generated_gallery_imaging_demo.py`. (`#141 <https://github.com/TCDSolar/stixpy/pull/141>`__)
- Added instructions for initializing and updating git submodules and fixed broken links. (`#157 <https://github.com/TCDSolar/stixpy/pull/157>`__)


Trivial/Internal Changes
------------------------

- Update project configuration move to `project.toml` and individual config files for each tool (isort, ruff, pytest, etc) and add zenodo config file and mailmap. (`#158 <https://github.com/TCDSolar/stixpy/pull/158>`__)


Stixpy 0.1.2 (2024-07-16)
==================================================

Features
--------

- Add a custom coordinate frame :class:`stixpy.coordinate.frames.STIXImaging` and the associated transformations (`~stixpy.coordinates.transforms.stixim_to_hpc`, `~stixpy.coordinates.transforms.hpc_to_stixim`) to and from `~sunpy.coordinates.frames.Helioprojective`.
  Also add a `stixpy.map.stix.STIXMap` for `~sunpy.map.Map` source which properly displays the :class:`stixpy.coordinate.frames.STIXImaging` coordinate frame. (`#90 <https://github.com/TCDSolar/stixpy/pull/90>`__)
- Merged `stixpy.calibration.get_visibility_info_giordano` and `stixpy.calibration.get_visibility_info` into `stixpy.calibration.get_uv_points_data` and removed `stixpy.calibration.correct_phase_projection`. (`#98 <https://github.com/TCDSolar/stixpy/pull/98>`__)
- Update livetime correction parameter `eta`, `tau` and  add code to calculate pileup 'beta' :func:`stixpy.calibration.livetime.pileup_correction_factor`. (`#100 <https://github.com/TCDSolar/stixpy/pull/100>`__)
- Add background subtraction step to imaging demo, update `EnergyEdgeMasks`. (`#106 <https://github.com/TCDSolar/stixpy/pull/106>`__)
- Update imaging demo to use images and construct `~sunpy.map.Map` using `~stixpy.coordinates.frames.STIXImaging` and `~sunpy.map.header_helper.make_fitswcs_header`. (`#107 <https://github.com/TCDSolar/stixpy/pull/107>`__)
- Update :func:`~stixpy.coordinates.transforms.get_hpc_info` to use mean or interpolate pointing and location data based on number of data points in given timerange. Add function cache for aux data frequenly needed during coordinate transforms. (`#111 <https://github.com/TCDSolar/stixpy/pull/111>`__)
- Make the imaging demo compatible with :class:`~xrayvision.visibility.Visibilities` (`#123 <https://github.com/TCDSolar/stixpy/pull/123>`__)
- Update code and examples to use new :class:`~xrayvision.visibility.Visibilities`, tidy API of `~stixpy.imaging.em.em`. (`#124 <https://github.com/TCDSolar/stixpy/pull/124>`__)
- Add missing QL products `~stixpy.product.sources.quicklook.QLVariance`, `~stixpy.product.sources.quicklook.QLFlareFlag`, and `~stixpy.product.sources.quicklook.QLTMStatusFlareList` to Product. Also added some useful properties to the for status words. (`#125 <https://github.com/TCDSolar/stixpy/pull/125>`__)


Bug Fixes
---------

- Fix ELUT correction when the requested data has less then than 32 energy channels. (`#106 <https://github.com/TCDSolar/stixpy/pull/106>`__)
- Update `~stixpy.calibration.energy.get_elut` to use correct science energy channels for given date. (`#107 <https://github.com/TCDSolar/stixpy/pull/107>`__)
- Fix a bug where the incorrect unit was set on :class:`~stixpy.timeseries.quicklook.QLLightCurve`, :class:`~stixpy.timeseries.quicklook.QLBackground` and :class:`~stixpy.timeseries.quicklook.QLVariance`. (`#118 <https://github.com/TCDSolar/stixpy/pull/118>`__)
- Fix bug where coordinate transforms did not support arrays of coordinates or times. To so support both integrated (e.g. images) and instantaneous (e.g. coarse flare flag) a new property `obstime_end` was added to `~stixpy.coordinates.frames.STIXImaging` when set if possible mean pointing and position information are use for transformations other wise they are interpolated between the two nearest data points. (`#128 <https://github.com/TCDSolar/stixpy/pull/128>`__)
- Fix typo in `~stixpy.product.sources.science.ScienceData` 'durtaion' -> 'duration'. (`#132 <https://github.com/TCDSolar/stixpy/pull/132>`__)
- Fix some inconsistencies on the data units in plots, also return handle to the image from `~stixpy.product.sources.science.SpectrogramPlotMixin.plot_spectrogram`. (`#134 <https://github.com/TCDSolar/stixpy/pull/134>`__)


Improved Documentation
----------------------

- Update README, fix graphviz RTD configuration, configure theme logo text. (`#89 <https://github.com/TCDSolar/stixpy/pull/89>`__)
- Correct calculation of the HPC coordinate of the center of the STIX pointing in the imaging demo. (`#112 <https://github.com/TCDSolar/stixpy/pull/112>`__)
- Update imaging demo for latest changes in `xrayvision <https://github.com/TCDSolar/xrayvision>`_. (`#115 <https://github.com/TCDSolar/stixpy/pull/115>`__)


Trivial/Internal Changes
------------------------

- Format code with ruff and turn on ruff format in pre-commit. (`#116 <https://github.com/TCDSolar/stixpy/pull/116>`__)


Stixpy 0.1.3 (2024-07-16)
==================================================

Bug Fixes
---------

- Fix some inconsistencies on the data units in plots, also return handle to the image from `~stixpy.product.sources.science.SpectrogramPlotMixin.plot_spectrogram`. (`#134 <https://github.com/TCDSolar/stixpy/pull/134>`__)
- Fix bug reintroduce by last fix in `~stixpy.imaging.em.em`, update code in `~stixpy.calibration.visibility.calibrate_visibility` and imaging demo to use Tx, Ty rather than Ty, Tx as before. (`#137 <https://github.com/TCDSolar/stixpy/pull/137>`__)
