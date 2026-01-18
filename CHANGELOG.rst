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


Stixpy 0.1.3.dev14+gb8fe13f.d20250429 (2025-11-07)
==================================================

Bug Fixes
---------

- Fix some inconsistencies on the data units in plots, also return handle to the image from `~stixpy.product.sources.science.SpectrogramPlotMixin.plot_spectrogram`. (`#134 <https://github.com/TCDSolar/stixpy/pull/134>`__)
- Fix bug reintroduce by last fix in `~stixpy.imaging.em.em`, update code in `~stixpy.calibration.visibility.calibrate_visibility` and imaging demo to use Tx, Ty rather than Ty, Tx as before. (`#137 <https://github.com/TCDSolar/stixpy/pull/137>`__)
