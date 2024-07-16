Stixpy 0.1.3 (2024-07-16)
=========================

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
- Fix bug reintroduce by last fix in `~stixpy.imaging.em.em`, update code in `~stixpy.calibration.visibility.calibrate_visibility` and imaging demo to use Tx, Ty rather than Ty, Tx as before. (`#137 <https://github.com/TCDSolar/stixpy/pull/137>`__)


Improved Documentation
----------------------

- Update README, fix graphviz RTD configuration, configure theme logo text. (`#89 <https://github.com/TCDSolar/stixpy/pull/89>`__)
- Correct calculation of the HPC coordinate of the center of the STIX pointing in the imaging demo. (`#112 <https://github.com/TCDSolar/stixpy/pull/112>`__)
- Update imaging demo for latest changes in `xrayvision <https://github.com/TCDSolar/xrayvision>`_. (`#115 <https://github.com/TCDSolar/stixpy/pull/115>`__)


Trivial/Internal Changes
------------------------

- Format code with ruff and turn on ruff format in pre-commit. (`#116 <https://github.com/TCDSolar/stixpy/pull/116>`__)
