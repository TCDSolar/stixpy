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
