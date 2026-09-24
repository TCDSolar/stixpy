.. _citing:

*****************
Citing ``STIXpy``
*****************

If you use ``STIXpy`` in your scientific work please citing it in
your publications. The continued growth and development of ``STIXpy`` is dependent
on the community being aware of the software.

Citing ``STIXpy`` in Publications
=================================

Please add a line such as the following within your methods, conclusion, or
acknowledgements section:

   *This research used version X.Y.Z (software citation) of the STIXpy
   open-source Python package (Maloney et al.).*

The **software citation** should be the Zenodo DOI for the specific version
you used. You can find the DOI for each release on the
`STIXpy Zenodo page <https://zenodo.org/doi/10.5281/zenodo.8086101>`_.
The concept DOI (covering all versions) is:

.. code-block:: none

   https://doi.org/10.5281/zenodo.8086101

BibTeX Entry
============

Use the following BibTeX entry, updating the ``year``, ``version``, and
``doi`` fields to match the release you used:

.. code:: bibtex

    @software{stixpy,
      author       = {Maloney, Shane and
                     Clarke, Brendan and
                     Hayes, Laura and
                     Ryan, Daniel F. and
                     Massa, Paolo and
                     Wilson, Alasdair and
                     Hochmuth, Nicky and
                     Christe, Steven and
                     Collier, Hannah and
                     Long, Thomas},
      title        = {STIXpy},
      abstract     = {An open-source Python analysis library for the Spectrometer Telescope for Imaging X-rays (STIX) instrument on Solar Obriter.},
      year         = {2026},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.8086101},
      url          = {https://github.com/TCDSolar/stixpy},
      license      = {BSD-3-Clause}
    }

You can also retrieve the citation string programmatically::

   import stixpy
   print(stixpy.__citation__)

.. note::

   If a peer-reviewed paper describing STIXpy is published, this page will be
   updated to include it as a preferred citation.

Acknowledging Dependencies
==========================

``STIXpy`` is built on a number of open-source packages. Where appropriate, please
also consider citing:

- `SunPy <https://docs.sunpy.org/en/stable/citation.html>`_ — solar data
  analysis framework that STIXpy builds on.
- `Astropy <https://www.astropy.org/acknowledging.html>`_ — core astronomy
  Python library.
- `Solar Orbiter / STIX instrument <https://doi.org/10.1051/0004-6361/201937154>`_
  — the Krucker et al. (2020) instrument paper.
