.. _product:

Product ('stixpy.product')
**************************

.. module:: stixpy.product

.. testsetup::

    import sunpy

.. currentmodule:: stixpy.product

Overview
========
One of core classes in stixpy is a Product. A stixpy Product object is a
container for STIX data. In order to make it easy to work
with STIX data in stixpy, the Product object provides a number of methods for
commonly performed operations.

Products are subclasses of `~stixpy.product.GenericProduct` and these objects are
created using the Product factory `~stixpy.product.Product`.


Creating Product Objects
========================
stixpy Product objects are constructed using the special factory
class `~stixpy.product.Product`: ::

    >>> x = stixpy.product.Product('file.fits')  # doctest: +SKIP

The result of a call to `~stixpy.product.Product` will be either a `~stixpy.product.GenericProduct` object,
or a subclass of `~stixpy.product.GenericProduct` which either deals with a specific type of STIX data,
e.g. `~stixpy.product.source.quicklook.QLLightCurve` or `~stixpy.product.sources.science.Spectrogram`
(see :ref:`product-sources` to see a list of all of them), or if no
product matches, `~stixpy.product.product.GenericProduct`.


.. autoclass:: stixpy.product.product_factory.ProductFactory

.. _product-classes:

stixpy.product Package
======================

All stixpy Products are derived from `stixpy.product.GenericProduct`, all the methods and attributes are documented in that class.

.. automodapi:: stixpy.product
    :no-main-docstr:
    :inherited-members:
    :include-all-objects:

.. _product-sources:

Product Classes
===============
Defined in ``stixpy.product.sources`` are a set of `~stixpy.product.GenericProduct` subclasses
which convert the specific metadata and other differences in each data type a standard `~stixpy.product.GenericProduct` interface.
These subclasses also provide a method, which describes to the `~stixpy.product.Product` factory
which inputs match its product.

.. automodapi:: stixpy.product.sources
    :no-main-docstr:
    :inherited-members:
