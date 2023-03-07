import pathlib
from urllib.request import Request

import numpy as np

import astropy.io.fits
from astropy.io import fits
from astropy.table import QTable

from sunpy.data import cache

from sunpy.util import expand_list
from sunpy.util.datatype_factory_base import (
    BasicRegistrationFactory,
    MultipleMatchError,
    NoMatchError,
    ValidationFunctionError,
)
from sunpy.util.exceptions import NoMapsInFileError, warn_user
from sunpy.util.functools import seconddispatch
from sunpy.util.io import is_url, parse_path, possibly_a_path
from sunpy.util.metadata import MetaDict

from stixpy.product.product import GenericProduct
from stixpy.utils.logging import get_logger
from stixpy.product.sources import *


__all__ = ['Product', 'ProductFactory']

BITS_TO_UINT = {
    8: np.ubyte,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64
}


logger = get_logger(__name__)


def read_qtable(file, hdu, hdul=None):
    """
    Read a fits extension into a QTable and maintain dtypes of columns with units

    Hack to work around QTable not respecting the dtype in fits file see
    https://github.com/astropy/astropy/issues/12494

    Parameters
    ----------
    input : `str`, `pathlib.Path`, `astropy.io.fits.HDUList` or `astropy.io.fits.HDU`
        Fits file
    hdu : int or str, optional
        The HDU to read the table from.
    hdul : `astropy.io.fits.HDUList`

    Returns
    -------
    `astropy.table.QTable`
        The corrected QTable with correct data types
    """
    if hdul is None:
        hdul = fits.open(file)

    qtable = QTable.read(file, hdu)

    for col in hdul[hdu].data.columns:
        if col.unit:
            logger.debug(f'Unit present dtype correction needed for {col}')
            dtype = col.dtype

            if col.bzero:
                logger.debug(f'Unit present dtype and bzero correction needed for {col}')
                bits = np.log2(col.bzero)
                if bits.is_integer():
                    dtype = BITS_TO_UINT[int(bits+1)]

            if hasattr(dtype, 'subdtype'):
                dtype = dtype.base

            qtable[col.name] = qtable[col.name].astype(dtype)

    return qtable


class ProductFactory(BasicRegistrationFactory):
    """
    A factory for generating stix data products

    Parameters
    ----------
    \\*inputs
        Inputs to parse for map objects. See the examples section for a
        detailed list of accepted inputs.

    Returns
    -------
    `stixpy.product.Product`
        If the input results in a singular product, then that is returned.
    `list` of `~stixpy.products.Product`
        If multiple inputs a list of `~stixpy.products.Product` objects will be
        returned.

    Examples
    --------

    """

    def _read_file(self, fname, **kwargs):
        """
        Read in a file name and return the list of (data, meta) pairs in that file.
        """
        # File gets read here.
        # NOTE: use os.fspath so that fname can be either a str or pathlib.Path
        # This can be removed once read_file supports pathlib.Path
        logger.debug(f'Reading {fname}')
        name = str(fname)
        if not name.endswith('.fits'):
            raise ValueError(f'File {fname} is not a fits file, only fits are supported')
        try:
            hdul = fits.open(fname, **kwargs)
        except Exception as e:
            msg = f"Failed to read {fname}."
            raise IOError(msg) from e

        if hdul[0].header.get('INSTRUME', '') != 'STIX':
            raise FileError(f"File '{fname}' is not a STIX fits file.")

        data = {'meta': hdul[0].header}
        for name in ['CONTROL', 'DATA', 'IDB_VERSIONS', 'ENERGIES']:
            try:
                data[name.lower()] = read_qtable(fname, hdu=name)
            except KeyError as e:
                if name in ('IDB_VERSIONS', 'ENERGIES'):
                    logger.debug(f"Extension '{name}' not in file '{fname}'")
                else:
                    raise e

        return [data]

    def _validate_meta(self, meta):
        """
        Validate a meta argument.
        """
        if isinstance(meta, astropy.io.fits.header.Header):
            return True
        elif isinstance(meta, dict):
            return True
        else:
            return False

    def _parse_args(self, *args, silence_errors=False, **kwargs):
        """
        Parses an args list into data-header pairs.
        args can contain any mixture of the following entries:
        * filename, as a str or pathlib.Path, which will be read
        * directory, as a str or pathlib.Path, from which all files will be read
        * glob, from which all files will be read
        * url, which will be downloaded and read
        * lists containing any of the above.
        Examples
        --------
        self._parse_args(['file1', 'file2', 'file3'],
                         'file4',
                         'directory1',
                         '*.fits')
        """
        # Account for nested lists of items
        args = expand_list(args)

        # Sanitise the input so that each 'type' of input corresponds to a different
        # class, so single dispatch can be used later
        nargs = len(args)
        i = 0
        while i < nargs:
            arg = args[i]
            if isinstance(arg, str) and is_url(arg):
                # Repalce URL string with a Request object to dispatch on later
                args[i] = Request(arg)
            elif possibly_a_path(arg):
                # Repalce path strings with Path objects
                args[i] = pathlib.Path(arg)
            i += 1

        # Parse the arguments
        # Note that this list can also contain GenericProducts if they are directly given to the factory
        meta_data_dicts = []
        for arg in args:
            try:
                meta_data_dicts += self._parse_arg(arg, **kwargs)
            except NoMapsInFileError as e:
                if not silence_errors:
                    raise
                warn_user(f"One of the arguments failed to parse with error: {e}")

        return meta_data_dicts

    # Note that post python 3.8 this can be @functools.singledispatchmethod
    @seconddispatch
    def _parse_arg(self, arg, **kwargs):
        """
        Take a factory input and parse into (data, header) pairs.
        Must return a list, even if only one pair is returned.
        """
        raise ValueError(f"Invalid input: {arg}")

    @_parse_arg.register(tuple)
    def _parse_tuple(self, arg, **kwargs):
        meta, control, data, *other = arg

        if self._validate_meta(meta):
            pair = (meta, control, data, *other)
        return [pair]

    # @_parse_arg.register(DatabaseEntryType)
    # def _parse_dbase(self, arg, **kwargs):
    #     return self._read_file(arg.path, **kwargs)

    @_parse_arg.register(GenericProduct)
    def _parse_map(self, arg, **kwargs):
        return [arg]

    @_parse_arg.register(Request)
    def _parse_url(self, arg, **kwargs):
        url = arg.full_url
        path = str(cache.download(url).absolute())
        pairs = self._read_file(path, **kwargs)
        return pairs

    @_parse_arg.register(pathlib.Path)
    def _parse_path(self, arg, **kwargs):
        return parse_path(arg, self._read_file, **kwargs)

    def __call__(self, *args, composite=False, sequence=False, silence_errors=False, **kwargs):
        """ Method for running the factory. Takes arbitrary arguments and
        keyword arguments and passes them to a sequence of pre-registered types
        to determine which is the correct Map-type to build.
        Arguments args and kwargs are passed through to the validation
        function and to the constructor for the final type. For Map types,
        validation function must take a data-header pair as an argument.
        Parameters
        ----------
        composite : `bool`, optional
            Indicates if collection of maps should be returned as a `~sunpy.map.CompositeMap`.
            Default is `False`.
        sequence : `bool`, optional
            Indicates if collection of maps should be returned as a `sunpy.map.MapSequence`.
            Default is `False`.
        silence_errors : `bool`, optional
            If set, ignore data-header pairs which cause an exception.
            Default is ``False``.
        Notes
        -----
        Extra keyword arguments are passed through to `sunpy.io.read_file` such
        as `memmap` for FITS files.
        """
        meta_control_data_dicts = self._parse_args(*args, silence_errors=silence_errors, **kwargs)
        new_products = list()

        # Loop over each registered type and check to see if WidgetType
        # matches the arguments.  If it does, use that type.
        for meta_control_data in meta_control_data_dicts:
            if isinstance(meta_control_data, GenericProduct):
                new_products.append(meta_control_data)
                continue

            try:
                new_product = self._check_registered_widgets(**meta_control_data, **kwargs)
                new_products.append(new_product)
            except (NoMatchError, MultipleMatchError,
                    ValidationFunctionError) as e:
                if not silence_errors:
                    raise
                warn_user(f"One of the inputs failed to validate with: {e}")

        if not len(new_products):
            raise RuntimeError('No maps loaded')

        if len(new_products) == 1:
            return new_products[0]

        return new_products

    def _check_registered_widgets(self, *, meta, control, data, **kwargs):

        candidate_widget_types = list()

        for key in self.registry:

            # Call the registered validation function for each registered class
            if self.registry[key](meta=meta, **kwargs):
                candidate_widget_types.append(key)

        n_matches = len(candidate_widget_types)

        if n_matches == 0:
            if self.default_widget_type is None:
                raise NoMatchError("No types match specified arguments and no default is set.")
            else:
                candidate_widget_types = [self.default_widget_type]
        elif n_matches > 1:
            raise MultipleMatchError("Too many candidate types identified "
                                     f"({candidate_widget_types}). "
                                     "Specify enough keywords to guarantee unique type "
                                     "identification.")

        # Only one is found
        WidgetType = candidate_widget_types[0]

        return WidgetType(meta=meta, control=control, data=data, **kwargs)


class InvalidMapInput(ValueError):
    """Exception to raise when input variable is not a Map instance and does
    not point to a valid Map input file."""


class InvalidMapType(ValueError):
    """Exception to raise when an invalid type of map is requested with Map
    """


class FileError(ValueError):
    """Exception to raise when input is not valid STIX file
    """


Product = ProductFactory(registry=GenericProduct._registry, default_widget_type=GenericProduct,
                 additional_validation_functions=['is_datasource_for'])
