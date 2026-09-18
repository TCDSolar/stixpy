"""
I/O helpers shared across stixpy.

.. note::
   :func:`is_valid_fits` is currently **not wired into any download path**.
   The data server truncates large FITS responses under concurrent load; CI
   sidesteps this by running the online tests serially (``-n 1``, see tox.ini).
   The helper and its tests are kept here on purpose: if end-user (non-CI)
   resilience is wanted, reintroduce a validate + re-download guard at the two
   download boundaries that read from the server —

   * ``stixpy.product.product_factory.ProductFactory._parse_url``
     (``cache.download(url, redownload=True)`` on corruption), and
   * ``stixpy.coordinates._ephemeris_fetcher.fetch_ephemeris_for_range``
     (``Fido.fetch(query["stix"], overwrite=True)`` on corruption).

   Prefer an *on-error* redownload (try the read, redownload only if it fails) over
   pre-validating every load, to avoid reading each file twice on the happy path.
"""

import warnings

from astropy.io import fits

from stixpy.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["is_valid_fits"]


def is_valid_fits(path) -> bool:
    """
    Return whether ``path`` is a readable, structurally complete FITS file.

    Opens the file and forces a read of every HDU's data so that *truncated*
    downloads — which open fine but fail once the missing bytes are touched —
    are detected. Remote data servers occasionally truncate responses under
    concurrent load, and parfive caches the partial file as if it were
    complete; this check lets callers notice that and re-download.

    Warnings are suppressed during the check: valid STIX FITS files emit benign
    verification warnings (e.g. the non-standard ``BLANK`` keyword) which must
    not be mistaken for corruption. Only hard read errors count.

    Parameters
    ----------
    path
        Path-like to the FITS file.

    Returns
    -------
    bool
        ``True`` if the file opens and every HDU's data can be read.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path) as hdul:
                for hdu in hdul:
                    _ = hdu.data
    except Exception as e:
        logger.debug(f"FITS validation failed for {path}: {e}")
        return False
    return True
