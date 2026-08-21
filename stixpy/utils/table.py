"""Generic helpers for `astropy.table` objects used across stixpy."""

__all__ = ["drop_fits_checksums"]

_NOISY_FITS_META_KEYS = ("DATASUM", "CHECKSUM")


def drop_fits_checksums(*tables) -> None:
    """
    Remove per-file FITS integrity checksums from each table's ``meta`` in place.

    ``DATASUM`` and ``CHECKSUM`` are HDU-level integrity hashes that correctly
    differ between source FITS files. When stacking tables from multiple files
    (e.g. daily ANC files), the merged table has no meaningful single
    checksum — dropping the keys before `~astropy.table.vstack` avoids the
    ``MergeConflictWarning`` while keeping any *other* metadata conflicts
    visible.

    Parameters
    ----------
    *tables
        One or more `~astropy.table.Table` (or subclass) instances. Each
        table's ``meta`` is mutated in place; missing keys are ignored.
    """
    for table in tables:
        for key in _NOISY_FITS_META_KEYS:
            table.meta.pop(key, None)
