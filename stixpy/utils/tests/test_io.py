import numpy as np

from astropy.io import fits

from stixpy.utils.io import is_valid_fits


def _write_fits(path, shape=(100, 10)):
    fits.HDUList([fits.PrimaryHDU(np.arange(np.prod(shape)).reshape(shape))]).writeto(path)
    return path


def test_is_valid_fits_complete(tmp_path):
    """A complete, readable FITS file validates."""
    path = _write_fits(tmp_path / "ok.fits")
    assert is_valid_fits(path) is True


def test_is_valid_fits_truncated(tmp_path):
    """A truncated download (data section cut off) is rejected."""
    path = _write_fits(tmp_path / "ok.fits")
    raw = path.read_bytes()
    truncated = tmp_path / "truncated.fits"
    truncated.write_bytes(raw[: len(raw) // 3])
    assert is_valid_fits(truncated) is False


def test_is_valid_fits_empty(tmp_path):
    """An empty file is rejected."""
    path = tmp_path / "empty.fits"
    path.write_bytes(b"")
    assert is_valid_fits(path) is False


def test_is_valid_fits_missing(tmp_path):
    """A non-existent path is rejected rather than raising."""
    assert is_valid_fits(tmp_path / "does-not-exist.fits") is False
