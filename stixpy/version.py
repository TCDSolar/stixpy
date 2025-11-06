# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    try:
        from ._dev.scm_version import version
    except ImportError:
        from ._version import version
except Exception:
    import warnings

    warnings.warn(f"could not determine {__name__.split('.')[0]} package version; this indicates a broken installation")
    del warnings

    version = "0.0.0"

from packaging.version import parse as _parse

_version = _parse(version)
major, minor, bugfix = [*_version.release, 0][:3]
release = not _version.is_devrelease
