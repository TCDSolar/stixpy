#!/usr/bin/env python
"""
Two-pass test runner used by the ``online`` tox environment.

Remote test data is served by a single host (``pub099.cs.technik.fhnw.ch``, which
``dataarchive.stix.i4ds.net`` redirects to). Under the concurrency of
``pytest -n auto`` that server occasionally truncates a response, and parfive caches
the partial file as if it were complete, so the test fails with an
``Empty or corrupt FITS file`` / ``No SIMPLE card found`` style ``OSError``. The
failure is intermittent and goes away on a clean, serial retry.

This wrapper therefore:

1. runs pytest once with whatever arguments it is given (the fast parallel pass), and
2. only if that pass fails *and* it was a remote-data run, clears the sunpy/parfive
   download caches and re-runs **just the failed tests** serially (``-n 1``) once.

``--last-failed`` selects the failed items by node id, so it covers regular tests,
doctests and setup errors alike. The wrapper is plain stdlib so it runs everywhere
tox does.

Usage (from tox)::

    python tools/run_online_tests.py <pytest args...>
"""

import sys
import shutil
import pathlib
import subprocess

# sunpy download locations to purge before the serial retry (see module docstring).
# ``~/.cache/sunpy`` backs ``sunpy.data.cache.download`` (Product + TimeSeries URLs);
# ``~/sunpy/data`` is the Fido download dir (ANC ephemeris).
CACHE_DIRS = (
    pathlib.Path.home() / ".cache" / "sunpy",
    pathlib.Path.home() / "sunpy" / "data",
)


def _is_remote_data_run(args):
    """Whether pytest was asked to run remote-data tests."""
    return any(a == "--remote-data" or a.startswith("--remote-data=") for a in args)


def _strip_numprocesses(args):
    """Drop any xdist ``-n``/``--numprocesses`` option so we can force serial."""
    out = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "-n" or a == "--numprocesses":
            skip_next = True  # also drop the following value (e.g. "auto")
            continue
        if a.startswith("-n") and a != "-n" or a.startswith("--numprocesses="):
            continue
        out.append(a)
    return out


def _run(args):
    return subprocess.call([sys.executable, "-m", "pytest", *args])


def _clear_caches():
    for path in CACHE_DIRS:
        shutil.rmtree(path, ignore_errors=True)
    print(f"Cleared download caches: {', '.join(str(p) for p in CACHE_DIRS)}", flush=True)


def main(argv):
    args = list(argv)

    rc = _run(args)
    if rc == 0:
        return rc

    if not _is_remote_data_run(args):
        # Offline failure -- a real problem, do not retry/mask it.
        return rc

    # GitHub Actions surfaces ``::warning::`` lines in the job summary.
    print(
        "::warning::Online tests failed; clearing the sunpy download cache and "
        "re-running the failed tests serially (-n 1). This guards against the data "
        "server truncating responses under concurrent load.",
        flush=True,
    )
    _clear_caches()

    rerun_args = [*_strip_numprocesses(args), "--last-failed", "--cov-append", "-n", "1"]
    return _run(rerun_args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
