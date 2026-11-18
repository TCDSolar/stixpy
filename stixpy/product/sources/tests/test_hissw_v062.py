"""Opt-in live checks. Set STIX_GSW_REPO and run pytest -m hissw --remote-data=any."""

import os
import subprocess

import numpy as np
import pytest

import astropy.units as u

from sunpy.data import cache

from stixpy.product import Product
from stixpy.product.sources._selection import _time_indices_format
from stixpy.product.sources.science import subtract_background
from stixpy.product.sources.tests import test_science_sunkit_spex_1dspec as pixel
from stixpy.product.sources.tests import test_science_sunkit_spex_1dspec_spec as spectrogram

pytestmark = [pytest.mark.hissw, pytest.mark.remote_data]


@pytest.fixture(scope="module")
def idl(tmp_path_factory):
    repo = os.environ.get("STIX_GSW_REPO")
    if not repo:
        pytest.skip("Set STIX_GSW_REPO to a checkout with the v0.6.2 tag.")
    hissw = pytest.importorskip("hissw")
    root = tmp_path_factory.mktemp("stix-gsw-v062")
    archive = subprocess.check_output(["git", "-C", repo, "archive", "v0.6.2", "stix"])
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    source_root = root / "stix" / "idl"

    # Capture each converter's output immediately before its OSPEX handoff.
    for routine in ("stx_convert_pixel_data", "stx_convert_spectrogram"):
        source = (source_root / "processing" / "spectrogram" / f"{routine}.pro").read_text()
        handoff = source.index("  stx_convert_spectrogram2ospex, spec_data")
        capture = (
            "  common stixpy_compare, live_counts, live_errors\n  live_counts = spec\n  live_errors = spec_error\nend\n"
        )
        (root / f"{routine}.pro").write_text(source[:handoff] + capture)

    return hissw.Environment(
        ssw_packages=["stix", "spex", "hessi", "xray"],
        extra_paths=[str(root), f"+{source_root}"],
        idl_home=os.environ.get("IDL_DIR"),
        ssw_home=os.environ.get("SSW"),
    )


@pytest.mark.parametrize("with_background", [False, True])
@pytest.mark.parametrize("case", ["top24", "background_detector", "spectrogram"])
def test_stix_gsw_v062(idl, case, with_background):
    if case == "spectrogram":
        url, window = spectrogram.SPEC_URL, spectrogram.T_RANGE
        routine, idl_selection, selection = "stx_convert_spectrogram", "", {}
    elif case == "background_detector":
        url, window = pixel.CPD_URL, pixel.T_RANGE_BKGDET
        routine = "stx_convert_pixel_data"
        idl_selection = ", det_ind=[9], pix_ind=[2,5]"
        selection = {"detector_indices": [9], "pixel_indices": [2, 5]}
    else:
        url, window = pixel.CPD_URL, pixel.T_RANGE_TOP24
        routine, idl_selection = "stx_convert_pixel_data", ", det_ind='top24'"
        selection = {"detector_indices": "top24"}

    science_file = cache.download(url)
    background_file = cache.download(pixel.BKG_URL) if with_background else None
    background_arg = f", fits_path_bk='{background_file}'" if with_background else ""
    result = idl.run(
        f"""
common stixpy_compare, live_counts, live_errors
{routine}, fits_path_data='{science_file}'{background_arg}{idl_selection}, plot=0, /silent
""",
        save_vars=["live_counts", "live_errors"],
        verbose=False,
    )
    science = Product(science_file)
    start, end = _time_indices_format(window, science.times, science.data["timedel"])[0]
    data = science.get_data(
        vtype="c",
        time_indices=list(range(start, end + 1)),
        elut_correction=False,
        sum_detectors=True,
        ssw_compatible=True,
        **selection,
    )
    if with_background:
        data = subtract_background(data, Product(background_file))

    for name, values in (("live_counts", data.counts), ("live_errors", data.counts_err)):
        assert result[name].shape[0] == len(science.times)
        np.testing.assert_allclose(values.to_value(u.ct)[:, 0, 0], result[name][start : end + 1], rtol=1e-7, atol=1e-6)
