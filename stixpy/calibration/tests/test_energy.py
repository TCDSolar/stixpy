from datetime import datetime

import pytest

from sunpy.util.exceptions import SunpyDeprecationWarning

from stixpy.calibration.energy import correct_counts, get_elut
from stixpy.product import Product


def test_get_elut_deprecated():
    with pytest.warns(SunpyDeprecationWarning, match=r"stixpy.calibration.elut.get_elut"):
        elut = get_elut(datetime(2024, 4, 9))
    assert elut.file == "elut_table_20240205.csv"


@pytest.mark.skip(reason="Better test data")
def test_correct_counts():
    uncorrected_prod = Product(
        "/Users/shane/Downloads/solo_L1_stix-sci-xray-cpd-2109270021_20210927T085625_20210927T092350_V01_63392.fits"
    )
    corrected_prod = correct_counts(uncorrected_prod)
    assert corrected_prod.e_cal == "rebin"
