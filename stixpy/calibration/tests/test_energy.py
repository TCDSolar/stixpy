from datetime import datetime

import pytest

from stixpy.calibration.energy import correct_counts, get_elut
from stixpy.product import Product


def test_get_elut():
    with pytest.raises(ValueError, match=r"No ELUT for for date.*"):
        get_elut(datetime(2019, 1, 1))

    elut = get_elut(datetime(2020, 6, 7))
    assert elut.file == "elut_table_20200519.csv"

    elut = get_elut(datetime(2024, 4, 9))
    assert elut.file == "elut_table_20231019.csv"


@pytest.mark.skip(reason="Better test data")
def test_correct_counts():
    uncorrected_prod = Product(
        "/Users/shane/Downloads/solo_L1_stix-sci-xray-cpd-2109270021_20210927T085625_20210927T092350_V01_63392.fits"
    )
    corrected_prod = correct_counts(uncorrected_prod)
    assert corrected_prod.e_cal == "rebin"
