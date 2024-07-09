import pytest
from astropy.coordinates import SkyCoord

from stixpy.product import Product
from stixpy.product.sources import QLFlareFlag


@pytest.mark.remote_data
def test_qlflareflag():
    ff = Product("https://pub099.cs.technik.fhnw.ch/fits/L1/2024/06/03/QL/solo_L1_stix-ql-flareflag_20240603_V02.fits")
    assert isinstance(ff, QLFlareFlag)
    assert isinstance(ff.flare_location, SkyCoord)
    assert ff.flare_location[0].transform_to(frame="helioprojective").frame.name == "helioprojective"
    assert ff.location_flag[0, 0] == "None"
    assert ff.thermal_index[0, 0] == "Minor"
    assert ff.non_thermal_index[0, 0] == "None"
