from pathlib import Path

import pytest

from stixpy.data import test
from stixpy.product.sources import QLLightCurve, Spectrogram
from stixpy.product.product_factory import Product

from stixpy.science import *

def test_product_factory_file():
    prod = Product(test.STIX_SCI_XRAY_SPEC)
    assert isinstance(prod, Spectrogram)

@pytest.mark.remote_data
def test_product_factory_url():
    ql_lc = Product('https://pub099.cs.technik.fhnw.ch/data/fits/L1/2022/11/15/QL/solo_L1_stix-ql-lightcurve_20221115_V02.fits')
    assert isinstance(ql_lc, QLLightCurve)

@pytest.mark.skip
@pytest.mark.remote_data
def test_product_factory_directory():
    test_data_path = 'https://pub099.cs.technik.fhnw.ch/data/fits/L1/2022/11/15/QL/' #Path(__file__).parent.parent.parent / 'data'
    products = Product(test_data_path, silence_errors=True)
    assert products
