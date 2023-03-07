from astropy.utils import deprecated

from stixpy.product import Product


class StixpyDepreciationWarning(Warning):
    @classmethod
    def from_fits(cls, file):
        return Product(file)
