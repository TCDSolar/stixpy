from astropy.utils import deprecated

from stixpy.products import Product


class StixpyDepreciationWarning(Warning):
    """
    A warning class to indicate a deprecated feature.
    """


@deprecated(since='0.1.0', alternative='stixpy.products.Product',
            warning_type=StixpyDepreciationWarning)
class ScienceData:
    @deprecated(since='0.1.0', alternative='stixpy.products.Product',
                warning_type=StixpyDepreciationWarning)
    @classmethod
    def from_fits(cls, file):
        return Product(file)
