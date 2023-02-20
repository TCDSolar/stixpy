from astropy.utils import deprecated

from stixpy.product import Product


class StixpyDepreciationWarning(Warning):
<<<<<<< HEAD
=======
    """
    A warning class to indicate a deprecated feature.
    """


@deprecated(since='0.1.0', alternative='stixpy.product.Product',
            warning_type=StixpyDepreciationWarning)
class ScienceData:
    @deprecated(since='0.1.0', alternative='stixpy.product.Product',
                warning_type=StixpyDepreciationWarning)
>>>>>>> 40540fc (Refactor docs)
    @classmethod
    def from_fits(cls, file):
        return Product(file)
