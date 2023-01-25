from stixpy.products import Product


class ScienceData:
    @classmethod
    def from_fits(cls, file):
        return Product(file)
