from sunpy.net.attr import SimpleAttr

__all__ = ["DataType", "DataProduct"]


class DataType(SimpleAttr):
    """
    Data type Quicklook (QL), science (SCI), or housekeeping (HK)
    """


class DataProduct(SimpleAttr):
    """
    Data product
    """
