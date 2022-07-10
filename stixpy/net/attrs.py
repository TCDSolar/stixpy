from sunpy.net.attr import SimpleAttr

__all__ = ['DataType', 'DataProduct']

DATA_TYPES = ['QL', 'SCI', 'HK', 'CAL']
DATA_PRODUCTS = ['ql_lightcurve', 'ql_background', 'ql_variance', 'ql_spectra',
                 'ql_calibration_spectrum', 'ql_flareflag', 'sci_xray_l0','sci_xray_l1',
                 'sci_xray_l2', 'sci_xray_l3', 'sci_spectrogram', 'sci_visibility']


class DataType(SimpleAttr):
    """
    Data type Quicklook (QL), science (SCI), or housekeeping (HK)
    """
    # def __init__(self, value):
    #     if str(value).casefold not in [t.casefold for t in DATA_TYPES]:
    #         raise ValueError(f'Data type must be one of {DATA_TYPES} not {value}')
    #     super().__init__(value)


class DataProduct(SimpleAttr):
    """
    Data product
    """
    # def __init__(self, value):
    #     if str(value).casefold not in [t.casefold for t in DATA_PRODUCTS]:
    #         raise ValueError(f'Data product must be one of {DATA_PRODUCTS} not {value}')
    #     super().__init__(value)
