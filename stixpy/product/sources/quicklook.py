from sunpy.time import TimeRange

from stixpy.product.product import L1Product


class QuickLookProduct(L1Product):
    """
    Basic science data class
    """
    @property
    def time_range(self):
        """
        A `sunpy.time.TimeRange` for the data.
        """
        return TimeRange(self.data['time'][0] - self.data['timedel'][0]/2,
                         self.data['time'][-1] + self.data['timedel'][-1]/2)


class QLLightCurve(QuickLookProduct):
    """
    Quicklook Lightcurves nominal in 5 energy bins every 4s
    """
    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (21, 6, 30) and level == 'L1':
            return True

    def __repr__(self):
        return f'{self.__class__.__name__}\n' \
               f'    {self.time_range}'


class QLBackground(QuickLookProduct):
    """
    Quicklook Background nominal in 5 energy bins every 8s
    """
    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (21, 6, 31) and level == 'L1':
            return True

    def __repr__(self):
        return f'{self.__class__.__name__}\n' \
               f'    {self.time_range}'
