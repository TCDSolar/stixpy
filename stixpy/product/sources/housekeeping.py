from sunpy.time import TimeRange

from stixpy.product.product import L1Product


class HouseKeepingProduct(L1Product):
    """
    Basic HouseKeeping
    """
    @property
    def time_range(self):
        """
        A `sunpy.time.TimeRange` for the data.
        """
        return TimeRange(self.data['time'][0] - self.data['timedel'][0]/2,
                         self.data['time'][-1] + self.data['timedel'][-1]/2)

class HKMini(HouseKeepingProduct):
    """
    HK “mini” report only generated during the Start-up SW.
    """
    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (3, 25, 1) and level == 'L1':
            return True

    def __repr__(self):
        return f'{self.__class__.__name__}\n' \
               f'    {self.time_range}'


class HKMaxi(HouseKeepingProduct):
    """
    HK “maxi” report with is reported in all modes of the Application SW. In Maintenance mode,
    however, all subsystems including Aspect System have to be disabled upon entry to the mode,
    so most of the HK will default to zero.
    """
    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ['STYPE', 'SSTYPE', 'SSID'])
        level = meta['level']
        if service_subservice_ssid == (3, 25, 2) and level == 'L1':
            return True

    def __repr__(self):
        return f'{self.__class__.__name__}\n' \
               f'    {self.time_range}'
