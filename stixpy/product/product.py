from astropy.time import Time


class BaseProduct:
    """
    Base Product that all other product inherit from contains the registry for the factory pattern
    """
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """__init_subclass__ hook initializes all of the subclasses of a given class.

        So for each subclass, it will call this block of code on import.
        This registers each subclass in a dict that has the `is_datasource_for` attribute.
        This is then passed into the factory so we can register them.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'is_datasource_for'):
            cls._registry[cls] = cls.is_datasource_for


class GenericProduct(BaseProduct):
    def __init__(self, *, meta, control, data, idb_versions=None, energies=None):
        """
        Generic product composed of meta, control, data and optionally idb, and energy information

        Parameters
        ----------
        control : stix_parser.products.quicklook.Control
            Table containing control information
        data : stix_parser.products.quicklook.Data
            Table containing data
        """
        self.meta = meta
        self.control = control
        self.data = data
        self.idb_versions = idb_versions
        self._energies = energies

    @property
    def level(self):
        return self.meta.get('level')

    @property
    def service_type(self):
        return self.meta.get('stype')

    @property
    def servie_subtype(self):
        return self.meta.get('stype')

    @property
    def ssid(self):
        return self.meta.get('ssid')


class LevelBinary(GenericProduct):
    """Level Binary data"""
    def __init__(self, **kwargs):
        raise NotImplementedError('Level Binary Data is not currently supported in stixpy')

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        level = meta['level']
        if level == 'LB':
            return True

class L1Product(GenericProduct):
    """Level Binary data"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        meta = kwargs['meta']
        data = kwargs['data']

        # TODO don't change the data add new property or similar
        try:
            data['time'] = Time(meta['date-obs']) + data['time']
        except KeyError:
            data['time'] = Time(meta['date_obs']) + data['time']

class Level2(GenericProduct):
    """Level Binary data"""

    def __init__(self, **kwargs):
        raise NotImplementedError('Level Binary Data is not currently supported in stixpy')

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        level = meta['level']
        if level == 'LB':
            return True
