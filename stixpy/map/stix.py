import astropy.units as u
from sunpy.map import GenericMap

__all__ = ['STIXMap']


class STIXMap(GenericMap):
    r"""
    A class to represent a STIX Map.
    """

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        return str(header.get('TELESCOP', '')).endswith('STIX')

    def plot(self, **kwargs):

        # Until this is resolved need to manually override https://github.com/astropy/astropy/issues/16339
        res = super().plot(**kwargs)
        if self.coordinate_frame.name == 'stiximaging':
            res.axes.coords[0].set_coord_type('longitude', coord_wrap=180*u.deg)
            res.axes.coords[1].set_coord_type('latitude')

            res.axes.coords[0].set_format_unit(u.arcsec)
            res.axes.coords[1].set_format_unit(u.arcsec)

            res.axes.coords[0].set_axislabel(r'STIX $\Theta_{X}$')
            res.axes.coords[1].set_axislabel(r'STIX $\Theta_{Y}$')

        return res
