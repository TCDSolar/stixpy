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

    # @property
    # def coordinate_system(self):
    #     """
    #     Coordinate system used
    #
    #     Overrides the values in the header which are not understood by Astropy WCS
    #     """
    #     return SpatialPair("SXLN-TAN", "SXLT-TAN")
    #
    # @property
    # def spatial_units(self):
    #     return SpatialPair(u.arcsec, u.arcsec)

    def plot(self, **kwargs):

        # Until this is resolved need to manually override https://github.com/astropy/astropy/issues/16339
        res = super().plot(**kwargs)

        res.axes.coords[0].set_coord_type('longitude', coord_wrap=180*u.deg)
        res.axes.coords[1].set_coord_type('latitude')

        res.axes.coords[0].set_format_unit(u.arcsec)
        res.axes.coords[1].set_format_unit(u.arcsec)

        return res

    # @property
    # def wcs(self):
    #     if not self.coordinate_system.axis1 == 'SXLN-TAN' and self.coordinate_system.axis2 == 'SXLT-TAN':
    #         return super().wcs
    #
    #     w2 = astropy.wcs.WCS(naxis=2)
    #
    #     # Add one to go from zero-based to one-based indexing
    #     w2.wcs.crpix = u.Quantity(self.reference_pixel) + 1 * u.pix
    #     # Make these a quantity array to prevent the numpy setting element of
    #     # array with sequence error.
    #     # Explicitly call ``.to()`` to check that scale is in the correct units
    #     w2.wcs.cdelt = u.Quantity([self.scale[0].to(self.spatial_units[0] / u.pix),
    #                                self.scale[1].to(self.spatial_units[1] / u.pix)])
    #     w2.wcs.crval = u.Quantity([self._reference_longitude, self._reference_latitude])
    #     w2.wcs.ctype = self.coordinate_system
    #     w2.wcs.pc = self.rotation_matrix
    #     w2.wcs.set_pv(self._pv_values)
    #     # FITS standard doesn't allow both PC_ij *and* CROTA keywords
    #     w2.wcs.crota = (0, 0)
    #     w2.wcs.cunit = self.spatial_units
    #     w2.wcs.dateobs = self.date.isot
    #     w2.wcs.aux.rsun_ref = self.rsun_meters.to_value(u.m)
    #
    #     obs_frame = self.observer_coordinate
    #     if hasattr(obs_frame, 'frame'):
    #         obs_frame = obs_frame.frame
    #
    #     w2.wcs.aux.hgln_obs = obs_frame.lon.to_value(u.deg)
    #     w2.wcs.aux.hglt_obs = obs_frame.lat.to_value(u.deg)
    #     w2.wcs.aux.dsun_obs = obs_frame.radius.to_value(u.m)
    #
    #     w2.wcs.dateobs = self.date.utc.iso
    #
    #     # Set the shape of the data array
    #     w2.array_shape = self.data.shape
    #
    #     return w2
