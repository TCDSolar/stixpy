import astropy.units as u
from astropy import wcs
from sunpy.coordinates import HeliographicStonyhurst
from sunpy.map import GenericMap

import stixpy.coordinates.transforms

__all__ = ["STIXMap"]


class STIXMap(GenericMap):
    r"""
    A class to represent a STIX Map.
    """

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        return str(header.get("TELESCOP", "")).endswith("STIX")

    def plot(self, **kwargs):
        # Until this is resolved need to manually override https://github.com/astropy/astropy/issues/16339
        res = super().plot(**kwargs)
        if self.coordinate_frame.name == "stiximaging":
            res.axes.coords[0].set_coord_type("longitude", coord_wrap=180 * u.deg)
            res.axes.coords[1].set_coord_type("latitude")

            res.axes.coords[0].set_format_unit(u.arcsec)
            res.axes.coords[1].set_format_unit(u.arcsec)

            res.axes.coords[0].set_axislabel(r"STIX $\Theta_{X}$")
            res.axes.coords[1].set_axislabel(r"STIX $\Theta_{Y}$")

        return res

    @property
    def wcs(self):
        """
        The `~astropy.wcs.WCS` property of the map.
        """
        w2 = wcs.WCS(naxis=2)

        # Add one to go from zero-based to one-based indexing
        w2.wcs.crpix = u.Quantity(self.reference_pixel) + 1 * u.pix
        # Make these a quantity array to prevent the numpy setting element of
        # array with sequence error.
        # Explicitly call ``.to()`` to check that scale is in the correct units
        w2.wcs.cdelt = u.Quantity(
            [self.scale[0].to(self.spatial_units[0] / u.pix), self.scale[1].to(self.spatial_units[1] / u.pix)]
        )
        w2.wcs.crval = u.Quantity([self._reference_longitude, self._reference_latitude])
        w2.wcs.ctype = self.coordinate_system
        w2.wcs.pc = self.rotation_matrix
        w2.wcs.set_pv(self._pv_values)
        # FITS standard doesn't allow both PC_ij *and* CROTA keywords
        w2.wcs.crota = (0, 0)
        w2.wcs.cunit = self.spatial_units

        if self.date_start is not None:
            w2.wcs.datebeg = self.date_start.isot
        if self.date_average is not None:
            w2.wcs.dateavg = self.date_average.isot
        if self.date_end is not None:
            w2.wcs.dateend = self.date_end.isot
        if self.date is not None:
            w2.wcs.dateobs = self.date.isot

        w2.wcs.aux.rsun_ref = self.rsun_meters.to_value(u.m)
        w2.wcs.aux.hgln_obs = self.observer_coordinate.lon.to_value(u.deg)
        w2.wcs.aux.hglt_obs = self.observer_coordinate.lat.to_value(u.deg)
        w2.wcs.aux.dsun_obs = self.observer_coordinate.radius.to_value(u.m)

        # Set the shape of the data array
        w2.array_shape = self.data.shape

        # Validate the WCS here.
        w2.wcs.set()
        return w2

    def to_hpc(self,
               reference_coordinate,
               reference_pixel: u.Unit('pix') = None):
        """
        Return a version of the map in the Helioprojective Cartesian (HPC) coordinate frame.

        This is quicker than a full reprojection, `~stixpy.coordinates.transforms.STIXImaging`
        is also a helioprojective frame with a different z-axis and rotation.
        Therefore, the new WCS information can be unambiguously reconstructed without
        altering the data array.

        Parameters
        ----------
        reference_coordinate: `astropy.coordinates.SkyCoord`
            The coordinate of the reference pixel.
            Must be transformable to `sunpy.coordinates.Helioprojective`.
        reference_pixel: `astropy.coordinate.Quantity` length-2 (optional)
            The (x, y) pixel index of the reference pixel.
            Default is center of the map's field of view.

        Returns
        -------
        hpc_map: `sunpy.map.Map`
            Map of the STIX image in the HPC coordinate frame.
        """
        if reference_pixel is None:
            reference_pixel = (np.array(self.data.shape)[::-1] / 2) << u.pix
        roll, solo_xyz, pointing = stixpy.coordinates.transforms.get_hpc_info(
            reference_coordinate.obstime)
        solo = HeliographicStonyhurst(*solo_xyz, obstime=reference_coordinate.obstime,
                                      representation_type="cartesian")
        header = make_fitswcs_header(
            self.data,
            reference_coordinate.transform_to(Helioprojective(obstime=reference_coordinate.obstime,
                                                              observer=solo)),
            telescope="STIX",
            observatory="Solar Orbiter",
            scale=u.Quantity(self.scale),
            rotation_angle=90 * u.deg + roll,)
        return sunpy.map.Map(self.data, header,
                             mask=self.mask, uncertainty=self.uncertainty, meta=self.meta)
