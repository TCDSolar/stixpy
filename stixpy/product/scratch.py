import astropy.units as u
import numpy as np
from astropy.nddata import NDUncertainty
from astropy.time import Time
from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate

from stixpy.extern.meta import Meta


class PossionUncertainty(NDUncertainty):
    @property
    def uncertainty_type(self):
        return "possion"

    def _data_unit_to_uncertainty_unit(self, value):
        return value.unit

    def _propagate_add(self, other_uncert, *args, **kwargs):
        return np.sqrt

    def _propagate_subtract(self, other_uncert, *args, **kwargs):
        return None

    def _propagate_multiply(self, other_uncert, *args, **kwargs):
        return None

    def _propagate_divide(self, other_uncert, *args, **kwargs):
        return None


# Lets fake some STIX pixel like data
pd_shape = (2, 3, 4, 5)  # time, detector, pixel, energy
pd = np.arange(np.prod(pd_shape)).reshape(pd_shape) * u.ct

# and associated times and quantities
times = Time("2024-01-01T00:00:00") + np.arange(pd_shape[0]) * 2 * u.s
energies = (1 + 2 * np.arange(pd_shape[-1])) * u.keV

# Create table coords
time_coord = TimeTableCoordinate(times, names="time", physical_types="time")
energy_coord = QuantityTableCoordinate(energies, names="energy", physical_types="em.energy")

# fake coords for detector and pixel
detector_coord = QuantityTableCoordinate(
    np.arange(pd_shape[1]) * u.dimensionless_unscaled, names="detector", physical_types="custom:detector"
)
pixel_coord = QuantityTableCoordinate(
    np.arange(pd_shape[2]) * u.dimensionless_unscaled, names="pixel", physical_types="custom:pixel"
)

combine_coord = energy_coord & pixel_coord & detector_coord & time_coord


# I'm not sure what should live in the meta vs extra_coords
meta = Meta({"detector": ["a", "b", "c"]}, data_shape=pd_shape)

scube = NDCube(data=pd, wcs=combine_coord.wcs, meta=meta)
scube.extra_coords.add("integration_time", 0, [2, 3] * u.s, physical_types="time.duration")
scube
