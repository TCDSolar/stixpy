import astropy.units as u
import numpy as np
from astropy.time import Time
from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate

from stixpy.extern.meta import Meta

# Lets fake some STIX pixel like data
pd_shape = (2, 3, 4, 5)  # time, detector, pixel, energy
pd = np.arange(np.prod(pd_shape)).reshape(pd_shape) * u.ct

# and associated times and quantities
times = Time("2024-01-01T00:00:00") + np.arange(pd_shape[0]) * 2 * u.s
energies = (1 + 2 * np.arange(pd_shape[-1])) * u.keV

# Create a gwcs
energy_coord = QuantityTableCoordinate(energies, names="energy", physical_types="em.energy")
time_coord = TimeTableCoordinate(times, names="time", physical_types="time")

combine_coord = time_coord & energy_coord

meta = Meta({"detector": ["a", "b", "c"]}, data_shape=pd_shape)

scube = NDCube(data=pd, wcs=combine_coord.wcs, meta=meta)
