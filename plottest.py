from matplotlib import pyplot as plt
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
from pathlib import Path

from stixpy.net.client import STIXClient  # Need to import to register with Fido
from stixpy import timeseries  # Need to import to register with timeseries
from stixpy.science import ScienceData
from stixcore.config.reader import read_subc_params
from stixpy.data import test

# Ok this look like an interesting event let look for some pixel data
query = Fido.search(a.Time('2020-11-17T00:00', '2020-11-17T23:59'), a.Instrument.stix, a.stix.DataProduct.sci_xray_l1)
query
# Again the 2nd to last file seem to match the time range of interest
pd_file = Fido.fetch(query[0])
pd = ScienceData.from_fits(pd_file[0])
print(pd_file[0])

# jsut printg the object gives a textual overview, pixel data support the same plot methods as
# pd

#l1 = ScienceData.from_fits(test.STIX_SCI_XRAY_L1)
l2 = ScienceData.from_fits(test.STIX_SCI_XRAY_L2)
l2.plot_pixels()

# plot the pixel data as default "kind=pixels"
pd.plot_pixels(energy_indices=[[1, 5], [5, 20], [20, 30]])
plt.show()

# plot the pixel data as line chart with errorbars"
pd.plot_pixels(kind='errorbar', energy_indices=[[1, 5], [5, 20], [20, 30]])
plt.show()

# plot the detector configuration (independent on the actual pixeldata)"
pd.plot_pixels(kind='config')
plt.show()
