from matplotlib import pyplot as plt
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
from pathlib import Path

from stixpy.net.client import STIXClient  # Need to import to register with Fido
from stixpy import timeseries  # Need to import to register with timeseries
from stixpy.science import ScienceData
from stixcore.config.reader import read_subc_params

# Ok this look like an interesting event let look for some pixel data
query = Fido.search(a.Time('2020-11-17T00:00', '2020-11-17T23:59'), a.Instrument.stix, a.stix.DataProduct.sci_xray_l1)
query
# Again the 2nd to last file seem to match the time range of interest
pd_file = Fido.fetch(query[0, -2])
pd = ScienceData.from_fits(pd_file[0])

# jsut printg the object gives a textual overview, pixel data support the same plot methods as
pd

scp = read_subc_params(Path(read_subc_params.__code__.co_filename).parent
                            / "data" / "common" / "detector" / "stx_subc_params.csv")


plt.plot(scp["SC Xcen"], scp["SC Ycen"], 'ro')
for i, txt in enumerate(scp['Grid Label']):
    plt.text(scp["SC Xcen"][i]+.03, scp["SC Ycen"][i]+0.3, txt, fontsize=9)

plt.show()

# spectrograms with an additional method to plot the pixel data
pd.plot_pixels(energy_indices=[[1, 10], [10, 30]])
plt.show()

print("Hallo")
