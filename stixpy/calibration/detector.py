from types import SimpleNamespace
from pathlib import Path

import astropy.units as u
import numpy as np
from sunpy.io.special import read_genx

from stixpy.calibration.transmission import Transmission
from stixpy.io.readers import read_energy_channel_index, read_sci_energy_channels

__all__ = ["get_srm", "get_pixel_srm", "get_sci_channels"]

SCI_INDEX = None
SCI_CHANNELS = {}


def get_srm():
    r"""
    Return the spectromber response matrix (SRM) by combing the attenuation with the detetor respoonse matrix (DRM)

    Returns
    -------

    """
    drm_save = read_genx("/Users/shane/Projects/STIX/git/stix_drm_20220713.genx")
    drm = drm_save["SAVEGEN0"]["SMATRIX"] * u.count / u.keV / u.photon
    energies_in = drm_save["SAVEGEN0"]["EDGES_IN"] * u.keV
    energies_in_width = np.diff(energies_in)
    energies_in_mean = energies_in[:-1] + energies_in_width / 2
    trans = Transmission()
    tot_trans = trans.get_transmission(energies=energies_in_mean)
    energies_out = drm_save["SAVEGEN0"]["EDGES_OUT"] * u.keV
    energies_out_width = drm_save["SAVEGEN0"]["EWIDTH"] * u.keV
    energies_out_mean = drm_save["SAVEGEN0"]["EMEAN"] * u.keV
    attenuation = tot_trans["det-0"]
    srm = (attenuation.reshape(-1, 1) * drm * energies_out_width) / 4  # avg grid transmission

    res = SimpleNamespace(
        drm=drm,
        srm=srm,
        attenuation=attenuation,
        energies_in=energies_in,
        energies_in_width=energies_in_width,
        energies_in_mean=energies_in_mean,
        energies_out=energies_out,
        energies_out_width=energies_out_width,
        energies_out_mean=energies_out_mean,
        area=1 * u.cm,
    )
    return res


def get_pixel_srm():
    pass


def get_sci_channels(date):
    r"""
    Get the science energy channels for given date.

    Parameters
    ----------
    date : `datetime.datetime`
        Date to lookup science energy channels.

    Returns
    -------
    `astropy.table.QTable`
        Science Energy Channels
    """
    global SCI_INDEX, SCI_CHANNELS

    # Cache index
    if SCI_INDEX is None:
        root = Path(__file__).parent.parent
        sci_chan_index_file = Path(root, *["config", "data", "detector", "science_echan_index.csv"])
        sci_chan_index = read_energy_channel_index(sci_chan_index_file)
        SCI_INDEX = sci_chan_index

    sci_info = SCI_INDEX.at(date)
    if len(sci_info) == 0:
        raise ValueError(f"No Science Energy Channel file found for date {date}")
    elif len(sci_info) > 1:
        raise ValueError(f"Multiple Science Energy Channel file for date {date}")
    start_date, end_date, sci_echan_file = list(sci_info)[0]

    # Cache sci channels
    if sci_echan_file.name in SCI_CHANNELS:
        sci_echan_table = SCI_CHANNELS[sci_echan_file.name]
    else:
        sci_echan_table = read_sci_energy_channels(sci_echan_file)
        SCI_CHANNELS[sci_echan_file.name] = sci_echan_table

    return sci_echan_table
