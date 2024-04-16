from pathlib import Path
from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from dateutil.parser import parse
from intervaltree import IntervalTree


def read_energy_channel_index(echan_index_file):
    r"""
    Read science energy channel index file

    Parameters
    ----------
    echan_index_file: `str` or `pathlib.Path`

    Returns
    -------
    Science Energy Channel lookup
    """
    echans = Table.read(echan_index_file)
    echan_it = IntervalTree()
    for i, start, end, file in echans.iterrows():
        date_start = parse(start)
        date_end = parse(end) if end != "none" else datetime(2100, 1, 1)
        echan_it.addi(date_start, date_end, echan_index_file.parent / file)
    return echan_it


def read_sci_energy_channels(path):
    """
    Read science energy channel definitions.

    Parameters
    ----------
    path : `pathlib.Path`
        path to the config file

    Returns
    -------
    `astropy.table.QTable`
        The science energy channels
    """
    converters = {
        "Channel Number": int,
        "Channel Edge": float,
        "Energy Edge": float,
        "Elower": float,
        "Eupper": float,
        "BinWidth": float,
        "dE/E": float,
        "QL channel": int,
    }

    # tuples of (<match string>, '0')
    bad_data = (("max ADC", "0"), ("maxADC", "0"), ("n/a", "0"), ("", "0"))
    sci_chans = QTable.read(
        path, delimiter=",", data_start=24, header_start=21, converters=converters, fill_values=bad_data
    )
    # set units can't use list comp
    for col in ["Elower", "Eupper", "BinWidth"]:
        sci_chans[col].unit = u.keV
    return sci_chans


def read_subc_params(path=None):
    """Read the configuration of the sub-collimator from the configuration file.

    Parameters
    ----------
    path : `pathlib.Path`
        path to the config file

    Returns
    -------
    `Table`
        params for all 32 sub-collimators
    """
    if path is None:
        path = Path(__file__).parent.parent / "config" / "data" / "detector" / "stx_subc_params.csv"
    return Table.read(path, format="ascii.ecsv")


def read_elut_index(elut_index):
    elut = Table.read(elut_index)
    elut_it = IntervalTree()
    for i, start, end, file in elut.iterrows():
        date_start = parse(start)
        date_end = parse(end) if end != "none" else datetime(2100, 1, 1)
        elut_it.addi(date_start, date_end, elut_index.parent / file)
    return elut_it


def read_elut(elut_file):
    root = Path(__file__).parent.parent
    sci_channels = read_sci_energy_channels(
        Path(root, *["config", "data", "detector", "ScienceEnergyChannels_1000.csv"])
    )
    elut_table = Table.read(elut_file, header_start=2)

    elut = type("ELUT", (object,), dict())
    elut.file = elut_file.name
    elut.offset = elut_table["Offset"].reshape(32, 12)
    elut.gain = elut_table["Gain keV/ADC"].reshape(32, 12)
    elut.pixel = elut_table["Pixel"].reshape(32, 12)
    elut.detector = elut_table["Detector"].reshape(32, 12)
    adc = np.vstack(list(elut_table.columns[4:].values())).reshape(31, 32, 12)
    adc = np.moveaxis(adc, 0, 2)
    elut.adc = adc
    elut.e_actual = (elut.adc - elut.offset[..., None]) * elut.gain[..., None]
    elut.e_width_actual = elut.e_actual[..., 1:] - elut.e_actual[..., :-1]

    elut.e = sci_channels["Elower"]
    elut.e_width = sci_channels["Eupper"] - sci_channels["Elower"]

    return elut
