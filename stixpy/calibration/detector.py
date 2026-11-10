from pathlib import Path

import numpy as np

from stixpy.io.readers import read_energy_channel_index, read_sci_energy_channels

# __all__ = ["get_srm", "get_pixel_srm", "get_sci_channels"]
__all__ = ["get_sci_channels", "tailing_matrix"]

SCI_INDEX = None
SCI_CHANNELS = {}


# def get_srm():
#     r"""
#     Return the spectromber response matrix (SRM) by combing the attenuation with the detetor respoonse matrix (DRM)

#     Returns
#     -------

#     """
#     # drm_save = read_genx("/Users/shane/Projects/STIX/git/stix_drm_20220713.genx")
#     drm_save = np.load("/home/jmitchell/software/stixpy-dev/stixpy/config/data/detector/")

#     drm = drm_save["SAVEGEN0"]["SMATRIX"] * u.count / u.keV / u.photon
#     energies_in = drm_save["SAVEGEN0"]["EDGES_IN"] * u.keV
#     energies_in_width = np.diff(energies_in)
#     energies_in_mean = energies_in[:-1] + energies_in_width / 2
#     trans = Transmission()
#     tot_trans = trans.get_transmission(energies=energies_in_mean)
#     energies_out = drm_save["SAVEGEN0"]["EDGES_OUT"] * u.keV
#     energies_out_width = drm_save["SAVEGEN0"]["EWIDTH"] * u.keV
#     energies_out_mean = drm_save["SAVEGEN0"]["EMEAN"] * u.keV
#     attenuation = tot_trans["det-0"]
#     srm = (attenuation.reshape(-1, 1) * drm * energies_out_width) / 4  # avg grid transmission

#     res = SimpleNamespace(
#         drm=drm,
#         srm=srm,
#         attenuation=attenuation,
#         energies_in=energies_in,
#         energies_in_width=energies_in_width,
#         energies_in_mean=energies_in_mean,
#         energies_out=energies_out,
#         energies_out_width=energies_out_width,
#         energies_out_mean=energies_out_mean,
#         area=1 * u.cm,
#     )
#     return res


# def get_pixel_srm():
#     pass


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


def tailing_matrix(
    ph_edges,
    xsec_energy,
    xsec,
    depth=0.1,
    trap_length_h=0.36e4,
    trap_length_e=24e4,
    damage_layer_depth=4.4e-5,
    r0=0.8,
    n_layers=1000,
):
    """
    Hole-tailing matrix, a port of STIX-GSW ``stx_tailing_matrix.pro`` as ``stx_build_drm`` calls it.

    IDL builds it on the photon bin means and applies it along the photon axis
    (``eloss_mat # tailing_matrix``), so it depends on the photon grid and is rebuilt here for
    each product's grid.

    Parameters
    ----------
    ph_edges : numpy.ndarray
        Photon bin edges in keV (the product's grid).
    xsec_energy, xsec : numpy.ndarray
        CdTe photoelectric + incoherent cross section in 1/cm (``det_xsec`` 'PE' + 'SI') and its
        energies in keV, interpolated log-log.

    Returns
    -------
    numpy.ndarray
        ``T[dest, src]`` over photon bins; apply to a (photon, count) matrix as ``T.T @ drm``.
    """
    energy = 0.5 * (ph_edges[1:] + ph_edges[:-1])  # IDL passes the photon bin means
    nen = energy.size
    tm = np.zeros((nen, nen))  # tm[src, dest], as in IDL

    # detector layers, with the finer damage layer at the front
    d = depth * 1e4
    dl = damage_layer_depth * 1e4
    x = d * np.arange(n_layers) / n_layers
    t = 10 * dl * np.arange(2 * n_layers) / (2 * n_layers)
    x = np.concatenate([t, x[x >= 10 * dl]])
    h = (trap_length_h * (1 - np.exp(-x / trap_length_h)) + trap_length_e * (1 - np.exp(-(d - x) / trap_length_e))) / d
    h = h * (1 - r0 * np.exp(-x / dl))  # charge collection efficiency per layer

    emin = 0.5 * (energy[1:] + energy[:-1])
    stot = np.exp(np.interp(np.log(emin), np.log(xsec_energy), np.log(xsec))) / 1e4  # 1/um
    mx, dx = 0.5 * (x[1:] + x[:-1]), np.diff(x)

    j = np.arange(nen - 1)
    for i in range(x.size - 1):
        f = energy * h[i]
        pslice = np.exp(-stot * mx[i]) * (1 - np.exp(-stot * dx[i])) / (1 - np.exp(-stot * d))
        g0 = np.searchsorted(energy, f[:-1], side="right") - 1  # IDL value_locate
        g1 = np.searchsorted(energy, f[1:], side="right") - 1
        width = f[1:] - f[:-1]

        same = (g0 == g1) & (g0 >= 0)
        tm[j[same], g0[same]] += pslice[same]

        low = (g0 != g1) & (g0 < 0)
        tm[j[low], g1[low]] += np.abs((f[1:][low] - energy[g1[low]]) / width[low]) * pslice[low]

        split = (g0 != g1) & (g0 >= 0)
        tm[j[split], g0[split]] += np.abs((f[:-1][split] - energy[g1[split]]) / width[split]) * pslice[split]
        tm[j[split], g1[split]] += np.abs((f[1:][split] - energy[g1[split]]) / width[split]) * pslice[split]

    return tm.T
