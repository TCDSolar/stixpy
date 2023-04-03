"""
Grid Calibration
"""
from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.table import Table


def get_grid_internal_shadowing(xy_flare):
    r"""
    Return the grid transmission for the 32 sub-collimators



    Read the grid tables and calculates grid transmission for 32 sub-collimators. For finest grids
    only transmission without correction for internal shadowing

    Parameters
    ----------
    xy_flare

    Returns
    -------

    """
    column_names = ['sc', 'p', 'o', 'phase', 'slit', 'grad', 'rms', 'thick', 'bwidth', 'bpitch']

    root = Path(__file__).parent.parent
    grid_info = Path(root, *['config', 'data', 'grid'])
    front = Table.read(grid_info / 'grid_param_front.txt', format='ascii', names=column_names)
    rear = Table.read(grid_info / 'grid_param_rear.txt', format='ascii', names=column_names)

    xy_flare_stix = np.array([xy_flare[1].value, -xy_flare[0].value])*u.arcsec
    # print,xy_flare
    # print,xy_flare_stix

    # radial
    r_flare = np.sqrt((xy_flare_stix**2).sum())
    # angle from +y direction (definition used by Matej in grid table)
    # o_flare=abs(asin(xy_flare_stix[1]/r_flare)/!pi*180)
    # o_flare=90-abs(asin(xy_flare_stix[1]/r_flare)/!pi*180)
    # new October 26

    # Paolo Massa (September 2021): corrected the definition of 'o_flare'

    if (xy_flare_stix[0] >= 0) and (xy_flare_stix[1] >= 0):
        o_flare = np.arccos(xy_flare_stix[1]/r_flare)
    if (xy_flare_stix[0] >= 0) and (xy_flare_stix[1] <= 0):
        o_flare = 90*u.deg+np.arcsin(np.abs(xy_flare_stix[1])/r_flare)
    if (xy_flare_stix[0] <= 0) and (xy_flare_stix[1] <= 0):
        o_flare = np.arccos(np.abs(xy_flare_stix[1])/r_flare)
    if (xy_flare_stix[0] <= 0) and (xy_flare_stix[1] >= 0):
        o_flare = 90*u.deg + np.arcsin(xy_flare_stix[1]/r_flare)

    # if (xy_flare_stix[0] >= 0) and (xy_flare_stix[1] >= 0):
    #     o_flare = np.arccos(xy_flare_stix[1] / r_flare)
    # if (xy_flare_stix[0] >= 0) and (xy_flare_stix[1] <= 0):
    #     o_flare = np.pi*u.rad/2 - np.arcsin(abs(xy_flare_stix[1]) / r_flare)
    # if (xy_flare_stix[0] <= 0) and (xy_flare_stix[1] <= 0):
    #     o_flare = -np.pi*u.rad/2 + np.arcsin(xy_flare_stix[1] / r_flare)
    # if (xy_flare_stix[0] <= 0) and (xy_flare_stix[1] >= 0):
    #     o_flare = -np.arccos(xy_flare_stix[1] / r_flare)

    # Paolo Massa (September 2021): corrected the definition of 'rel_f' and 'rel_r'. Indeed, the
    # grids in this case are considered looking towards the Sun. A minus sign is added to
    # 'front['o']' and 'rear['o']' to keep into account the different definition of the orientation
    # angle used in Matej's grid table

    #  #orientations of grids relative to flare
    #  rel_f=abs(front['o']-o_flare)
    #  rel_r=abs(rear['o']-o_flare)
    rel_f = np.abs((180 - front['o'])*u.deg - o_flare)
    rel_r = np.abs((180 - rear['o'])*u.deg - o_flare)

    # correction only applies to component normal to the grid orientation
    cor_f = np.sin(rel_f)
    cor_r = np.sin(rel_r)

    # nominal thickness with glue
    nom_h = 0.42  # * u.mm
    # maximal internal grid shadowing when see for grids with orientation of 90 degrees from
    # center to flare line
    max_c = np.tan(r_flare)*nom_h
    # the shadow in mm for each of the entries# this value will be subtracted from the nominal
    # slit width (see below)
    shadow_r = cor_r * max_c
    shadow_f = cor_f * max_c

    # for bridges correction is largest for grids with orientation parallel to the center-flare
    # line
    np.cos(rel_f)
    np.cos(rel_r)
    # the shadow to be added to bridge width (see below)
    cor_r * max_c
    cor_f * max_c

    gtrans32f = np.zeros(32) - 2
    gtrans32r = np.zeros(32) - 2
    btrans32f = np.zeros(32) + 1
    btrans32r = np.zeros(32) + 1

    # for the case the flare location is at [0,0], the code above gives NAN, but correction for
    # internal shadowing should be zero
    if xy_flare[0].to_value('arcsec') == 0 and xy_flare[1].to_value('arcsec') == 0:
        shadow_f = np.zeros(39)
        shadow_r = np.zeros(39)

    for i in range(32):
        this_list = np.where(rear['sc'] == i + 1)[0]
        # print,i,front['sc'][this_list]
        # single layer: transmission = slit/pitch
        if (this_list.size == 1) and (this_list[0] != -1):
            # gtrans32f[i]=front['slit'](this_list[0])/front['p'](this_list[0])
            gtrans32f[i] = ((front['slit'][this_list[0]] - shadow_f[this_list[0]])
                            / front['p'][this_list[0]])
            # gtrans32r[i]=rear['slit'](this_list[0])/rear['p'](this_list[0])
            gtrans32r[i] = ((rear['slit'][this_list[0]] - shadow_r[this_list[0]])
                            / rear['p'][this_list[0]])
            if front['bpitch'][this_list[0]] != 0.0:
                btrans32f[i] = 1 - front['bwidth'][this_list[0]] / front['bpitch'][this_list[0]]
                btrans32r[i] = 1 - rear['bwidth'][this_list[0]] / rear['bpitch'][this_list[0]]

        # multi layer grids:
        if this_list.size >= 2:
            # slats in each layer (covered part)
            this_slat_f_each = front['p'][this_list] - front['slit'][this_list]
            # sum of all slats gives lengths of total covered part
            this_slat_f_total = this_slat_f_each.sum()
            # transmission is 1 minus covered/pitch
            gtrans32f[i] = 1. - this_slat_f_total / np.average(front['p'][this_list])
            # slats in each layer (covered part)
            this_slat_r_each = rear['p'][this_list] - rear['slit'][this_list]
            # sum of all slats gives lengths of total covered part
            this_slat_r_total = this_slat_r_each.sum()
            # transmission is 1 minus covered/pitch
            gtrans32r[i] = 1. - this_slat_r_total / np.average(rear['p'][this_list])
            # bridge: only one value in the table
            btrans32f[i] = 1 - front['bwidth'][this_list[0]] / front['bpitch'][this_list[0]]
            btrans32r[i] = 1 - rear['bwidth'][this_list[0]] / rear['bpitch'][this_list[0]]

    # no bridge measurement for 5 front use 0.05, the nominal value
    # (in file it is set to 0.5 to mark the missing value)
    btrans32f[4] = 1. - 0.05 / front['bpitch'][4]
    # could apply averaged deviation from nominal instead?

    gtrans32 = gtrans32f * gtrans32r * btrans32f * btrans32r
    # gtrans32_no_b = gtrans32f * gtrans32r

    # set CFL and BKG to zero for now
    gtrans32[8:10] = 0
    return gtrans32
