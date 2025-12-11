"""
Grid Calibration
"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
import xraydb

__all__ = ["get_grid_transmission", "_calculate_grid_transmission"]

from stixpy.coordinates.frames import STIXImaging


def get_grid_transmission(ph_energy, flare_location: STIXImaging):
    r"""
    Return the grid transmission for the 32 sub-collimators corrected for internal shadowing.

    Read the grid tables and calculate the grid transmission for 32 sub-collimators.
    For the finest grids only transmission without correction for internal shadowing

    Parameters
    ----------
    flare_location :
        Location of the flare
    """
    column_names = ["sc", "p", "o", "phase", "slit", "grad", "rms", "thick", "bwidth", "bpitch"]

    root = Path(__file__).parent.parent
    grid_info = Path(root, *["config", "data", "grid"])
    front = Table.read(grid_info / "grid_param_front.txt", format="ascii", names=column_names)
    rear = Table.read(grid_info / "grid_param_rear.txt", format="ascii", names=column_names)

    # ;; Orientation of the slits of the grid as seen from the detector side
    grid_orient_front_all = front['o']
    grid_orient_rear_all = rear['o']
    
    pitch_front_all = front['p']
    pitch_rear_all = rear['p']
    
    thickness_front_all = front['thick']
    thickness_rear_all = rear['thick']

    sc = front['sc']

    
    # fpath = loc_file( 'CFL_subcoll_transmission.txt', path = getenv('STX_GRID') )
    column_names_cfl = ["subc_n", "subc_label", "intercept", "slope[1/deg]"]

    subcol_transmission = Table.read(grid_info / "CFL_subcoll_transmission.txt", format="ascii", names=column_names_cfl)
    
    subc_n_all = subcol_transmission['subc_n']
    subc_label = subcol_transmission['subc_label']
    intercept_all = subcol_transmission['intercept']
    slope_all = subcol_transmission['slope[1/deg]']


    muvals = xraydb.material_mu('W', ph_energy * 1e3, density=19.30, kind='total') / 10 # in units of mm^-1
    L = 1 / muvals
    print('L  = ', L)
    # trans = np.exp(-0.4 / L)
    subc_transm=L

    det_indices_top24 =  np.array([0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 19, 
                                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    # det_all = np.arange(0,32,1)

    # det_indices_top24 =  [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 19, 
    #                                  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    idx_full = det_indices_top24
    idx = [i for i,x in enumerate(sc-1) if x in  det_indices_top24]
    print('idx = ', idx)
    print('lenidx = ',len(idx))
    # idx = det_all

    # for i,idx in enumerate(idx_full):
    print(grid_orient_front_all)

    grid_orient_front = grid_orient_front_all[idx]
    pitch_front = pitch_front_all[idx]
    thickness_front = thickness_front_all[idx]
    
    # ;;------ Rear grid
    grid_orient_rear = grid_orient_rear_all[idx]
    pitch_rear = pitch_rear_all[idx]
    thickness_rear = thickness_rear_all[idx]

    grid_orient_avg = (grid_orient_front + grid_orient_rear) / 2

    flare_loc_deg = flare_location / 3600  #. ;; Convert coordinates to deg
    theta = flare_loc_deg[0] * np.cos(np.deg2rad(grid_orient_avg)) + flare_loc_deg[1] * np.sin(np.deg2rad(grid_orient_avg)) 
    print('THETA = ', theta)
    
# ;;------ Subcollimator tranmsission at low energies
# idx = np.where(subc_n_all eq (subc_n+1))

    intercept = intercept_all[det_indices_top24]
    slope = slope_all[det_indices_top24]
    subc_transm_low_e = intercept + slope * theta

    # ;;------ Transmission of front and rear grid
    slit_to_pitch = np.sqrt(subc_transm_low_e)

    slit_front = slit_to_pitch*pitch_front
    slit_rear = slit_to_pitch*pitch_rear
    
    transm_front = stx_grid_transmission(pitch_front, slit_front, thickness_front, L)
    transm_rear = stx_grid_transmission(pitch_rear, slit_rear, thickness_rear, L)
        
        # subc_transm.append(transm_front * transm_rear)

    subc_transm = transm_front * transm_rear

    # transmission_front = _calculate_grid_transmission(front, flare_location)
    # transmission_rear = _calculate_grid_transmission(rear, flare_location)
    # total_transmission = transmission_front * transmission_rear

    # The finest grids are made from multiple layers for the moment remove these and set 1
    # final_transmission = np.ones(32)
    # sc = front["sc"]
    # finest_scs = [11, 12, 13, 17, 18, 19]  # 1a, 2a, 1b, 2c, 1c, 2b
    # idx = np.argwhere(np.isin(sc, finest_scs, invert=True)).ravel()
    # final_transmission[sc[idx] - 1] = total_transmission[idx]
   
    print('subc = ',subc_transm)
    return subc_transm


def stx_grid_transmission(pitch, slit, thickness, L):

    ds = 5e-3
    dh = 5e-2

    # n_energies = np.shape(L)[0]
    # n_subc = np.shape(pitch)

    slit_rep = slit.reshape(1, len(slit))
    pitch_rep = pitch.reshape(1, len(pitch))
    H_rep = thickness.reshape(1, len(thickness))
    L_rep = L.reshape(len(L), 1)

    print('slit = ',slit[0])
    print('pitch = ',pitch[0])
    print('h_rep = ',thickness[0])

    # slit_rep = np.tile(slit, (n_energies, 1))
    # pitch_rep = np.tile(pitch, (n_energies, 1))
    # H_rep = np.tile(thickness, (n_energies, 1))
    # L_rep = np.tile(L, (n_subc, 1)).T
  
    print(np.shape(slit_rep))
    print(np.shape(pitch_rep))
    print(np.shape(H_rep))
    print(np.shape(L_rep))

    # ;; Transmission for a wedge shape model for grid imperfections
    g0 = slit_rep / pitch_rep + (pitch_rep - slit_rep) / pitch_rep * np.exp( - H_rep / L_rep )
    ttt = L_rep / dh * ( 1. - np.exp(- dh / L_rep ) )
    g1 = 2. * ds / pitch_rep * (ttt - np.exp( - H_rep / L_rep ))
    
    print('g0 = ',g0[:,0])
    print('g1 = ',g1[:,0])

    g_transmission = g0 + g1

    print('transmission = ',g_transmission)
  
    return g_transmission


def _calculate_grid_transmission(grid_params, flare_location):
    r"""
    Calculate grid transmission accounting for internal shadowing.

    Parameters
    ----------
    grid_params
        Grid parameter tables
    flare_location
        Position of flare in stix imaging frame

    Returns
    -------
    Transmission through all grids defined by the grid parameters
    """
    orient = grid_params["o"] * u.deg  # As viewed from the detector side (data recorded from front)
    pitch = grid_params["p"]
    slit = grid_params["slit"]
    thick = grid_params["thick"]
    flare_dist = np.abs(flare_location.Tx * np.cos(orient)) + flare_location.Ty * np.sin(orient)
    shadow_width = thick * np.tan(flare_dist)
    transmission = (slit - shadow_width) / pitch
    return transmission
