import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates  # registers Helioprojective, HeliographicStonyhurst, etc.
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
import warnings


__all__ = [
    "flare_spacecraft_angle",
]

def flare_spacecraft_angle(
    spacecraft_coord: SkyCoord,
    flare_hpc: SkyCoord,
) -> u.Quantity:
    """
    Calculate the angle between the flare surface normal and the
    spacecraft direction as seen from the flare.
 
    The flare surface normal is the outward radial unit vector from the
    Sun's centre through the flare location (i.e. the local vertical).
 
    Parameters
    ----------
    spacecraft_coord : SkyCoord
        Position of the spacecraft in *any* frame that Astropy/SunPy can
        transform.  Typically HeliographicStonyhurst (HGS) with a distance.
    flare_hpc : SkyCoord
        Flare position in Helioprojective coordinates (Tx, Ty).  Must be
        observed from *some* observer — usually Earth or a spacecraft.
        The observer frame's `obstime` is used for the coordinate transform.
 
    Returns
    -------
    angle : ~astropy.units.Quantity
        Angle in degrees between the flare surface normal and the
        line-of-sight to the spacecraft.  0° means the spacecraft is
        directly above the flare; 90° means it is on the limb as seen
        from the flare.
    """

    obstime = flare_hpc.obstime
 
    hgs_frame = HeliographicStonyhurst(obstime=obstime)
 
    flare_hgs = flare_hpc.transform_to(hgs_frame)

    if np.isnan(flare_hgs.lat.value):
        warnings.warn("Flare location is off the limb so no angle can be calculated.")
        return np.nan

    else:

        lon_rad = flare_hgs.lon.to(u.rad).value
        lat_rad = flare_hgs.lat.to(u.rad).value
    
        flare_cart = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ])

        sc_hgs = spacecraft_coord.transform_to(hgs_frame)

        sc_cart = sc_hgs.cartesian.xyz.value  
    
        sc_unit = sc_hgs.cartesian.xyz.unit
        flare_pos_m = flare_cart *  (1.0 * u.R_sun).to(sc_unit).value        
                                                      
        cos_theta = (
            np.dot(flare_cart, (sc_cart - flare_pos_m))
            / (np.linalg.norm(flare_cart) * np.linalg.norm((sc_cart - flare_pos_m)))
        )

        angle = np.degrees(np.arccos(cos_theta)) * u.deg
    
        return angle