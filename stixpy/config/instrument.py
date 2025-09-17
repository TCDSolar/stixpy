from types import SimpleNamespace

import astropy.units as u
import numpy as np

from stixpy.io.readers import read_det_adc_mapping, read_pixel_params, read_subc_params

__ALL__ = (["STIX_INSTRUMENT", "_get_uv_points_data"],)

_SUBCOL_PARAMS = read_subc_params()


@u.quantity_input
def _get_uv_points_data(d_det: u.Quantity[u.mm] = 47.78 * u.mm, d_sep: u.Quantity[u.mm] = 545.30 * u.mm):
    r"""
    Return the STIX (u,v) points coordinates defined in [1], ordered with respect to the detector index.

    Parameters
    ----------
    d_det: `u.Quantity[u.mm]` optional
        Distance between the rear grid and the detector plane (in mm). Default, 47.78 * u.mm

    d_sep: `u.Quantity[u.mm]` optional
        Distance between the front and the rear grid (in mm). Default, 545.30 * u.mm

    Returns
    -------
    A dictionary containing sub-collimator indices, sub-collimator labels and coordinates of the STIX (u,v) points (defined in arcsec :sup:`-1`)

    References
    ----------
    [1] Massa et al., 2023, The STIX Imaging Concept, Solar Physics, 298,
        https://doi.org/10.1007/s11207-023-02205-7

    """

    subc = _SUBCOL_PARAMS
    imaging_ind = np.where((subc["Grid Label"] != "cfl") & (subc["Grid Label"] != "bkg"))

    # filter out background monitor and flare locator
    subc_imaging = subc[imaging_ind]

    # assign detector numbers to visibility index of subcollimator (isc)
    isc = subc_imaging["Det #"]

    # assign the stix sc label for convenience
    label = subc_imaging["Grid Label"]

    # save phase orientation of the grids to the visibility
    phase_sense = subc_imaging["Phase Sense"]

    # see Equation (9) in [1]
    front_unit_vector_comp = (((d_det + d_sep) / subc_imaging["Front Pitch"]) / u.rad).to(1 / u.arcsec)
    rear_unit_vector_comp = ((d_det / subc_imaging["Rear Pitch"]) / u.rad).to(1 / u.arcsec)

    uu = (
        np.cos(subc_imaging["Front Orient"].to("deg")) * front_unit_vector_comp
        - np.cos(subc_imaging["Rear Orient"].to("deg")) * rear_unit_vector_comp
    )
    vv = (
        np.sin(subc_imaging["Front Orient"].to("deg")) * front_unit_vector_comp
        - np.sin(subc_imaging["Rear Orient"].to("deg")) * rear_unit_vector_comp
    )

    uu = -uu * phase_sense
    vv = -vv * phase_sense

    uv_data = {
        "isc": isc,  # sub-collimator indices
        "label": label,  # sub-collimator labels
        "u": uu,
        "v": vv,
    }

    return uv_data


STIX_INSTRUMENT = SimpleNamespace(
    __doc__="""This namespace contains the instrument name, visibility information, sub-collimator ADC mapping, sub-collimator parameters, and pixel configuration.

        The visibility information is obtained from the _get_uv_points_data function,
        the sub-collimator ADC mapping is read from the detector ADC mapping file,
        the sub-collimator parameters are read from the sub-collimator parameters file,
        and the pixel configuration is read from the pixel parameters file.
        The namespace is used to encapsulate all the configuration data related to the STIX instrument.
        This allows for easy access and management of the instrument's configuration data.""",
    name="STIX",
    vis_info=_get_uv_points_data(),
    subcol_adc_mapping=np.array(read_det_adc_mapping()["Adc #"]),
    subcol_params=_SUBCOL_PARAMS,
    pixel_config=read_pixel_params(),
)
