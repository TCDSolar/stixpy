from pathlib import Path
from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.table.table import Table
from roentgen.absorption.material import Material, Stack

from stixpy.io.readers import read_sci_energy_channels

__all__ = ["Transmission"]

MIL_SI = 0.0254 * u.mm

# TODO move to configuration files
COMPONENTS = OrderedDict(
    [
        ("front_window", [("solarblack", 0.005 * u.mm), ("be_alloy", 2 * u.mm)]),
        ("rear_window", [("be_alloy", 1 * u.mm)]),
        ("grid_covers", [("kapton", 4 * 2 * MIL_SI)]),
        ("dem", [("kapton", 2 * 3 * MIL_SI)]),
        ("attenuator", [("al_alloy", 0.6 * u.mm)]),
        (
            "mli",
            [
                ("al", 1000 * u.angstrom),
                ("kapton", 3 * MIL_SI),
                ("al", 40 * 1000 * u.angstrom),
                ("mylar", 20 * 0.25 * MIL_SI),
                ("pet", 21 * 0.005 * u.mm),
                ("kapton", 3 * MIL_SI),
                ("al", 1000 * u.angstrom),
            ],
        ),
        ("calibration_foil", [("al", 4 * 1000 * u.angstrom), ("kapton", 4 * 2 * MIL_SI)]),
        ("dead_layer", [("te_o2", 392 * u.nm)]),
    ]
)

# For attenuator see https://www.metallservice.ch/msm/msm-home/services/infoservice/
# produktinfos-datenbl%C3%A4tter/aluminium-platten/en_aw-7075_1-3.pdf

MATERIALS = OrderedDict(
    [
        ("al", ({"Al": 1.0}, 2.7 * u.g / u.cm**3)),
        (
            "al_alloy",
            (
                {
                    "Al": 0.89345,
                    "Si": 0.002,
                    "Fe": 0.0025,
                    "Cu": 0.016,
                    "Mn": 0.0015,
                    "Mg": 0.025,
                    "Cr": 0.0023,
                    "Ni": 0.00025,
                    "Zn": 0.056,
                    "Ti": 0.001,
                },
                2.8 * u.g / u.cm**3,
            ),
        ),
        (
            "be_alloy",
            (
                {"Al": 0.0005, "Be": 0.9974, "C": 0.00075, "Fe": 0.00065, "Mg": 0.0004, "Si": 0.0003},
                1.84 * u.g / u.cm**3,
            ),
        ),
        ("kapton", ({"H": 0.026362, "C": 0.691133, "N": 0.073270, "O": 0.209235}, 1.43 * u.g / u.cm**3)),
        ("mylar", ({"H": 0.041959, "C": 0.625017, "O": 0.333025}, 1.38 * u.g / u.cm**3)),
        ("pet", ({"H": 0.041960, "C": 0.625016, "O": 0.333024}, 1.370 * u.g / u.cm**3)),
        ("solarblack_oxygen", ({"H": 0.002, "O": 0.415, "Ca": 0.396, "P": 0.187}, 3.2 * u.g / u.cm**3)),
        ("solarblack_carbon", ({"C": 0.301, "Ca": 0.503, "P": 0.195}, 3.2 * u.g / u.cm**3)),
        ("te_o2", ({"Te": 0.7995088158691722, "O": 0.20049124678825841}, 5.670 * u.g / u.cm**3)),
    ]
)

# TODO get file from config
ENERGY_CHANNELS = read_sci_energy_channels(
    Path(__file__).parent.parent / "config" / "data" / "detector" / "ScienceEnergyChannels_1000.csv"
)


class Transmission:
    """
    Calculate the energy dependent transmission of X-ray through the instrument
    """

    def __init__(self, solarblack="solarblack_carbon"):
        """
        Create a new instance of the transmission with the given solar black composition.

        Parameters
        ----------

        solarblack : `str` optional
            The SolarBlack composition to use.
        """
        if solarblack not in ["solarblack_oxygen", "solarblack_carbon"]:
            raise ValueError("solarblack must be either 'solarblack_oxygen' or 'solarblack_carbon'.")

        self.solarblack = solarblack
        self.materials = MATERIALS
        self.components = COMPONENTS
        self.components = dict()
        self.energies = ENERGY_CHANNELS[1:32]["Elower"]

        for name, layers in COMPONENTS.items():
            parts = []
            for material, thickness in layers:
                if material == "solarblack":
                    material = self.solarblack
                mass_frac, den = MATERIALS[material]
                mat = Material(mass_frac, thickness=thickness, density=den)
                mat.name = name
                parts.append(mat)
            self.components[name] = Stack(parts)

    def get_transmission(self, energies=None, attenuator=False):
        """
        Get the transmission for each detector at the center of the given energy bins.

        If energies are not supplied will evaluate at standard science energy channels

        Parameters
        ----------
        energies : `astropy.units.Quantity`, optional
            The energies to evaluate the transmission
        attenuator : `bool`, optional
            True for attenuator in X-ray path, False for attenuator not in X-ray path

        Returns
        -------
        `astropy.table.Table`
            Table containing the transmission values for each energy and detector
        """
        base_comps = [
            self.components[name]
            for name in ["front_window", "rear_window", "dem", "mli", "calibration_foil", "dead_layer"]
        ]

        if energies is None:
            energies = self.energies

        if attenuator:
            base_comps.append(self.components["attenuator"])

        base = Stack(base_comps)
        base_trans = base.transmission(energies)

        fine = Stack(base_comps + [self.components["grid_covers"]])
        fine_trans = fine.transmission(energies)

        # TODO need to move to configuration db
        fine_grids = np.array([11, 13, 18, 12, 19, 17]) - 1
        transmission = Table()
        # transmission['sci_channel'] = range(1, 31)
        transmission["energies"] = energies
        for i in range(32):
            name = f"det-{i}"
            if np.isin(i, fine_grids):
                transmission[name] = fine_trans
            else:
                transmission[name] = base_trans
        transmission["attenuator"] = self.components["attenuator"].transmission(energies)
        return transmission

    def get_transmission_by_component(self):
        """
        Get the contributions to the total transmission by broken down by component.

        Returns
        -------
        `dict`
            Entries are Compounds for each component
        """
        return self.components

    def get_transmission_by_material(self):
        """
        Get the contribution to the transmission by total thickness for each material.

        Layers of the same materials are combined to return one instance with the total thickness.

        Returns
        -------
        `dict`
            Entries are materials with the total thickness for that material.
        """
        material_thickness = dict()
        for name, layers in COMPONENTS.items():
            for material_name, thickness in layers:
                if material_name == "solarblack":
                    material_name = self.solarblack
                if material_name in material_thickness.keys():
                    material_thickness[material_name] += thickness.to("mm")
                else:
                    material_thickness[material_name] = thickness.to("mm")
        res = {}
        for name, thickness in material_thickness.items():
            frac_mass, density = self.materials[name]
            mat = Material(frac_mass, density=density, thickness=thickness)
            mat.name = name
            res[name] = mat

        return res


def generate_transmission_tables():
    from datetime import datetime

    cur_date = datetime.now().strftime("%Y%m%d")
    datetime.now().strftime("%Y%m%d")
    trans = Transmission()

    energies = np.hstack([np.arange(2, 20, 0.01), np.arange(20, 160, 0.1)]) * u.keV

    norm_sci_energies = trans.get_transmission(attenuator=True)
    norm_sci_energies.write(f"stix_transmission_sci_energies_{cur_date}.csv")
    norm_high_res = trans.get_transmission(energies=energies, attenuator=True)
    norm_high_res.write(f"stix_transmission_highres_{cur_date}.csv")

    comps = trans.get_transmission_by_component()

    comps_sci_energies = Table(
        [c.transmission(trans.energies) for c in comps.values()], names=[k for k in comps.keys()]
    )
    comps_sci_energies["energy"] = trans.energies
    comps_sci_energies.write(f"stix_transmission_by_component_sci_energies_{cur_date}.csv")

    comps_highres = Table([c.transmission(energies) for c in comps.values()], names=[k for k in comps.keys()])
    comps_highres["energy"] = energies
    comps_highres.write(f"stix_transmission_by_component_highres_{cur_date}.csv")
