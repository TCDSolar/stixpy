
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u

from stixpy.calibration.transmission import Transmission


def test_transmission_create_material():
    hydrogen = Transmission.create_material(name='hydrogen', fractional_masses={'H': 1.0},
                                            thickness=1*u.cm, density=8.375e-05*u.g/u.cm**3)

    assert hydrogen.name == 'hydrogen'
    assert hydrogen.thickness == 1.0 * u.cm
    assert hydrogen.density == 8.375e-05*u.g/u.cm**3
    # taken from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z01.html
    ref_attenuation_coefficient = [7.217E+00, 2.944E-01, 3.254E-02] * u.cm**2/u.g
    attenuation_coefficient = hydrogen.mass_attenuation_coefficient.func([1e-3, 1e-1, 1e+1] * u.MeV)
    assert np.allclose(attenuation_coefficient, ref_attenuation_coefficient)
    transmission = np.exp(-hydrogen.density*hydrogen.thickness*attenuation_coefficient)
    assert np.allclose(transmission, hydrogen.transmission([1e-3, 1e-1, 1e+1] * u.MeV))


def test_transmission_get_transmission():
    # test_materials = OrderedDict([('test1', [('al', 1 * u.mm), ('be', 2*u.mm)]),
    #                           ('test2', [('be', 1*u.mm)])])
    #
    # test_componenets = OrderedDict([('al', ({'Al': 1.0}, 2.7 * u.g/u.cm**3)),
    #                           ('be', ({'Be': 1.0}, 1.85 * u.g/u.cm**3))])

    trans = Transmission()
    res = trans.get_transmission()
    assert len(res.columns) == 34
    assert len(res) == 31
    assert res['energies'][0] == 4.0
    assert res['energies'][-1] == 150.0
    assert_allclose(res['det-0'][0], 1.2304064853433878e-05)
    assert_allclose(res['det-0'][-1], 0.9223594048282234)
    assert_allclose(res['det-10'][0], 2.866929141727759e-06)
    assert_allclose(res['det-10'][-1], 0.9186555303212736)
