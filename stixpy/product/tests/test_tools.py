from types import SimpleNamespace

import astropy.nddata
import numpy as np
import pytest

from .. import tools

@pytest.fixture
def cube():
    data = np.ones((2, 5))
    mask = np.array([[False, False, False,  True, False],
                     [ True,  True,  True, False,  True]], dtype=bool)
    uncertainty = astropy.nddata.StdDevUncertainty(data*0.1)
    return SimpleNamespace(data=data, mask=mask, uncertainty=uncertainty, shape=data.shape)

def test_rebin_irregular(cube):
    # Run function.
    axes_idx_edges = [0, 2], [0, 1, 3, 5]
    data, mask, uncertainty = tools.rebin_irregular(cube, axes_idx_edges, operation=np.sum,
                                                    operation_ignores_mask=False,
                                                    handle_mask=np.all,
                                                    propagate_uncertainties=True)
    # Define expected outputs.
    expected_data = np.array([[1., 2., 2.]])
    expected_mask = np.zeros(expected_data.shape, dtype=bool)
    expected_uncertainty = astropy.nddata.StdDevUncertainty([[0.1,
                                                              0.14142135623730953,
                                                              0.14142135623730953]])
    # Compare outputs with expected.
    np.testing.assert_allclose(data, expected_data)
    assert (mask == expected_mask).all()
    assert isinstance(uncertainty, type(expected_uncertainty))
    np.testing.assert_allclose(uncertainty.array, expected_uncertainty.array)
