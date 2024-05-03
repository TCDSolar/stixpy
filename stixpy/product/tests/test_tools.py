from types import SimpleNamespace

import numpy as np
import pytest

from .. import tools

@pytest.fixture
def cube():
    data = np.ones((3, 4, 5))
    mask = np.array([[[ True,  True,  True, False, False],
                      [ True, False,  True, False, False],
                      [ True, False, False, False, False],
                      [ True, False,  True,  True,  True]],
                     [[ True,  True,  True,  True,  True],
                      [ True,  True, False, False,  True],
                      [False, False,  True, False, False],
                      [ True,  True, False, False,  True]],
                     [[False, False,  True,  True, False],
                      [False,  True, False, False,  True],
                      [ True, False,  True,  True,  True],
                      [ True,  True,  True, False, False]]])
    return SimpleNamespace(data=data, mask=mask, shape=data.shape)


def test_rebin_irregular(cube):
    # Run function.
    axes_idx_edges = [0, 2, 3], [0, 2, 4], [0, 3, 5]
    data, mask = tools.rebin_irregular(cube, axes_idx_edges, operation=np.sum,
                                       operation_ignores_mask=False, handle_mask=np.all)
    # Define expected outputs.
    expected_data = np.array([[[2., 5.],
                               [6., 5.]],
                              [[4., 2.],
                               [1., 2.]]])
    expected_mask = np.zeros(expected_data.shape, dtype=bool)
    # Compare outputs with expected.
    np.testing.assert_allclose(data, expected_data)
    assert (mask == expected_mask).all()
