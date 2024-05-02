import numpy as np

from .. import tools

def test_rebin_irregular():
    # Define inputs
    axes_idx_edges = [0, 2, 3], [0, 2, 4], [0, 3, 5]
    data = np.ones((3, 4, 5))
    operation = np.sum
    # Run function.
    output = tools.rebin_irregular(data, axes_idx_edges, operation=operation)
    # Compare with expected
    expected = np.array([[[12.,  8.],
                           [12.,  8.]],
                          [[ 6.,  4.],
                           [ 6.,  4.]]])
    np.testing.assert_allclose(output, expected)

