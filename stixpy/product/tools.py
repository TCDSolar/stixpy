"""
Analysis tools.
"""

import numpy as np


def rebin_irregular(data, axes_idx_edges, operation=np.mean):
    """
    Downsample array by combining irregularly sized, but contiguous, blocks of elements into bins.

    Parameters
    ----------
    data:

    axes_idx_edges: iterable of iterable of `int`
        The array indices for each axis defined in the edges of the contiguous blocks
        to rebin the data to. Each block includes the lower index and up to but not
        including the upper index.

    operation: func
        The operation defining how the elements in each block should be combined.
        Must be able to be applied over a specfic axis via an axis= kwarg.
        Defealt=`numpy.mean`

    Returns
    -------
    :

    Example
    -------
    >>> import numpy as np
    >>> from stixpy.product.tools import rebin_irregular
    >>> axes_idx_edges = [0, 2, 3], [0, 2, 4], [0, 3, 5]
    >>> data = np.ones((3, 4, 5))
    >>> rebin_irregular(data, axes_idx_edges, operation=np.sum)
    array([[[12.,  8.],
        [12.,  8.]],
       [[ 6.,  4.],
        [ 6.,  4.]]])
    """
    # TODO: extend this function to handle NDCube, not just an array.
    ndim = data.ndim
    if len(axes_idx_edges) != ndim:
        raise ValueError(f"length of axes_idx_edges must be {ndim}")
    for i, edges in enumerate(axes_idx_edges):
        if edges is None:
            continue
        r = []
        item = [slice(None)] * ndim
        for j in range(len(edges)-1):
            item[i] = slice(edges[j], edges[j+1])
            r.append(operation(data[*item], axis=i))
        data = np.stack(r, axis=i)
    return data
