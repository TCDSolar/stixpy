"""
Analysis tools.
"""

import numpy as np


def rebin_irregular(data, axes_idx_edges, operation=np.mean):
    """
    Downsample array by combining irregularly sized, but contiguous, blocks of elements into bins.
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
