from pathlib import Path

import numpy as np

from stixpy.calibration.detector import get_sci_channels
from stixpy.io.readers import read_elut, read_elut_index
from stixpy.product import Product
from stixpy.utils.rebining import rebin_proportional

__all__ = ['get_elut','correct_counts']

def get_elut(date):
    root = Path(__file__).parent.parent
    elut_index_file = Path(root, *["config", "data", "elut", "elut_index.csv"])

    elut_index = read_elut_index(elut_index_file)
    elut_info = elut_index.at(date)
    if len(elut_info) == 0:
        raise ValueError(f"No ELUT for for date {date}")
    elif len(elut_info) > 1:
        raise ValueError(f"Multiple ELUTs for for date {date}")
    start_date, end_date, elut_file = list(elut_info)[0]
    sci_channels = get_sci_channels(date)
    elut_table = read_elut(elut_file, sci_channels)

    return elut_table


def correct_counts(product, method="rebin"):
    """
    Correct count for individual pixel energy calibration


    Parameters
    ----------
    product :
        A stix product
    method : str optional
        The correction method default is to rebin onto science energy channels

    Returns
    -------
    The product with the correct counts
    """
    if method not in ["rebin", "width"]:
        raise ValueError("method not on of the supported methods 'rebin' or 'width'")

    if not isinstance(product, Product) and product.level == "L1":
        raise ValueError("Only supports L1 science products")

    elut = get_elut(product.utc_timerange.center.datetime)

    # rebin back onto science channels
    counts = product.data["counts"][...]
    for did, pid in zip(elut.detector.flat, elut.pixel.flat):
        sum_orig_counts = counts[:, did, pid, 1:-1].sum(axis=-1)
        calibration_edges = elut.e_actual[did, pid, :]

        if method == "rebin":
            # need bin below 4 and above 150 to conserve total counts so [0, 4, ... 150, 160]
            sci_edges = np.hstack([elut.e, 160])
            rebinned = np.apply_along_axis(
                rebin_proportional, -1, counts[:, did, pid, 1:-1], calibration_edges, sci_edges
            )

            sum_rebined_counts = rebinned.sum(axis=-1)
            if not np.allclose(sum_orig_counts.value, sum_rebined_counts):
                raise ValueError("We are losing the precious counts")

            # zero the counts the were rebinned then add all 32 ch
            counts[:, did, pid, 1:-1] = 0
            counts[:, did, pid, :] = counts[:, did, pid, :] + rebinned * counts.unit
        elif method == "width":
            real_width = np.diff(calibration_edges)
            counts[:, pid, did, 1:-1] = counts[:, pid, did, 1:-1] / real_width

    product.data["counts"] = counts
    product.e_cal = method

    return product
