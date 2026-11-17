import numpy as np

from sunpy.util.decorators import deprecated

from stixpy.calibration.elut import get_elut as _get_elut
from stixpy.product import Product
from stixpy.utils.rebining import rebin_proportional

__all__ = ["get_elut", "correct_counts"]


@deprecated(since="0.4", alternative="stixpy.calibration.elut.get_elut")
def get_elut(date):
    r"""
    Get the energy lookup table (ELUT) for the given date.

    Deprecated alias of `stixpy.calibration.elut.get_elut`.

    Parameters
    ----------
    date : `datetime.datetime` or `astropy.time.Time`
        Date to look up the ELUT.

    Returns
    -------
    ELUT
        See `stixpy.calibration.elut.get_elut`.
    """
    return _get_elut(date)


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

    elut = _get_elut(product.utc_timerange.center.datetime)

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
