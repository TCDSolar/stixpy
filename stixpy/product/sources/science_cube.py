"""
Experimental `ndcube.NDCube` representation of STIX science pixel data.

Counts are stored as an NDCube with array axes ``(time, detector, pixel, energy)``
(or ``(time, energy)`` for spectrograms). World coordinates are table coordinates
(bin centres) and all per-axis ancillary data (durations, energy edges, masks,
triggers, rcr) live in an axis-aware `STIXMeta` so they stay consistent under slicing
and summation.
"""

import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from ndcube import NDCube, NDMeta
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate

from stixpy.product.tools import rebin_by_edges

__all__ = ["STIXMeta", "ScienceCube"]

TIME, DETECTOR, PIXEL, ENERGY = range(4)

# How axis-aware metadata is combined when bins are summed along an axis.
# Keys not listed here fall back to `NDMeta` behaviour (axis-awareness dropped).
# `None` marks index labels, merged into tuples of the contributing indices.
_AGGREGATORS = {
    "timedel": np.sum,
    "e_low": np.min,
    "e_high": np.max,
    "detector_masks": np.any,
    "pixel_masks": np.any,
    "triggers": np.sum,
    "detector_index": None,  # handled specially, list of source indices
    "pixel_index": None,
    "time_start": np.min,
}


class STIXMeta(NDMeta):
    """
    Axis-aware metadata that aggregates known STIX quantities when data is rebinned.

    Overrides the `ndcube.NDMeta` rebin hook so durations are summed, energy edges widened, masks combined etc.
    rather than axis-awareness being dropped. Unknown axis-aware entries fall back to `ndcube.NDMeta` behaviour.
    """

    def rebin(self, bin_shape):
        """
        Rebin metadata consistent with `ndcube.NDCube.rebin`.

        Parameters
        ----------
        bin_shape : array-like
            The number of pixels in a bin in each dimension.
        """
        bin_shape = np.round(bin_shape).astype(int)[: len(self.data_shape)]
        if (np.mod(self.data_shape, bin_shape) != 0).any():
            raise ValueError("bin_shape must be integer factors of their associated axes.")
        edges = [None if b == 1 else np.arange(0, n + 1, b) for n, b in zip(self.data_shape, bin_shape)]
        return self.rebin_by_edges(edges)

    def rebin_by_edges(self, axes_idx_edges):
        """
        Rebin metadata consistent with `stixpy.product.tools.rebin_by_edges`.

        Parameters
        ----------
        axes_idx_edges : iterable of iterable of `int` or `None`
            For each array axis, the indices of the edges of the contiguous blocks.
        """
        new = self
        for axis, edges in enumerate(axes_idx_edges):
            if edges is not None:
                new = new._rebin_axis(axis, edges)
        return new

    def take(self, axis, indices):
        """
        Select ``indices`` along ``axis`` for all metadata associated with that axis.
        """
        new = STIXMeta(key_comments=None)
        for name, value in self.items():
            axes = self.axes.get(name)
            if axes is not None and axis in axes:
                pos = list(axes).index(axis)
                value = value[(slice(None),) * pos + (np.asarray(indices),)]
            new.add(name, value, key_comment=self.key_comments.get(name), axes=axes)
        return new

    def _rebin_axis(self, axis, edges):
        new = STIXMeta()
        for name, value in self.items():
            axes = self.axes.get(name)
            comment = self.key_comments.get(name)
            if axes is None or axis not in axes:
                new.add(name, value, key_comment=comment, axes=axes)
                continue
            pos = list(axes).index(axis)
            agg = _AGGREGATORS.get(name, "drop")
            if agg == "drop":
                new.add(name, value, key_comment=comment, axes=None)
                continue
            chunks = []
            for start, end in zip(edges[:-1], edges[1:]):
                chunk = value[(slice(None),) * pos + (slice(start, end),)]
                if agg is None:
                    merged = np.empty(1, dtype=object)
                    merged[0] = tuple(i for c in chunk for i in np.atleast_1d(c))
                    chunks.append(merged)
                elif isinstance(value, Time):
                    chunks.append(chunk.min().reshape(1))
                elif value.dtype.names:  # structured arrays used for non grid-aligned per-axis data
                    out = np.zeros(1, dtype=value.dtype)
                    for field in value.dtype.names:
                        out[field] = agg(chunk[field], axis=0)
                    chunks.append(out)
                else:
                    chunks.append(np.expand_dims(agg(chunk, axis=pos), pos))
            new.add(name, np.concatenate(chunks, axis=pos), key_comment=comment, axes=axes)
        return new


def _structured(name, values):
    """Pack an ``(N, ...)`` array into a length-N structured array so it can be tied to a single axis."""
    values = np.asarray(values)
    out = np.zeros(values.shape[0], dtype=[(name, values.dtype, values.shape[1:])])
    out[name] = values
    return out


def _energy_centres(e_low, e_high):
    # The overflow channel has e_high = inf, use its lower edge as the coordinate.
    centres = (e_low + e_high) / 2
    return np.where(np.isfinite(centres), centres, e_low)


def _index_labels(indices):
    labels = np.empty(len(indices), dtype=object)
    labels[:] = [(i,) for i in indices]
    return labels


class ScienceCube(NDCube):
    """
    STIX science pixel data as an `~ndcube.NDCube`.

    Construct using `ScienceCube.from_product`.
    """

    @classmethod
    def from_arrays(cls, *, counts, uncertainty, meta):
        """
        Build cube and coordinates from counts and an axis-aware `STIXMeta`.

        Coordinates are derived from ``meta`` (``time_start``, ``timedel``, ``e_low``, ``e_high``, ``*_index``)
        so the cube can be rebuilt consistently after any rebinning.
        """
        times = meta["time_start"] + meta["timedel"] / 2
        tc = TimeTableCoordinate(times, names="time", physical_types="time")
        ec = QuantityTableCoordinate(
            _energy_centres(meta["e_low"], meta["e_high"]), names="energy", physical_types="em.energy"
        )
        if counts.ndim == 4:
            # Index coordinates are the mean of the contributing detectors/pixels (as `rebin` would do).
            det = [np.mean(i) for i in meta["detector_index"]] * u.one
            pix = [np.mean(i) for i in meta["pixel_index"]] * u.one
            dc = QuantityTableCoordinate(det, names="detector", physical_types="custom:detector")
            pc = QuantityTableCoordinate(pix, names="pixel", physical_types="custom:pixel")
            # Compound order is WCS (reversed array) order.
            wcs = (ec & pc & dc & tc).wcs
        else:
            wcs = (ec & tc).wcs
        if uncertainty is not None:
            uncertainty = StdDevUncertainty(uncertainty.to_value(counts.unit))
        return cls(counts.value, wcs=wcs, unit=counts.unit, uncertainty=uncertainty, meta=meta)

    @classmethod
    def from_product(cls, product):
        """
        Create a `ScienceCube` from a `~stixpy.product.sources.science.ScienceData` product.
        """
        data, energies = product.data, product.energies
        counts = data["counts"].astype(float)
        # Poisson plus compression error
        uncertainty = np.sqrt(counts.value + data["counts_err"].astype(float).value ** 2) * counts.unit
        n_time = len(data)

        meta = STIXMeta(dict(product.meta))
        timedel = data["timedel"].to(u.s)
        meta.add("timedel", timedel, axes=TIME)
        meta.add("time_start", data["time"] - timedel / 2, axes=TIME)
        meta.add("rcr", np.asarray(data["rcr"]), axes=TIME)
        meta.add("triggers", _structured("triggers", data["triggers"]), axes=TIME)
        meta.add("e_low", energies["e_low"], axes=counts.ndim - 1)
        meta.add("e_high", energies["e_high"], axes=counts.ndim - 1)

        if counts.ndim == 4:
            meta.add("detector_index", _index_labels(range(counts.shape[DETECTOR])), axes=DETECTOR)
            meta.add("pixel_index", _index_labels(range(counts.shape[PIXEL])), axes=PIXEL)
            meta.add("detector_masks", np.asarray(data["detector_masks"]).astype(bool), axes=(TIME, DETECTOR))
            pixel_masks = np.asarray(data["pixel_masks"]).astype(bool)
            if pixel_masks.ndim == 2:
                meta.add("pixel_masks", pixel_masks, axes=(TIME, PIXEL))
            else:
                # Summed pixel data: (time, pixel_set, 12) does not align with the pixel axis.
                meta.add("pixel_masks", _structured("pixel_masks", pixel_masks), axes=TIME)
        else:
            detector_masks = np.asarray(product.control["detector_masks"][data["control_index"]]).astype(bool)
            meta.add("detector_masks", _structured("detector_masks", detector_masks), axes=TIME)
            meta.add("pixel_masks", _structured("pixel_masks", data["pixel_masks"]), axes=TIME)

        assert meta.data_shape[0] == n_time
        return cls.from_arrays(counts=counts, uncertainty=uncertainty, meta=meta)

    # Convenience accessors --------------------------------------------------------------------

    @property
    def times(self):
        """Centre of each time bin."""
        return self.meta["time_start"] + self.meta["timedel"] / 2

    @property
    def durations(self):
        return self.meta["timedel"]

    @property
    def energy_edges(self):
        return self.meta["e_low"], self.meta["e_high"]

    # Operations -------------------------------------------------------------------------------

    def rebin(self, bin_shape, operation=np.sum, propagate_uncertainties=True, **kwargs):
        """
        `ndcube.NDCube.rebin` with coordinates rebuilt from the rebinned metadata.

        Defaults to summing, as appropriate for counts, and propagating uncertainties.
        """
        new = super().rebin(bin_shape, operation=operation, propagate_uncertainties=propagate_uncertainties, **kwargs)
        return self._rebuild(new)

    def rebin_by_edges(self, axes_idx_edges, operation=np.sum, propagate_uncertainties=True, **kwargs):
        """
        `stixpy.product.tools.rebin_by_edges` with coordinates rebuilt from the rebinned metadata.

        Coordinates are recomputed from the combined bin edges e.g. the energy of a 4-150 keV bin is 77 keV,
        not the mean of the original bin centres. Defaults to summing and propagating uncertainties.
        """
        new = rebin_by_edges(
            self, axes_idx_edges, operation=operation, propagate_uncertainties=propagate_uncertainties, **kwargs
        )
        return self._rebuild(new)

    def sum_ranges(self, axis, ranges):
        """
        Sum the data over inclusive index ranges along an axis, as `ScienceData.get_data`.

        Parameters
        ----------
        axis : `int` or `str`
            Array axis index or one of 'time', 'detector', 'pixel', 'energy'.
        ranges : `list`
            Sorted, non-overlapping inclusive ``[start, end]`` pairs e.g. ``[[0, 4], [5, 9]]``.
            Gaps between ranges are excluded e.g. ``[[0, 2], [5, 7]]``.

        Returns
        -------
        `ScienceCube`
        """
        if isinstance(axis, str):
            axis = self._axis_names().index(axis)
        ranges = np.atleast_2d(ranges)
        starts, stops = ranges[:, 0], ranges[:, 1] + 1
        if np.any(stops <= starts) or np.any(starts[1:] < stops[:-1]):
            raise ValueError("ranges must be sorted and non-overlapping [start, end] pairs.")

        edges = np.unique(np.concatenate([starts, stops]))
        axes_idx_edges = [None] * self.data.ndim
        axes_idx_edges[axis] = edges
        new = self.rebin_by_edges(axes_idx_edges)

        keep = np.flatnonzero(np.isin(edges[:-1], starts))
        if len(keep) == len(edges) - 1:
            return new
        # Drop the bins formed from gaps between ranges
        uncertainty = None if new.uncertainty is None else np.take(new.uncertainty.array, keep, axis=axis) * new.unit
        return type(self).from_arrays(
            counts=np.take(new.data, keep, axis=axis) * new.unit,
            uncertainty=uncertainty,
            meta=new.meta.take(axis, keep),
        )

    def _rebuild(self, cube):
        uncertainty = None if cube.uncertainty is None else cube.uncertainty.array * cube.unit
        return type(self).from_arrays(counts=cube.data * cube.unit, uncertainty=uncertainty, meta=cube.meta)

    def to_rate(self, vtype="dcr"):
        """
        Normalise counts by duration (``'cr'``) or duration and energy width (``'dcr'``).
        """
        if vtype == "c":
            return self
        shape = [1] * self.data.ndim
        shape[0] = -1
        norm = self.meta["timedel"].to(u.s).reshape(shape)
        if vtype == "dcr":
            shape = [1] * self.data.ndim
            shape[-1] = -1
            norm = norm * (self.meta["e_high"] - self.meta["e_low"]).reshape(shape)
        elif vtype != "cr":
            raise ValueError("vtype must be one of 'c', 'cr', 'dcr'.")
        # np.broadcast_to drops the unit so reattach it
        return self / (np.broadcast_to(norm.value, self.shape) * norm.unit)

    def _axis_names(self):
        return ["time", "detector", "pixel", "energy"] if self.data.ndim == 4 else ["time", "energy"]
