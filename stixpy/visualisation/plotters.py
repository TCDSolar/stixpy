import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle, Patch
from matplotlib.widgets import Slider

from stixpy.io.readers import read_subc_params

SubCollimatorConfig = read_subc_params(
    Path(__file__).parent.parent / "config" / "data" / "detector" / "stx_subc_params.csv"
)


class PixelPlotter:
    """
    Plot individual pixel data for each detector.

    Support three kinds of plots:
       * 'pixel' which show counts as rectangular patches in the correct pixel locations using a color map
       * 'errorbar' which shows the counts and error as error bar plots one per detector
       * 'config' which display per sub-collimator configuration

    Parameters
    ----------
    prod : `Product`
        Pixel data product to plot
    kind : `string`  optional
        This sets the visualization type of the subplots the supported options are: 'pixel', 'errorbar', 'config'.
    time_indices : `list` or `numpy.ndarray`
        If an 1xN array will be treated as mask if 2XN array will sum data between given
        indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
        sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
    energy_indices : `list` or `numpy.ndarray`
        If an 1xN array will be treated as mask if 2XN array will sum data between given
        indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
        sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
    fig : optional `matplotlib.figure`
        The figure where to which the pixel plot will be added.
    cmap : `string` | `colormap` optional
        If the kind is `pixels` a colormap will be shown.

        NOTE: If the color of the special detectors 'cfl', 'bkg' is way above
        the imaging detectors, the color will be automatically set to white.

    Returns
    -------
    axes : `matplotlib.axes.Axes`

    """

    def __init__(self, prod, time_indices=None, energy_indices=None):
        self.time_indices = time_indices
        self.energy_indices = energy_indices
        from stixpy.product.sources import CompressedPixelData, RawPixelData, SummedCompressedPixelData

        if not isinstance(prod, (RawPixelData, CompressedPixelData, SummedCompressedPixelData)):
            raise ValueError(f"Can not create a pixel plot as {prod.__class__} does not contain pixel data.")
        self.prod = prod
        self.kind = "pixel"
        self.fig = None
        self.axes = None

        self.containers = defaultdict(list)
        self._prepare_data()

    def plot(self, kind="pixel", fig=None, cmap=None):
        r"""
        Generates and returns the main plot figure and axes

        Parameters
        ----------
        kind : str
            The visualization type: 'pixels', 'errorbar', or 'config'.
        fig : matplotlib.figure.Figure, optional
            An existing figure to draw on.
        cmap : str or colormap, optional
            The colormap for the 'pixels' plot.

        Returns
        -------

        """
        if kind not in ["pixel", "errorbar", "config"]:
            raise ValueError(f"Kind must be 'pixel', 'errorbar' or 'config' not '{kind}'.")
        self.kind = kind

        if fig is None:
            fig, axes = plt.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(7, 7))
        else:
            fig, axes = fig.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(7, 7))

        self.fig = fig
        self.axes = axes

        self._setup_plot_elements(cmap)
        self._create_main_layout()
        self._create_sliders()
        self._connect_update_function()

        return self.fig, self.axes

    def _setup_plot_elements(self, cmap):
        """Sets up normalization, colormaps, and fonts."""
        max_counts = self.counts[self.counts.value > 0].max().value
        min_counts = self.counts[self.counts.value > 0].min().value
        self.norm = LogNorm(min_counts, max_counts)
        self.det_font = {"weight": "regular", "size": 8}
        self.axes_font = {"weight": "regular", "size": 7}
        self.quadrant_font = {"weight": "regular", "size": 15}

        if cmap is None:
            self.clrmap = copy.copy(cm.get_cmap("viridis"))
            self.clrmap.set_over("w")
        elif isinstance(cmap, str):
            self.clrmap = copy.copy(cm.get_cmap(cmap))
        else:
            self.clrmap = cmap

    def _prepare_data(self):
        # Get the necessary data from the product
        counts, count_err, times, durations, energies = self.prod.get_data(
            time_indices=self.time_indices, energy_indices=self.energy_indices
        )

        # Pad the data arrays back to (...,32,12,...) so can loop over for the plot
        full_shape = (counts.shape[0], 32, 12, counts.shape[-1])
        counts_pad = np.full(full_shape, np.nan)
        count_err_pad = np.full(full_shape, np.nan)

        dmask = self.prod.detector_masks.masks[0].astype(bool)
        largest_pmask = self.prod.pixel_masks.masks.astype(bool)[
            self.prod.pixel_masks.masks.astype(bool).sum(axis=1).argmax()
        ]

        indices = (slice(None), *np.ix_(dmask, largest_pmask))
        counts_pad[indices] = counts
        count_err_pad[indices] = count_err

        self.times = times
        self.energies = energies
        self.counts = counts_pad * counts.unit
        self.count_err = count_err_pad * count_err.unit

    def _create_main_layout(self):
        """Draws the instrument layout and the 32 detector subplots."""
        self._draw_instrument_layout()
        if self.kind == "pixel":
            self._draw_colorbar()

        xnorm = Normalize(SubCollimatorConfig["SC Xcen"].min() * 1.5, SubCollimatorConfig["SC Xcen"].max() * 1.5)
        ynorm = Normalize(SubCollimatorConfig["SC Ycen"].min() * 1.4, SubCollimatorConfig["SC Ycen"].max() * 1.4)

        pixel_ids = [slice(0, 4), slice(4, 8), slice(8, 12)]
        if self.counts.shape[2] == 4:
            pixel_ids = [slice(0, 4)]

        for det_id in range(32):
            row, col = divmod(det_id, 8)
            ax = self.axes[row, col]
            plot_container = None

            if self.kind == "pixel":
                plot_container = self._det_pixels_plot(self.counts[0, det_id, :, 0], ax, last=(det_id == 31))
            elif self.kind == "errorbar":
                plot_container = self._det_errorbar_plot(
                    self.counts[0, det_id, :, 0], self.count_err[0, det_id, :, 0], pixel_ids, det_id, ax
                )
            elif self.kind == "config":
                plot_container = self._det_config_plot(SubCollimatorConfig[det_id], ax, det_id)

            ax.set_zorder(100)
            ax.set_position(
                [
                    xnorm(SubCollimatorConfig["SC Xcen"][det_id]),
                    ynorm(SubCollimatorConfig["SC Ycen"][det_id]),
                    1 / 11.0,
                    1 / 11.0,
                ]
            )
            self.containers[row, col].append(plot_container)
            ax.set_title(f"Det {SubCollimatorConfig['Grid Label'][det_id]}", y=0.89, **self.det_font)

    def _create_sliders(self):
        """Creates the time and energy sliders."""
        axcolor = "lightgoldenrodyellow"
        axenergy = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor)
        self.senergy = SliderCustomValue(
            ax=axenergy,
            label="Energy",
            valmin=0,
            valmax=len(self.energies) - 1,
            format_func=self._format_energy,
            valinit=0,
            valstep=1,
        )
        axetime = plt.axes([0.15, 0.01, 0.55, 0.03], facecolor=axcolor)
        self.stime = SliderCustomValue(
            ax=axetime,
            label="Time",
            valmin=0,
            valmax=self.counts.shape[0] - 1,
            format_func=self._format_time,
            valinit=1,
            valstep=1,
        )

    # --- Formatting and Drawing Helpers ---

    def _format_time(self, val):
        return self.times[val].isot

    def _format_energy(self, val):
        return f"{self.energies[val]['e_low'].value}-{self.energies[val]['e_high']}"

    def _draw_colorbar(self):
        """Creates a colormap on the left side of the figure."""
        min_c = self.counts[self.counts.value > 0].min().value
        max_c = self.counts[self.counts.value > 0].max().value
        norm = Normalize(vmin=min_c, vmax=max_c)
        cax = self.fig.add_axes([0.05, 0.15, 0.025, 0.8])
        cbar = self.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=self.clrmap), orientation="vertical", cax=cax)
        cbar.ax.set_title(f"{str(self.counts.unit)}", rotation=90, x=-0.8, y=0.4)

    def _draw_instrument_layout(self):
        """Shows the layout of the instrument."""
        x = [0, 2]
        y = [1, 1]
        ax = self.fig.add_axes([0.06, 0.055, 0.97, 0.97])
        ax.plot(x, y, c="b")
        ax.plot(y, x, c="b")
        ax.axis("off")

        ax = self.fig.add_axes([0.09, 0.08, 0.91, 0.92])
        draw_circle_1 = Circle((0.545, 0.540), 0.443, color="b", alpha=0.1)
        draw_circle_2 = Circle((0.545, 0.540), 0.07, color="#2b330b", alpha=0.95)
        self.fig.add_artist(draw_circle_1)
        self.fig.add_artist(draw_circle_2)
        ax.axis("off")

        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.text(0.19, 0.89, "Q1", **self.quadrant_font)
        ax.text(0.19, 0.17, "Q2", **self.quadrant_font)
        ax.text(0.86, 0.17, "Q3", **self.quadrant_font)
        ax.text(0.86, 0.89, "Q4", **self.quadrant_font)
        ax.axis("off")

    # --- Per-Detector Plotting Logic ---

    def _det_pixels_plot(self, counts, axes, last=False):
        """Shows a plot to visualize the pixel counts."""
        x_pos, bar1, bar2, bar3 = ["A", "B", "C", "D"], [1] * 4, [-1] * 4, [0.2] * 4
        counts = counts.reshape(3, 4)

        top = axes.bar(
            x_pos, bar1, color=self.clrmap(self.norm(counts[0, :])), width=1, zorder=1, edgecolor="w", linewidth=0.5
        )
        bottom = axes.bar(
            x_pos, bar2, color=self.clrmap(self.norm(counts[1, :])), width=1, zorder=1, edgecolor="w", linewidth=0.5
        )
        small = axes.bar(
            x_pos,
            bar3,
            color=self.clrmap(self.norm(counts[2, :])),
            width=-0.5,
            align="edge",
            bottom=-0.1,
            zorder=1,
            edgecolor="w",
            linewidth=0.5,
        )

        axes.axes.get_yaxis().set_visible(False)
        if last:
            axes.set_xticks(range(4))
            axes.set_xticklabels(x_pos)
            axes.axes.get_xaxis().set_visible(True)
        else:
            axes.set_xticks([])
            axes.axes.get_xaxis().set_visible(False)

        for i in range(4):
            top[i].data = counts[0, i]
            bottom[i].data = counts[1, i]
            small[i].data = counts[2, i]

        self._create_hover_tooltip(axes, [top, bottom, small], last)
        return top, bottom, small

    def _det_errorbar_plot(self, counts, count_err, pixel_ids, detector_id, axes):
        """Shows an errorbar plot of counts."""
        plot_cont = [
            axes.errorbar((0.5, 1.5, 2.5, 3.5), counts[pid], yerr=count_err[pid], xerr=0.5, ls="") for pid in pixel_ids
        ]
        axes.set_xticks([])
        if detector_id > 0:
            axes.set_ylabel("")
        return plot_cont

    def _det_config_plot(self, detector_config, axes, detector_id):
        """Shows a plot with detector configurations."""

        # Create Functions to convert 'Front' and 'Rear Orient'.
        def mm2deg(x):
            return x * 360.0 / 1

        def deg2mm(x):
            return x / 360.0 * 1

        # get the information that will be plotted
        if detector_config["Phase Sense"] > 0:
            phase_sense = "+"
        elif detector_config["Phase Sense"] < 0:
            phase_sense = "-"
        else:
            phase_sense = "n"

        y = [
            detector_config["Slit Width"],
            detector_config["Front Pitch"],
            detector_config["Rear Pitch"],
            0,
            deg2mm(detector_config["Front Orient"]),
            deg2mm(detector_config["Rear Orient"]),
        ]

        x = np.arange(len(y))
        color = ["black", "orange", "#1f77b4", "b", "orange", "#1f77b4"]

        # plot the information on axes
        axes.bar(x, y, color=color)
        axes.text(x=0.8, y=0.7, s=f"Phase: {phase_sense}", **self.axes_font)
        axes.set_ylim(0, 1)
        axes.axes.get_xaxis().set_visible(False)

        # Create secondary y axis
        ax2 = axes.secondary_yaxis("right", functions=(mm2deg, deg2mm))
        ax2.set_yticks([0, 90, 270, 360])
        ax2.set_yticklabels(["0°", "90°", "270°", "360°"], fontsize=8)
        ax2.set_visible(False)
        axes.axes.get_yaxis().set_visible(False)

        # Create axes labeling and legend
        if detector_id == 0:
            axes.set_yticks([0, 1])
            axes.set_ylabel("mm", **self.axes_font)
            axes.yaxis.set_label_coords(-0.1, 0.5)
            axes.axes.get_yaxis().set_visible(True)
            legend_bars = [Patch(facecolor="orange"), Patch(facecolor="#1f77b4")]
            axes.legend(legend_bars, ["Front", "Rear"], loc="center right", bbox_to_anchor=(0, 2.5))
        if detector_id == 31:
            ax2.set_visible(True)
            axes.axes.get_xaxis().set_visible(True)
            axes.set_xticks([0, 1.5, 4.5])
            axes.set_xticklabels(["Slit Width", "Pitch", "Orientation"], rotation=90)
            # leave the spaces to set the correct x position of the label!
            ax2.set_ylabel("               deg °", rotation=0, **self.axes_font)
            # x parameter doesn't change anything because it's a secondary
            # y axis (has only 1 x position).
            ax2.yaxis.set_label_coords(x=1, y=0.55)

    def _create_hover_tooltip(self, axes, artists_list, last):
        """Creates and manages the hover annotation for a subplot."""
        annot = axes.annotate(
            "",
            xy=(0, 0),
            xytext=(-60, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-"),
            zorder=33,
        )
        annot.set_visible(False)

        def update_annot(artist):
            center_x = artist.get_x() + artist.get_width() / 2
            center_y = artist.get_y() + artist.get_height() / 2
            annot.xy = (center_x, center_y)
            annot.set_text(artist.data.round(decimals=3))

        def hover(event):
            annot.set_visible(False)
            if event.inaxes == axes:
                for artist_group in artists_list:
                    for artist in artist_group:
                        contains, _ = artist.contains(event)
                        if contains:
                            update_annot(artist)
                            annot.set_visible(True)
                            break
            if last:
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect("motion_notify_event", hover)

    # --- Update Functions for Sliders ---

    def _connect_update_function(self):
        """Connects the appropriate update function to the sliders."""
        update_function = self._update_void
        if self.kind == "pixel":
            update_function = self._update_pixels
        elif self.kind == "errorbar":
            update_function = self._update_errorbar

        self.senergy.on_changed(update_function)
        self.stime.on_changed(update_function)

    def _update_void(self, _):
        """Dummy update function for static plots."""
        pass

    def _update_pixels(self, _):
        """Updates the pixel colors based on slider values."""
        energy_index, time_index = self.senergy.val, self.stime.val
        for detector_id in range(32):
            row, col = divmod(detector_id, 8)
            cnts = self.counts[time_index, detector_id, :, energy_index]
            top, bottom, small = self.containers[row, col][0]
            cnts = cnts.reshape([3, 4])
            for pix_artist, pix in zip(range(4), range(12)):
                norm_counts = self.norm(cnts[0][pix].value)
                top[pix_artist].set_color(self.clrmap(norm_counts))
                top[pix_artist].data = cnts[0][pix]
                top[pix_artist].set_edgecolor("w")

                norm_counts = self.norm(cnts[1][pix].value)
                bottom[pix_artist].set_color(self.clrmap(norm_counts))
                bottom[pix_artist].data = cnts[1][pix]
                bottom[pix_artist].set_edgecolor("w")

                norm_counts = self.norm(cnts[2][pix].value)
                small[pix_artist].set_color(self.clrmap(norm_counts))
                small[pix_artist].data = cnts[2][pix]
                small[pix_artist].set_edgecolor("w")
        self.fig.canvas.draw_idle()

    def _update_errorbar(self, _):
        energy_index = self.senergy.val
        time_index = self.stime.val
        pids_ = [slice(0, 4), slice(4, 8), slice(8, 12)]
        if self.counts.shape[2] == 4:
            pids_ = [slice(0, 4)]

        for did in range(32):
            r, c = divmod(did, 8)
            self.axes[r, c].set_ylim(0, np.nanmax(self.counts[time_index, :, :, energy_index]) * 1.2)

            for i, pid in enumerate(pids_):
                lines, caps, bars = self.containers[r, c][0][i]
                lines.set_ydata(self.counts[time_index, did, pid, energy_index])

                # horizontal bars at value
                segs = np.array(bars[0].get_segments())
                if segs.size > 0:
                    segs[:, 0, 0] = [0.0, 1.0, 2.0, 3.0]
                    segs[:, 1, 0] = [1.0, 2.0, 3.0, 4.0]
                    segs[:, 0, 1] = self.counts[time_index, did, pid, energy_index]
                    segs[:, 1, 1] = self.counts[time_index, did, pid, energy_index]
                    bars[0].set_segments(segs)
                    # vertical bars at +/- error
                    segs = np.array(bars[1].get_segments())
                    segs[:, 0, 0] = [0.5, 1.5, 2.5, 3.5]
                    segs[:, 1, 0] = [0.5, 1.5, 2.5, 3.5]
                    segs[:, 0, 1] = (
                        self.counts[time_index, did, pid, energy_index]
                        - self.count_err[time_index, did, pid, energy_index]
                    )
                    segs[:, 1, 1] = (
                        self.counts[time_index, did, pid, energy_index]
                        + self.count_err[time_index, did, pid, energy_index]
                    )
                    bars[1].set_segments(segs)
        self.fig.canvas.draw_idle()


class SliderCustomValue(Slider):
    """
    A slider with a customisable formatter
    """

    def __init__(self, *args, format_func=None, **kwargs):
        if format_func is not None:
            self._format = format_func
        super().__init__(*args, **kwargs)
