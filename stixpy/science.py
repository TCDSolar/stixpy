from collections import defaultdict
from pathlib import Path

import astropy.units as u
import numpy as np

from astropy.io import fits
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.dates import date2num, HourLocator, DateFormatter
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from sunpy.time.timerange import TimeRange
from stixcore.config.reader import read_subc_params

__all__ = ['ScienceData', 'RawPixelData', 'CompressedPixelData', 'SummedCompressedPixelData',
           'Visibility', 'Spectrogram', 'PPrintMixin', 'IndexMasks', 'DetectorMasks', 'PixelMasks',
           'EnergyMasks']

quantity_support()

SCP = read_subc_params(Path(read_subc_params.__code__.co_filename).parent
                            / "data" / "common" / "detector" / "stx_subc_params.csv")
class PPrintMixin:
    """
    Provides pretty printing for index masks.
    """
    @staticmethod
    def _pprint_indices(indices):
        groups = np.split(np.r_[:len(indices)], np.where(np.diff(indices) != 1)[0] + 1)
        out = ''
        for group in groups:
            if group.size < 3:
                out += f'{indices[group]}'
            else:
                out += f'[{indices[group[0]]}...{indices[group[-1]]}]'

        return out


class IndexMasks(PPrintMixin):
    """

    Attributes
    ----------
    masks : 'numpy.ndarray`
        The mask arrays
    indices : `numpy.ndarray'
        The indices the mask/s applies to

    """
    def __init__(self, mask_array):
        masks = np.unique(mask_array, axis=0)
        indices = [np.argwhere(np.all(mask_array == mask, axis=1)).reshape(-1)
                   for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f'{self.__class__.__name__}\n'
        for m, i in zip(self.masks, self.indices):
            text += f'    {self._pprint_indices(i)}: ' \
                    f'[{",".join(np.where(m, np.arange(m.size), np.full(m.size, "_")))}]\n'
        return text


class DetectorMasks(IndexMasks):
    """
    Detector Index Masks
    """
    pass


class EnergyMasks(IndexMasks):
    """
    Energy Index Mask
    """
    pass


class PixelMasks(PPrintMixin):
    """
    Pixel Index Masks
    """
    def __init__(self, pixel_masks):
        masks = np.unique(pixel_masks, axis=0)
        indices = []
        if masks.ndim == 2:
            indices = [np.argwhere(np.all(pixel_masks == mask, axis=1)).reshape(-1)
                       for mask in masks]
        elif masks.ndim == 3:
            indices = [np.argwhere(np.all(pixel_masks == mask, axis=(1, 2))).reshape(-1)
                       for mask in masks]
        self.masks = masks
        self.indices = indices

    def __repr__(self):
        text = f'{self.__class__.__name__}\n'
        for m, i in zip(self.masks, self.indices):
            text += f'    {self._pprint_indices(i)}: ' \
                    f'[{str(np.where(m.shape[0], np.eye(*m.shape), np.full(m.shape, "_")))}]\n'
        return text


class SpectrogramPlotMixin:
    """
    Spectrogram plot mixin providing spectrogram plotting for pixel data.
    """
    def plot_spectrogram(self, axes=None, time_indices=None, energy_indices=None):
        """
        Plot a spectrogram for the selected time and energies.

        Parameters
        ----------
        axes : optional `matplotlib.axes`
            The axes the plot the spectrogram.
        time_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.

        Returns
        -------
        `matplotlib.axes`
        """

        if axes is None:
            fig, axes = plt.subplots()
        else:
            axes = plt.gca()
            fig = axes.get_figure()

        counts_shape = self.data['counts'].shape
        pid = None
        did = None
        if len(counts_shape) == 4:
            pid = [[0, counts_shape[2]]]
            did = [[0, counts_shape[1]]]

        counts, errors, times, timedeltas, energies = self.get_data(detector_indices=did,
                                                                    pixel_indices=pid,
                                                                    time_indices=time_indices,
                                                                    energy_indices=energy_indices)

        labels = [f'{el.value} - {eh.value}' for el, eh in energies['e_low', 'e_high']]

        x_lims = date2num(times.to_datetime())
        y_lims = [0, len(energies) - 1]

        im = axes.imshow(counts[:, 0, 0, :].T.value,
                         extent=[x_lims[0], x_lims[-1], y_lims[0], y_lims[1]],
                         origin='lower', aspect='auto', interpolation='none', norm=LogNorm())

        fig.colorbar(im).set_label(format(counts.unit))
        axes.xaxis_date()
        axes.set_yticks(range(y_lims[0], y_lims[1] + 1))
        axes.set_yticklabels(labels)
        minor_loc = HourLocator()
        axes.xaxis.set_minor_locator(minor_loc)
        axes.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()

        return axes


class TimesSeriesPlotMixin:
    """
    TimesSeries plot mixin providing timeseries plotting for pixel data.
    """
    def plot_timeseries(self, time_indices=None, energy_indices=None, axes=None, error_bar=False):
        """
        Plot a times series of the selected times and energies.

        Parameters
        ----------
        axes : optional `matplotlib.axes`
            The axes the plot the spectrogram.
        time_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
        error_bar : optional `bool`
            Add error bars to plot.

        Returns
        -------
        `matplotlib.axes`

        """
        if axes is None:
            fig, axes = plt.subplots()
        else:
            axes = plt.gca()
            fig = axes.get_figure()

        counts_shape = self.data['counts'].shape
        counts, errors, times, timedeltas, energies \
            = self.get_data(detector_indices=None,
                            pixel_indices=None,
                            time_indices=time_indices, energy_indices=energy_indices)
        labels = [f'{el.value} - {eh.value}' for el, eh in energies['e_low', 'e_high']]

        if error_bar:
            [axes.errorbar(times.to_datetime(), counts[:, 0, 0, ech], yerr=errors[:, 0, 0, ech],
                           fmt='.', label=labels[ech]) for ech in range(len(energies))]
        else:
            axes.plot(times.to_datetime(), counts[:, 0, 0, :])
        axes.set_yscale('log')
        # axes.legend()
        axes.xaxis.set_major_formatter(DateFormatter("%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()

        return axes


class PixelPlotMixin:
    """
    Pixel plot mixin providing pixel plotting for pixel data.
    """
    def plot_pixels(self, *, kind='pixels', time_indices=None, energy_indices=None, fig=None, cmap='viridis'):
        """
        Plot individual pixel data for each detector.

        Parameters
        ----------
        plottype : `string`
            The user can choose how he wants to visualize the data. The possible options ar 'errorbar' and 'pixels'.
            The data will then be shown in the selected style.
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
        cmap : optional `colormap' type
            If the plottype is `bar` a colormap will be shown. Select different types of colormaps
            to change the colormap used.

        Returns
        -------
        `matplotlib.figure`
            The figure
        """

        if not kind in ["pixels", "errorbar", "config"]:
            kind = 'pixels'

        if fig:
            axes = fig.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(7, 7))
        else:
            fig, axes = plt.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(7, 7))

        counts, count_err, times, dt, energies = self.get_data(time_indices=time_indices,
                                                               energy_indices=energy_indices)

        max_counts = counts.max().value
        min_counts = counts.min().value
        norm = plt.Normalize(min_counts, max_counts)
        det_font = {'weight': 'regular', 'size': 6}
        ax_font = {'weight': 'regular', 'size': 7}
        q_font = {'weight': 'regular', 'size': 15}
        clrmap = cm.get_cmap(cmap)

        def timeval(val):
            return times[val].isot

        def energyval(val):
            return f"{energies[val]['e_low'].value}-{energies[val]['e_high']}"

        def det_pixels_plot(counts, norm, axes, clrmap, fig, last=False):
            """
            Parameters
            ----------
            counts = data collection of the number of counts
            norm = normalizes the data in the parentheses
            axes = the axes in which the data will be plotted
            clrmap = the colormap which will be used to visualize the data/counts
            fig = helps the hover function to understand in which figure it should return the label

            Returns
            -------
            A barplot which visualizes the data with the help of a colorbar.
            """
            # Set the variables needed.
            bar1 = [1, 1, 1, 1]
            bar2 = [-1, -1, -1, -1]
            bar3 = [.2, .2, .2, .2]
            x_pos = ['A', 'B', 'C', 'D']

            cdata_top = counts[0:4]
            cdata_bottom = counts[4:8]
            cdata_small = counts[8:12]

            top = axes.bar(x_pos, bar1, color=clrmap(norm(cdata_top)), width=1, zorder=1, edgecolor="w", linewidth=0.5)
            bottom = axes.bar(x_pos, bar2, color=clrmap(norm(cdata_bottom)), width=1, zorder=1, edgecolor="w", linewidth=0.5)
            small = axes.bar(x_pos, bar3, color=clrmap(norm(cdata_small)), width=-0.5, align='edge', bottom=-0.1, zorder=1, edgecolor="w", linewidth=0.5)

            for i in range(4):
                top[i].data = cdata_top[i]
                bottom[i].data = cdata_bottom[i]
                small[i].data = cdata_small[i]

            axes.axes.get_xaxis().set_visible(False)
            axes.axes.get_yaxis().set_visible(False)

            annot = axes.annotate("", xy=(0, 0), xytext=(-60, 20),
                                  textcoords="offset points",
                                  bbox=dict(boxstyle="round", fc="w"),
                                  arrowprops=dict(arrowstyle="-"), zorder=33)

            annot.set_visible(False)

            def update_annot(artist, annot):
                """ update tooltip when hovering a given plotted object """
                # find the middle of the bar
                center_x = artist.get_x() + artist.get_width() / 2
                center_y = artist.get_y() + artist.get_height() / 2
                annot.xy = (center_x, center_y)

                annot.set_text(artist.data)
                annot.get_bbox_patch().set_alpha(1)

            def hover(event):
                """ update and show a tooltip while hovering an object; hide it otherwise """
                # one wants to hide the annotation only if no artist in the graph is hovered
                annot.set_visible(False)
                if isinstance(event.inaxes, type(axes)):
                    for p in [top, bottom, small]:
                        for artist in p:
                            contains, _ = artist.contains(event)
                            if contains:
                                update_annot(artist, annot)
                                annot.set_visible(True)
                if last:
                    fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)
            return (top, bottom, small)

        def det_config_plot(detector_config, axes, font, detector_id):

            def mm2deg(x):
                return x * 360.0 / 1

            def deg2mm(x):
                return x / 360.0 * 1

            if detector_config['Phase Sense'] > 0:
                phase_sense = '+'
            elif detector_config['Phase Sense'] < 0:
                phase_sense = '-'
            else:
                phase_sense = 'n'

            y = [detector_config['Slit Width'],
                 detector_config['Front Pitch'],
                 detector_config['Rear Pitch'],
                 0,
                 deg2mm(detector_config['Front Orient']),
                 deg2mm(detector_config['Rear Orient'])]

            x = np.arange(len(y))
            x_ticklabels = ['Slit Width', 'Pitch', '', '', 'Orientation', '']
            color = ['black', 'orange', '#1f77b4', 'b', 'orange', '#1f77b4']

            # plot the information on axes
            axes.bar(x, y, color=color)
            axes.text(x=0.8, y=0.7, s=f'Phase: {phase_sense}', **font)
            axes.set_ylim(0, 1)
            axes.axes.get_xaxis().set_visible(False)

            # Create secondary y axis
            ax2 = axes.secondary_yaxis('right', functions=(mm2deg, deg2mm))
            ax2.set_yticks([0, 90, 270, 360])
            ax2.set_yticklabels(['0°', '90°', '270°', '360°'], fontsize=8)

            if not detector_id == 22:
                axes.set_yticks([0, 1])
                axes.set_ylabel('mm', **font)
                axes.yaxis.set_label_coords(-0.1, 0.5)
                ax2.set_visible(False)
                axes.axes.get_yaxis().set_visible(False)
            elif detector_id == 22:
                # leave the spaces to set the correct x position of the label!!
                ax2.set_ylabel('               deg °', rotation=0, **font)
                # x parameter doesn't change anything because it's a secondary y axis (has only 1 x position).
                ax2.yaxis.set_label_coords(x=1, y=0.55)

            if detector_id == 0:
                axes.axes.get_yaxis().set_visible(True)
            elif detector_id == 27:
                axes.axes.get_xaxis().set_visible(True)
                axes.set_xticklabels(x_ticklabels, fontsize=8, rotation=-90)

        def colorbar(counts, min_counts, max_counts, clrmap, fig):
            """ Creates a colormap at the left side of the created figure. """
            norm = colors.Normalize(vmin=min_counts, vmax=max_counts)
            cax = fig.add_axes([0.05, 0.15, 0.025, 0.8])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=clrmap), orientation='vertical', cax=cax)
            cbar.ax.set_title(f'{str(counts.unit)}', rotation=90, x=-0.8, y=0.4)

        def helpers(fig, font):
            """ Shows helpers in the background of the subplots to make it easier to locate the detectors. """
            x = [0, 2]
            y = [1, 1]
            fig.add_axes([0.06, 0.055, 0.97, 0.97])
            plt.plot(x, y, c='b')
            plt.plot(y, x, c='b')
            plt.axis('off')
            fig.add_axes([0.09, 0.08, 0.91, 0.92])
            draw_circle_1 = plt.Circle((0.545, 0.540), 0.443, color='b', alpha=0.1)
            draw_circle_2 = plt.Circle((0.545, 0.540), 0.07, color='#2b330b', alpha=0.95)
            fig.add_artist(draw_circle_1)
            fig.add_artist(draw_circle_2)
            plt.axis('off')
            fig.add_axes([0, 0, 1, 1])
            plt.text(0.19, 0.89, 'Q1', **font)
            plt.text(0.19, 0.17, 'Q2', **font)
            plt.text(0.86, 0.17, 'Q3', **font)
            plt.text(0.86, 0.89, 'Q4', **font)
            plt.axis('off')

        helpers(fig, q_font)

        """
        def _switch(event):
            Switches the plottype if the 'switch' button is getting pressed.
            # det_config_plot(SCP[detector_id], axes[3, 5], det_font)
            # axes[3, 5].set_zorder(500)
            # plt.show()
            return bt

        button_pos = fig.add_axes([0.87, 0.09, 0.1, 0.06])
        button_pos.set_zorder(200)
        bt = Button(button_pos, 'switch', color='gray', hovercolor='green')
        bt.on_clicked(_switch)
        """

        axcolor = 'lightgoldenrodyellow'
        axenergy = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor)
        senergy = SliderCustomValue(ax=axenergy,
                                    label='Energy',
                                    valmin=0,
                                    valmax=len(energies)-1,
                                    format_func=energyval,
                                    valinit=0, valstep=1)
        axetime = plt.axes([0.15, 0.01, 0.55, 0.03], facecolor=axcolor)
        stime = SliderCustomValue(ax=axetime,
                                  label='Time',
                                  valmin=0,
                                  valmax=counts.shape[0]-1,
                                  format_func=timeval,
                                  valinit=1, valstep=1)

        pixel_ids = [slice(0, 4), slice(4, 8), slice(8, 12)]
        if counts.shape[2] == 4:
            pixel_ids = [slice(0, 4)]

        containers = defaultdict(list)

        xnorm = plt.Normalize(SCP["SC Xcen"].min()*1.5, SCP["SC Xcen"].max()*1.5)
        ynorm = plt.Normalize(SCP["SC Ycen"].min()*1.5, SCP["SC Ycen"].max()*1.5)
        if kind == 'pixels':
            colorbar(counts, min_counts, max_counts, clrmap, fig)

        for detector_id in range(32):
            row, col = divmod(detector_id, 8)
            plot_cont = object
            if kind == 'pixels':
                plot_cont = det_pixels_plot(counts[0, detector_id, :, 0], norm, axes[row, col], clrmap, fig, last=detector_id==31)
                axes[row, col].set_xticks([])
            elif kind == 'errorbar':
                for pixel_id in pixel_ids:
                    plot_cont = axes[row, col].errorbar((0.5, 1.5, 2.5, 3.5),
                                                      counts[0, detector_id, pixel_id, 0],
                                                      yerr=count_err[0, detector_id, pixel_id, 0],
                                                      xerr=0.5, ls='')
                axes[row, col].set_xticks([])
            elif kind == 'config':
                # if not detector_id in [8, 9]:
                plot_cont = det_config_plot(SCP[detector_id], axes[row, col], ax_font, detector_id)

            axes[row, col].set_zorder(100)
            x = SCP["SC Xcen"][detector_id]
            y = SCP["SC Ycen"][detector_id]
            axes[row, col].set_position([xnorm(x), ynorm(y), 1/11.0, 1/11.0])

            containers[row, col].append(plot_cont)
            axes[row, col].set_title(f'Det {SCP["Grid Label"][detector_id]}', y=0.89,  **det_font)
            if detector_id > 0:
                axes[row, col].set_ylabel('')

        def update_void(_):
            energy_index = senergy.val
            time_index = stime.val

        def update_pixels(_):
            energy_index = senergy.val
            time_index = stime.val

            for detector_id in range(32):
                row, col = divmod(detector_id, 8)
                # axes[row, col].clear()
                cnts = counts[time_index, detector_id, :, energy_index]
                top, bottom, small = containers[row, col][0]
                cnts = cnts.reshape([3, 4])
                for pix_artist, pix in zip(range(4), range(12)):
                    norm_counts = norm(cnts[0][pix].value)
                    top[pix_artist].set_color(clrmap(norm_counts))
                    top[pix_artist].data = cnts[0][pix]
                    top[pix_artist].set_edgecolor("w")

                    norm_counts = norm(cnts[1][pix].value)
                    bottom[pix_artist].set_color(clrmap(norm_counts))
                    bottom[pix_artist].data = cnts[1][pix]
                    bottom[pix_artist].set_edgecolor("w")

                    norm_counts = norm(cnts[2][pix].value)
                    small[pix_artist].set_color(clrmap(norm_counts))
                    small[pix_artist].data = cnts[2][pix]
                    small[pix_artist].set_edgecolor("w")

        def update_errorbar(_):
            energy_index = senergy.val
            time_index = stime.val
            pids_ = [slice(0, 4), slice(4, 8), slice(8, 12)]
            if counts.shape[2] == 4:
                pids_ = [slice(0, 4)]

            imaging_mask = np.ones(32, bool)
            imaging_mask[8:10] = False

            for did in range(32):
                r, c = divmod(did, 8)
                axes[r, c].set_ylim(0, counts[time_index, imaging_mask, :,
                                              energy_index].max()*1.2)

                for i, pid in enumerate(pids_):
                    lines, caps, bars = containers[r, c][i]
                    lines.set_ydata(counts[time_index, did, pid, energy_index])

                    # horizontal bars at value
                    segs = np.array(bars[0].get_segments())
                    segs[:, 0, 0] = [0., 1., 2., 3.]
                    segs[:, 1, 0] = [1., 2., 3., 4.]
                    segs[:, 0, 1] = counts[time_index, did, pid, energy_index]
                    segs[:, 1, 1] = counts[time_index, did, pid, energy_index]
                    bars[0].set_segments(segs)
                    # vertical bars at +/- error
                    segs = np.array(bars[1].get_segments())
                    segs[:, 0, 0] = [0.5, 1.5, 2.5, 3.5]
                    segs[:, 1, 0] = [0.5, 1.5, 2.5, 3.5]
                    segs[:, 0, 1] = counts[time_index, did, pid, energy_index] \
                        - count_err[time_index, did, pid, energy_index]
                    segs[:, 1, 1] = counts[time_index, did, pid, energy_index] \
                        + count_err[time_index, did, pid, energy_index]
                    bars[1].set_segments(segs)

        updatefunction = update_pixels

        if kind == 'config':
            updatefunction = update_void
        elif kind == 'errorbar':
            updatefunction = update_errorbar

        senergy.on_changed(updatefunction)
        stime.on_changed(updatefunction)


class ScienceData:
    """
    Basic science data class
    """
    def __init__(self, *, header, control, data, energies):
        """

        Parameters
        ----------
        header : `astropy.fits.Header`
            Fits header
        control : `astropy.table.QTable`
            Fits file control extension
        data :` astropy.table.QTable`
            Fits file data extension
        energies : `astropy.table.QTable`
            Fits file energy extension
        """
        self.count_type = 'rate'
        self.header = header
        self.control = control
        self.data = data
        self.energies_ = energies
        if 'detector_masks' in self.data.colnames:
            self.detector_masks = DetectorMasks(self.data['detector_masks'])
        if 'pixel_masks' in self.data.colnames:
            self.pixel_masks = PixelMasks(self.data['pixel_masks'])
        if 'energy_bin_mask' in self.control.colnames:
            self.energy_masks = EnergyMasks(self.control['energy_bin_mask'])
            self.dE = (energies['e_high'] - energies['e_low'])
            self.dE[[0, -1]] = 1 * u.keV

    @property
    def time_range(self):
        """
        A `sunpy.time.TimeRange` for the data.
        """
        return TimeRange(self.data['time'].min(), self.data['time'].max())

    @property
    def pixels(self):
        """
        A `stixpy.science.PixelMasks` object representing the pixels contained in the data
        """
        return self.pixel_masks

    @property
    def detectors(self):
        """
        A `stixpy.science.DetectorMasks` object representing the detectors contained in the data.
        """
        return self.detector_masks

    @property
    def energies(self):
        """
        A `astropy.table.Table` object representing the energies contained in the data.
        """
        return self.energies_

    @property
    def times(self):
        """
        An `astropy.time.Time` array representing the center of the observed time bins.
        """
        return self.data['time']

    def get_data(self, time_indices=None, energy_indices=None, detector_indices=None,
                 pixel_indices=None, sum_all_times=False):
        """
        Return the counts, errors, times, durations and energies for selected data.

        Optionally summing in time and or energy.

        Parameters
        ----------
        time_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `time_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `time_indices=[[0, 2],[3, 5]]` would sum the data between.
        energy_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `energy_indices=[0, 2, 5]` would return only the first, third and
            sixth times while `energy_indices=[[0, 2],[3, 5]]` would sum the data between.
        detector_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `detector_indices=[0, 2, 5]` would return only the first, third and
            sixth detectors while `detector_indices=[[0, 2],[3, 5]]` would sum the data between.
        pixel_indices : `list` or `numpy.ndarray`
            If an 1xN array will be treated as mask if 2XN array will sum data between given
            indices. For example `pixel_indices=[0, 2, 5]` would return only the first, third and
            sixth pixels while `pixel_indices=[[0, 2],[3, 5]]` would sum the data between.
        sum_all_times : `bool`
            Flag to sum all give time intervals into one

        Returns
        -------
        `tuple`
            Counts, errors, times, deltatimes,  energies

        """
        counts = self.data['counts']
        counts_var = self.data['counts_err']**2
        shape = counts.shape
        if len(shape) < 4:
            counts = counts.reshape(shape[0], 1, 1, shape[-1])
            counts_var = counts_var.reshape(shape[0], 1, 1, shape[-1])

        energies = self.energies_
        times = self.times

        if detector_indices is not None:
            detecor_indices = np.asarray(detector_indices)
            if detecor_indices.ndim == 1:
                detector_mask = np.full(32, False)
                detector_mask[detecor_indices] = True
                counts = counts[:, detector_mask, ...]
                counts_var = counts_var[:, detector_mask, ...]
            elif detecor_indices.ndim == 2:
                counts = np.hstack([np.sum(counts[:, dl:dh + 1, ...], axis=1, keepdims=True)
                                    for dl, dh in detecor_indices])

                counts_var = np.concatenate(
                    [np.sum(counts_var[:, dl:dh + 1, ...], axis=1, keepdims=True)
                     for dl, dh in detecor_indices], axis=1)

        if pixel_indices is not None:
            pixel_indices = np.asarray(pixel_indices)
            if pixel_indices.ndim == 1:
                pixel_mask = np.full(12, False)
                pixel_mask[pixel_indices] = True
                counts = counts[..., pixel_mask, :]
                counts_var = counts_var[..., pixel_mask, :]
            elif pixel_indices.ndim == 2:
                counts = np.concatenate([np.sum(counts[..., pl:ph + 1, :], axis=2, keepdims=True)
                                         for pl, ph in pixel_indices], axis=2)

                counts_var = np.concatenate(
                    [np.sum(counts_var[..., pl:ph + 1, :], axis=2, keepdims=True)
                     for pl, ph in pixel_indices], axis=2)

        e_norm = self.dE
        if energy_indices is not None:
            energy_indices = np.asarray(energy_indices)
            if energy_indices.ndim == 1:
                energy_mask = np.full(32, False)
                energy_mask[energy_indices] = True
                counts = counts[..., energy_mask]
                counts_var = counts_var[..., energy_mask]
                e_norm = self.dE[energy_mask]
                energies = self.energies_[energy_mask]
            elif energy_indices.ndim == 2:
                counts = np.concatenate([np.sum(counts[..., el:eh+1], axis=-1, keepdims=True)
                                         for el, eh in energy_indices], axis=-1)

                counts_var = np.concatenate(
                    [np.sum(counts_var[..., el:eh+1], axis=-1, keepdims=True)
                     for el, eh in energy_indices], axis=-1)

                e_norm = np.hstack([(energies['e_high'][eh] - energies['e_low'][el])
                                    for el, eh in energy_indices])

                energies = np.atleast_2d(
                    [(self.energies_['e_low'][el].value, self.energies_['e_high'][eh].value)
                     for el, eh in energy_indices])
                energies = QTable(energies*u.keV, names=['e_low', 'e_high'])

        t_norm = self.data['timedel']
        if time_indices is not None:
            time_indices = np.asarray(time_indices)
            if time_indices.ndim == 1:
                time_mask = np.full(times.shape, False)
                time_mask[time_indices] = True
                counts = counts[time_mask, ...]
                counts_var = counts_var[time_mask, ...]
                t_norm = self.data['timedel'][time_mask]
                times = times[time_mask]
                # dT = self.data['timedel'][time_mask]
            elif time_indices.ndim == 2:
                new_times = []
                dt = []
                for tl, th in time_indices:
                    ts = times[tl] - self.data['timedel'][tl] * 0.5
                    te = times[th] + self.data['timedel'][th] * 0.5
                    td = te - ts
                    tc = ts + (td * 0.5)
                    dt.append(td.to('s'))
                    new_times.append(tc)
                dt = np.hstack(dt)
                times = Time(new_times)
                counts = np.vstack([np.sum(counts[tl:th+1, ...], axis=0, keepdims=True) for tl, th
                                    in time_indices])

                counts_var = np.vstack([np.sum(counts_var[tl:th+1, ...], axis=0, keepdims=True)
                                        for tl, th in time_indices])
                t_norm = dt

                if sum_all_times and len(new_times) > 1:
                    counts = np.sum(counts, axis=0, keepdims=True)
                    counts_var = np.sum(counts_var, axis=0, keepdims=True)
                    t_norm = np.sum(dt)

        if e_norm.size != 1:
            e_norm = e_norm.reshape(1, 1, 1, -1)

        if t_norm.size != 1:
            t_norm = t_norm.reshape(-1, 1, 1, 1)

        counts_err = np.sqrt(counts*u.ct + counts_var)/(e_norm * t_norm)
        counts = counts / (e_norm * t_norm)

        return counts, counts_err, times, t_norm, energies

    def concatenate(self, others):
        """
        Concatenate two or more science products.

        Parameters
        ----------
        others: `list` [`stixpy.science.ScienceData`]
            The other/s science products to concatenate

        Returns
        -------
        `stixpy.science.ScienceData`
            The concatenated science products
        """
        others = others if isinstance(others, list) else [others]
        if all([isinstance(o, type(self)) for o in others]):
            control = self.control[:]
            data = self.data[:]
            for other in others:
                self_control_ind_max = data['control_index'].max() + 1
                other.control['index'] = other.control['index'] + self_control_ind_max
                other.data['control_index'] = other.data['control_index'] + self_control_ind_max

                try:
                    [(table.meta.pop('DATASUM'), table.meta.pop('CHECKSUM'))
                     for table in [control, other.control, data, other.data]]
                except KeyError:
                    pass

                control = vstack([control, other.control])
                data = vstack([data, other.data])

            return type(self)(header=self.header, control=control, data=data,
                              energies=self.energies)

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'{self.time_range}' \
               f'    {self.detector_masks}\n' \
               f'    {self.pixel_masks}\n' \
               f'    {self.energy_masks}'

    @classmethod
    def from_fits(cls, file):
        """
        Parses STIX FITS data files to create science products.

        Parameters
        ----------
        file : `str` or `pathlib.Path`
            The path to the file to open read.
        """

        file = Path(file)
        header = fits.getheader(file)
        control = QTable.read(file, hdu=1)
        data = QTable.read(file, hdu=2)
        data['time'] = Time(header['date_obs']) + data['time']
        energies = QTable.read(file, hdu=3)

        filename = file.name
        if 'xray-l0' in filename:
            return RawPixelData(header=header, control=control, data=data, energies=energies)
        elif 'xray-l1' in filename:
            return CompressedPixelData(header=header, control=control,
                                       data=data, energies=energies)
        elif 'xray-l2' in filename:
            return SummedCompressedPixelData(header=header, control=control,
                                             data=data, energies=energies)
        elif 'xray-l3' in filename:
            return Visibility(header=header, control=control, data=data, energies=energies)
        elif 'spectrogram' in filename:
            return Spectrogram(header=header, control=control, data=data, energies=energies)
        else:
            raise TypeError(f'File {file} does not contain pixel data')


class RawPixelData(ScienceData, PixelPlotMixin, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Uncompressed or raw count data from selected pixels, detectors and energies.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.science import ScienceData
    >>> raw_pd = ScienceData.from_fits(test.STIX_SCI_XRAY_L0)
    >>> raw_pd
    RawPixelData   <sunpy.time.timerange.TimeRange object at ...>
        Start: 2020-05-06 00:00:00
        End:   2020-05-06 00:00:16
        Center:2020-05-06 00:00:08
        Duration:0.0001851851851850661 days or
               0.0044444444444415865 hours or
               0.2666666666664952 minutes or
               15.999999999989711 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0...4]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    <BLANKLINE>
    EnergyMasks
    [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
    """
    pass


class CompressedPixelData(ScienceData, PixelPlotMixin, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Compressed count data from selected pixels, detectors and energies.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.science import ScienceData
    >>> compressed_pd = ScienceData.from_fits(test.STIX_SCI_XRAY_L1)
    >>> compressed_pd
    CompressedPixelData   <sunpy.time.timerange.TimeRange object at ...>
        Start: 2020-05-06 00:00:00
        End:   2020-05-06 00:00:16
        Center:2020-05-06 00:00:08
        Duration:0.0001851851851850661 days or
                0.0044444444444415865 hours or
                0.2666666666664952 minutes or
                15.999999999989711 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
    PixelMasks
        [0...4]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    <BLANKLINE>
        EnergyMasks
    [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    """
    pass


class SummedCompressedPixelData(ScienceData, PixelPlotMixin, TimesSeriesPlotMixin,
                                SpectrogramPlotMixin):
    """
    Compressed and Summed count data from selected pixels, detectors and energies.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.science import ScienceData
    >>> summed_pd = ScienceData.from_fits(test.STIX_SCI_XRAY_L2)
    >>> summed_pd
    SummedCompressedPixelData   <sunpy.time.timerange.TimeRange object at ...>
        Start: 2020-05-06 00:00:00
        End:   2020-05-06 00:00:16
        Center:2020-05-06 00:00:08
        Duration:0.0001851851851850661 days or
               0.0044444444444415865 hours or
               0.2666666666664952 minutes or
               15.999999999989711 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0...4]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']]]
    <BLANKLINE>
        EnergyMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    """
    pass


class Visibility(ScienceData):
    """
    Compressed visibilities from selected pixels, detectors and energies.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.science import ScienceData
    >>> visibility = ScienceData.from_fits(test.STIX_SCI_XRAY_L3)
    >>> visibility
    Visibility   <sunpy.time.timerange.TimeRange object at ...>
        Start: 2020-05-07 23:59:58
        End:   2020-05-08 00:00:14
        Center:2020-05-08 00:00:06
        Duration:0.00018518518518517713 days or
               0.004444444444444251 hours or
               0.26666666666665506 minutes or
               15.999999999999304 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
        [1]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
        [2]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
        [3]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
        [4]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0' '0.0']
     ['0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '1.0']]]
    <BLANKLINE>
        EnergyMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
    """
    def __init__(self, *, header, control, data, energies):
        super().__init__(header=header, control=control, data=data, energies=energies)
        self.pixel_masks = PixelMasks(self.pixels)

    @property
    def pixels(self):
        return np.vstack([self.data[f'pixel_mask{i}'][0] for i in range(1, 6)])


class Spectrogram(ScienceData, TimesSeriesPlotMixin, SpectrogramPlotMixin):
    """
    Spectrogram from selected pixels, detectors and energies.

    Parameters
    ----------
    header : `astropy.fits.Header`
    control : `astropy.table.QTable`
    data : `astropy.table.QTable`
    energies : `astropy.table.QTable`

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.science import ScienceData
    >>> spectogram = ScienceData.from_fits(test.STIX_SCI_XRAY_L2)
    >>> spectogram
    SummedCompressedPixelData   <sunpy.time.timerange.TimeRange object at ...>
        Start: 2020-05-06 00:00:00
        End:   2020-05-06 00:00:16
        Center:2020-05-06 00:00:08
        Duration:0.0001851851851850661 days or
               0.0044444444444415865 hours or
               0.2666666666664952 minutes or
               15.999999999989711 seconds
        DetectorMasks
        [0...4]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
        PixelMasks
        [0...4]: [[['1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']
     ['0.0' '0.0' '0.0' '1.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0' '0.0']]]
    <BLANKLINE>
        EnergyMasks
        [0]: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    <BLANKLINE>
    """
    def __init__(self, *, header, control, data, energies):
        """

        Parameters
        ----------
        header : astropy.fits.Header
        control : astropy.table.QTable
        data : astropy.table.QTable
        energies : astropy.table.QTable
        """
        super().__init__(header=header, control=control, data=data, energies=energies)
        self.count_type = 'rate'
        self.header = header
        self.control = control
        self.data = data
        self.energies_ = energies
        self.detector_masks = DetectorMasks(self.control['detector_mask'])
        self.pixel_masks = PixelMasks(self.control['pixel_mask'])
        self.energy_masks = EnergyMasks(self.control['energy_bin_mask'])
        self.dE = np.hstack([[1], np.diff(energies['e_low'][1:]).value, [1]]) * u.keV


class SliderCustomValue(Slider):
    """
    A slider with a customisable formatter
    """
    def __init__(self, *args, format_func=None, **kwargs):
        if format_func is not None:
            self._format = format_func
        super().__init__(*args, **kwargs)
