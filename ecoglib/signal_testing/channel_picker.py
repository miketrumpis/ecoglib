from time import ctime
import numpy as np
from traits.api import Float, on_trait_change, Instance, Str, Button, Bool
from traitsui.api import Item, View, UItem, HGroup, VGroup, HSplit, VSplit
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from ecogdata.devices.maskdb import MaskDB
from ecogdata.datasource import MappedSource
from ecoglib.vis.plot_modules import BlitPlot, PagedTimeSeriesPlot, AxesScrubber
from ecoglib.vis.gui_tools import ArrayMap
from ecoglib.vis.data_scroll import ChannelScroller
from ecoglib.vis.traitsui_bridge import MPLFigureEditor, PingPongStartup
from ecoglib.vis.plot_util import light_boxplot
from .signal_tools import safe_avg_power, bad_channel_mask
import seaborn as sns


sns.reset_orig()


__all__ = ['interactive_mask', 'ChannelPicker']

_IN_COLOR = '#5280FF'
_OUT_COLOR = '#FFD152'


def interactive_mask(dataset, scroll_len=5, cancel_is_empty=False, **kwargs):
    if ChannelScroller is None:
        print('The ChannelScroller GUI is not available')
        return np.ones(len(dataset.data), '?')
    scr = ChannelPicker(dataset, scroll_len, **kwargs)
    v = scr.default_traits_view()
    v.buttons = ['OK', 'Cancel']
    v.kind = 'livemodal'
    # foo = scr.configure_traits(view=v)
    foo = scr.edit_traits(view=v)
    if foo:
        return scr.chan_mask
    else:
        if cancel_is_empty:
            return ()
        return np.ones_like(scr.chan_mask)


class InteractiveBoxplot(BlitPlot):
    val_hi = Float(0)
    val_lo = Float(0)

    def __init__(self, vals, axes=None, **boxplot_kw):
        if axes is None:
            super(InteractiveBoxplot, self).__init__()
            axes = self.ax
        else:
            super(InteractiveBoxplot, self).__init__(
                figure=axes.figure, axes=axes
            )

        self.vals = vals
        light_boxplot([vals], ax=self.ax, **boxplot_kw)
        self._horiz = boxplot_kw.get('horiz', False)
        scale_axis = 'x' if self._horiz else 'y'
        self.ax.autoscale(axis=scale_axis)
        self.trait_setq(ylim=self.ax.get_ylim())
        # self.xlim = self.ax.get_xlim()
        self.trait_setq(xlim=self.ax.get_xlim())
        sns.despine(ax=self.ax)
        self.fig.subplots_adjust(left=0.2, right=0.95)
        self.draw()
        self.add_static_artist(self.ax.lines)
        self.add_static_artist(self.ax.patches)

    ##     self._map_lines(vals, h)

    ## def _map_lines(self, orig_vals, horiz):
    ##     # find the map from the given order of the
    ##     # original values to the boxplot elements
    ##     self._map = dict()
    ##     value_points = list()
    ##     values = orig_vals.tolist()
    ##     for ln in self.ax.lines:
    ##         d = ln.get_data()
    ##         d_ord = d[0] if horiz else d[1]
    ##         d_abs = d[1] if horiz else d[0]
    ##         if (len( np.unique(d_ord) ) == 1) and len( d_abs ) == 2:
    ##             print 'caught level mark'
    ##             continue

    def clear_rectangles(self):
        self.remove_dynamic_artist(self.dynamic_artists[:])
        self.draw_dynamic()

    def connect_live_interaction(self, extra_connections=(), sense_button=1, transient=True):
        # connect a sequence of callbacks to
        # click -> enable scrolling
        # drag -> scroll time bar (if scrolling)
        # release click -> disable scrolling
        # Note: this is now handled by a PlotInteraction type
        connections = AxesScrubber.gen_event_associations(
            self, 'val_lo', 'val_hi', sense_button=sense_button,
            scrub_x=self._horiz, transient=transient
        )
        connections = connections + extra_connections
        super(InteractiveBoxplot, self).connect_live_interaction(
            extra_connections=connections
        )


if ChannelScroller is None:
    ChannelPicker = None
else:
    class ChannelPicker(ChannelScroller):
        """
        Stripped down ChannelScroller with only the paged timeseries.
        """

        clear_array = Button('Clear channel mask')
        # TODO -- clear all time patches
        clear_time = Button('Clear time mask')
        auto_mask = Button('Auto channel mask')

        ts_plot = Instance(PagedTimeSeriesPlot)
        ts_fig = Instance(Figure)

        array_fig = Instance(ArrayMap)
        # TODO -- color-coded boxplot of RMS values, horizontal plot preferred
        box_fig = Instance(Figure)
        rms_plot = Instance(InteractiveBoxplot)

        dset_name = Str
        _save_status = Str
        use_db = Bool(True)
        save_masks = Button('Save masks')
        load_masks = Button('Load masks')
        overwrite_db = Bool(True)

        def __init__(self, dataset, page_len, mask_db=None, **traits_n_kw):
            """
            This object requires an actual dataset Bunch with the following
            standard-ish attributes:

            * data (2d channel x time data block)
            * chan_map
            * Fs
            * units
            * name (in "session.recording" format)

            """
            # interactivity book-keeping
            self.__alive = False
            self._sense_pick = True
            self._dragging = False
            self._rectangles = dict()
            # make chans appear to be empty -- this skips some weird re-indexing
            self.chan_map = dataset.chan_map
            # TODO: data access in the ChannelScroller type is deeply tied to the ndarray.. needs future decoupling
            if isinstance(dataset.data, MappedSource):
                # For now just grab a bit of data
                mx_pts = 10e6
                t = int(mx_pts // dataset.data.shape[0])
                array_data = dataset.data[:, :t]
            else:
                array_data = dataset.data.data_buffer
            ChannelScroller.__init__(
                self, array_data, page_len,
                Fs=dataset.Fs, units=dataset.units, **traits_n_kw
            )

            # book-keeping for channel and time masks
            self.chan_mask = np.ones(len(self.chan_map), dtype='?')
            self._existing_rms_mask = np.ones_like(self.chan_mask)
            self.time_mask = np.ones(dataset.data.shape[-1], dtype='?')
            self.dset_name = dataset.name
            self.ts_fig = self.ts_plot.fig
            # a provisional channel mask gets set when computing the RMS values (in case there are any NaNs)
            self._init_boxplot()
            # then the initial channel map is drawn with the provisional mask set
            self._init_mask_plot()
            # Status (are masks for this dataset already stored?)
            if self.use_db:
                self._dbman = MaskDB(dbfile=mask_db)
                node = self._dbman.lookup(self.dset_name)
                if len(node):
                    self._save_status = 'Masks exist DB'
                else:
                    self._save_status = 'No masks in DB'
            else:
                self._dbman = None

        def _init_mask_plot(self):
            self.array_fig = ArrayMap(self.chan_map, mark_site=False)
            cb = self.array_fig.cbar

            # self.array_fig = Figure(figsize=(3,3))
            # self.array_ax = self.array_fig.add_subplot(111)
            # _, cb = self.chan_map.image(arr=self.chan_mask.astype('d'), ax=self.array_ax, cmap='binary', clim=(0, 1))
            cb.set_ticks([0.25, 0.75])
            cb.set_ticklabels(['Masked', 'Unmasked'])

        def _init_boxplot(self):
            self.box_fig = Figure(figsize=(3, 2))
            self.box_ax = self.box_fig.add_subplot(111)
            self._rms_values = safe_avg_power(self.ts_plot.x.T, self.page_length, iqr_thresh=15)
            # if any channels were all zero, then rms returns NaN
            self._existing_rms_mask = ~np.isnan(self._rms_values)
            self.chan_mask = self._existing_rms_mask.copy()
            self.box_fig.canvas = None
            self.rms_plot = InteractiveBoxplot(
                self._rms_values, axes=self.box_ax, names=['RMS Voltage'], horiz=True,
                box_ls='solid'
            )

        # assume rms_plot.val_lo and rms_plot.val_hi always change at the same time
        @on_trait_change('rms_plot.val_hi')
        def _set_from_rmsplot(self):
            if not self.__alive:
                return
            plot = self.rms_plot
            lo = plot.val_lo
            hi = plot.val_hi
            m = (plot.vals > lo) & (plot.vals < hi)
            self._existing_rms_mask &= ~m
            self.chan_mask = self._existing_rms_mask
            self._mask_changed()

        def construct_ts_plot(self, *args, **kwargs):
            kwargs['color'] = _IN_COLOR
            plot = super(ChannelPicker, self).construct_ts_plot(*args, **kwargs)
            ii, jj = self.chan_map.to_mat()
            ## #jj, ii = zip(*sorted(zip(jj, ii)))
            ## ii, jj = zip(*sorted(zip(ii, jj)))
            c_num = self.chan_map.lookup(ii, jj)
            plot.ax.set_yticklabels(
                ['%d: (%d, %d)' % x for x in zip(c_num, ii, jj)],
                fontsize=8
            )
            self._line_to_chan = dict(zip(plot.traces, c_num))
            self._chan_to_line = dict(zip(c_num, plot.traces))
            return plot

        def _log_times(self, ts, tf, patch):
            i1 = int(round(ts * self.Fs))
            i2 = int(round(ts * self.Fs))
            self.time_mask[i1:i2] = False
            self._rectangles[(ts, tf)] = patch

        def _masked_patches(self, t, tf=None):
            # returns rectangle patches whose range contains t,
            # or that are included between the interval (t, tf)
            if tf is None:
                # looking for single rectangle containing t
                for itvl, patch in self._rectangles.items():
                    if t > itvl[0] and t < itvl[1]:
                        return patch
            patches = list()

            # catch everything from closing edge of earliest patch
            # to opening edge of latest patch
            def closes_within(i1, i2):
                return (i1 < t) and (i2 > t)

            def opens_within(i1, i2):
                return (i1 < tf) and (i2 > tf)

            def all_within(i1, i2):
                return (i1 > t) and (i2 < tf)

            for itvl, patch in self._rectangles.items():
                if opens_within(*itvl) or closes_within(*itvl) or all_within(*itvl):
                    patches.append(patch)
            return patches

        def _post_canvas_hook(self):
            import qtpy.QtCore as QtCore
            # connect time-marker manipulation with right mouse button
            con = (('key_press_event', self._key_event),
                   ('key_release_event', self._key_event),
                   ('button_press_event', self._rect_event),
                   ('button_release_event', self._rect_event),
                   ('motion_notify_event', self._rect_event))
            self.ts_plot.connect_live_interaction(extra_connections=con,
                                                  sense_button=3)
            self.ts_plot.fig.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
            self.ts_plot.fig.canvas.setFocus()
            self.rms_plot.connect_live_interaction(sense_button=1, transient=False)
            self._set_picker_event()
            self.__alive = True
            self.array_fig.fig.canvas.mpl_connect('button_press_event', self.array_fig.click_listen)

        def _key_event(self, ev):
            if ev.name == 'key_press_event' and ev.key.lower() == 'shift':
                self._sense_pick = False
            elif ev.name == 'key_release_event' and ev.key.lower() == 'shift':
                self._sense_pick = True

        def _rect_event(self, ev):
            if self._sense_pick:
                return
            if not ev.inaxes or ev.button != 1:
                return
            if ev.name == 'button_press_event':
                self._dragging = True

                self._rect_start = ev.xdata
                yl = self.ts_plot.ylim
                xl = self.ts_plot.xlim
                dx = (xl[1] - xl[0]) / 100.0
                # start a rectangle patch
                self._active_patch = Rectangle(
                    (ev.xdata, yl[0]), dx, yl[1] - yl[0],
                    fc='k', ec='k', alpha=.25
                )
                self.ts_plot.ax.add_artist(self._active_patch)
                self.ts_plot.add_dynamic_artist(self._active_patch)
                self.ts_plot.draw()
                # self._rectangles.append(patch)

            if ev.name == 'motion_notify_event' and self._dragging:
                self._active_patch.set_width(ev.xdata - self._rect_start)
                self.ts_plot.draw_dynamic()

            if ev.name == 'button_release_event' and self._dragging:
                self._dragging = False
                self.ts_plot.remove_dynamic_artist(self._active_patch)
                self.ts_plot.ax.add_artist(self._active_patch)
                self.ts_plot.add_static_artist(self._active_patch)
                self._log_times(self._rect_start, ev.xdata, self._active_patch)

        def _set_picker_event(self):
            vscale = self.ts_plot.ax.transData.get_matrix()[1, 1]
            for line in self.ts_plot.traces:
                dyn_range = line.get_data()[1].ptp()
                pr = max(0.5, min(1.0 / (dyn_range * vscale), 2.5))
                line.set_picker(True)
                line.set_pickradius(pr)
            self.ts_plot.fig.canvas.mpl_connect('pick_event', self._picked_channel)

        def _mask_changed(self):
            p_color = {False: _OUT_COLOR, True: _IN_COLOR}
            for i, m in enumerate(self.chan_mask):
                self._chan_to_line[i].set_color(p_color[m])
            # img = self.chan_map.embed(self.chan_mask.astype('d'))
            # self.array_ax.images[0].set_array(img)
            self.array_fig.update_map(self.chan_mask.astype('d'))
            # self.array_fig.canvas.draw()
            self.ts_plot.draw()

        def _picked_channel(self, event):
            if not self._sense_pick:
                return
            artist = event.artist
            if artist in self.ts_plot.traces:
                # find out channel number and toggle current status
                idx = self.ts_plot.traces.index(artist)
                text = self.ts_plot.ax.get_yticklabels()[idx].get_text()
                chan = int(text.split(':')[0])
                m = ~self.chan_mask[chan]
                self.chan_mask[chan] = m
                self._mask_changed()

        @on_trait_change('array_fig.selected_site')
        def _toggle_site_from_array_fig(self):
            if self.array_fig.selected_site >= 0:
                site = self.array_fig.selected_site
                self.chan_mask[site] = not self.chan_mask[site]
            self._mask_changed()

        def _clear_array_fired(self):
            self._existing_rms_mask = ~np.isnan(self._rms_values)
            self.chan_mask = self._existing_rms_mask.copy()
            self.rms_plot.clear_rectangles()
            self._mask_changed()

        def _auto_mask_fired(self):
            rms = safe_avg_power(self.ts_plot.x.T, self.page_length, iqr_thresh=15)
            lrms = np.log(rms)
            m1 = bad_channel_mask(lrms, iqr=3)
            nc_ = len(m1)
            nc = m1.sum()
            t = 0.2
            while nc != nc_:
                nc_ = nc
                m1[m1] = bad_channel_mask(lrms[m1], iqr=3 + t)
                t += 0.2
                nc = m1.sum()
                if nc == 0:
                    break

            self.chan_mask = m1
            self._mask_changed()

        def _save_masks_fired(self):
            if not self.use_db:
                return
            try:
                self._dbman.stash(
                    self.dset_name, chan_mask=self.chan_mask,
                    time_mask=self.time_mask, overwrite=self.overwrite_db
                )
                self._save_status = 'New masks saved ' + ctime()
            except RuntimeError:
                self._save_status = 'Error: existing masks blocked stash'

        def _load_masks_fired(self):
            if not self.use_db:
                return
            masks = self._dbman.lookup(self.dset_name)
            u = False
            if 'chan_mask' in masks:
                self.chan_mask = masks.chan_mask
                u = True
            if 'time_mask' in masks:
                self.time_mask = masks.time_mask
                # u = True (nothing to do yet)
            if u:
                self._mask_changed()
                self._save_status = 'Masks loaded from DB'

        def undraw_events(self):
            # remove rectangle patches here
            xl = self.ts_plot.xlim
            rects = self._masked_patches(xl[0], xl[1])
            self.ts_plot.remove_static_artist(rects)
            super(ChannelPicker, self).undraw_events()

        def draw_events(self):
            # need to re-draw any masked windows here
            xl = self.ts_plot.xlim
            rects = self._masked_patches(xl[0], xl[1])
            for r in rects:
                self.ts_plot.ax.add_artist(r)
            self.ts_plot.add_static_artist(rects)
            super(ChannelPicker, self).draw_events()
            self.ts_plot.draw()

        # Gobble some callbacks
        @on_trait_change('eps')
        def _update_eps(self):
            full_limits = max(abs(self.min_ts_amp), abs(self.max_ts_amp))
            lim = self._map_eps(self.eps, (-full_limits, full_limits))
            self.ts_plot.ylim = lim

        @classmethod
        def from_dataset_bunch(cls, dset, window):
            # simple wrapper, but need to overload parent's
            scr = cls(dset, window)
            return scr

        def default_traits_view(self):
            view = View(
                HSplit(
                    Item(
                        'ts_fig', editor=MPLFigureEditor(),
                        show_label=False, width=700, height=900
                    ),
                    VSplit(
                        Item(
                            'array_fig', editor=MPLFigureEditor(),
                            height=300, width=300, show_label=False
                        ),
                        Item(
                            'box_fig', editor=MPLFigureEditor(),
                            height=200, width=300, show_label=False
                        ),
                        HGroup(
                            UItem('auto_mask'),
                            UItem('clear_array'),
                            UItem('clear_time')
                        ),
                        VGroup(
                            HGroup(UItem('save_masks'), UItem('load_masks'),
                                   Item('overwrite_db', label='Overwrite DB entries')),
                            HGroup(UItem('dset_name', style='readonly'),
                                   Item('_save_status',
                                        label='Mask DB status', style='readonly')
                                   ),
                            visible_when='use_db==True'
                        ),
                        HGroup(
                            VGroup(
                                Item('page_up', label='Page FWD', show_label=False),
                                Item('page_dn', label='Page BWD', show_label=False),
                                HGroup(
                                    Item(
                                        'auto_scale', label='Auto Scale',
                                        show_label=False
                                    ),
                                    Item(
                                        'channel_scale', label='Scale',
                                        show_label=False, enabled_when='not auto_scale'
                                    )
                                )
                            ),
                            VGroup(
                                Item('page', label='Page Num'),
                                Item('page_length', label='Page Len'),
                                Item('window_shift', label='Shift Win')
                            )
                        )
                    )
                ),
                resizable=True, title='ChannelPicker',
                handler=PingPongStartup()
            )
            return view

if __name__ == '__main__':
    from ecogdata.devices.data_util import load_experiment_auto

    d = load_experiment_auto('viventi/2019-04-30', '2019-04-30_18-16-55_031', bandpass=(2, 100))
    scr = ChannelPicker.from_dataset_bunch(d, 5)
    scr.configure_traits()
