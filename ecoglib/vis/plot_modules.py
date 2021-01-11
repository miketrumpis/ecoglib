import numpy as np

# ETS
from traits.api import HasTraits, on_trait_change, Float, Int, Tuple, Range, Button

import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib import cm


class BlitPlot(HasTraits):
    """
    Provides a generic MPL plot that has static and dynamic components.
    Dynamic parts (MPL "actors") can be re-rendered quickly without
    having to re-render the entire figure.
    """

    xlim = Tuple
    ylim = Tuple

    def __init__(self, figure=None, figsize=(6, 4), axes=None, **traits):
        if figure is not None:
            self.fig = figure
        else:
            self.fig = Figure(figsize=figsize)
            # I think this doesn't need to happen (??)
            # self.fig.canvas = None
        if axes:
            if not axes in self.fig.axes:
                self.fig.add_axes(axes)
            self.ax = axes
        else:
            self.ax = self.fig.add_subplot(111)
        xlim = traits.pop('xlim', self.ax.get_xlim())
        ylim = traits.pop('ylim', self.ax.get_ylim())
        self.static_artists = []
        self.dynamic_artists = []
        self._bkgrnd = None
        self._old_size = tuple(self.ax.bbox.size)
        self._mpl_connections = []
        traits.update(dict(xlim=xlim, ylim=ylim))
        HasTraits.__init__(self, **traits)

    def add_static_artist(self, a):
        if not np.iterable(a):
            a = (a,)
        self.static_artists.extend(a)

    def add_dynamic_artist(self, a):
        if not np.iterable(a):
            a = (a,)
        self.dynamic_artists.extend(a)

    def remove_static_artist(self, a):
        if not np.iterable(a):
            a = (a,)
        for artist in a:
            artist.remove()
            self.static_artists.remove(artist)

    def remove_dynamic_artist(self, a):
        if not np.iterable(a):
            a = (a,)
        for artist in a:
            artist.remove()
            self.dynamic_artists.remove(artist)

    @on_trait_change('xlim')
    def set_xlim(self, *xlim):
        if not xlim:
            xlim = self.xlim
        else:
            self.trait_setq(xlim=xlim)
        self.ax.set_xlim(xlim)
        self.draw()

    @on_trait_change('ylim')
    def set_ylim(self, *ylim):
        if not ylim:
            ylim = self.ylim
        else:
            self.trait_setq(ylim=ylim)
        self.ax.set_ylim(ylim)
        self.draw()

    # this is a full drawing -- it saves the new background (dynamic artists
    # are set invisible before saving), and then draws the dynamic artists
    # back in
    def draw(self):
        """
        Draw full figure from scratch.
        """
        if self.fig.canvas is None:
            return
        for artist in self.dynamic_artists:
            artist.set_visible(False)
        self.fig.canvas.draw()
        # self.fig.canvas.draw_idle() # thread-safe??
        try:
            self._bkgrnd = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        except AttributeError:
            self._bkgrnd = None
        for artist in self.dynamic_artists:
            artist.set_visible(True)
        self.fig.canvas.draw()
        # self.fig.canvas.draw_idle() # thread-safe??
        try:
            self._old_size = (self.fig.canvas.width(), self.fig.canvas.height())
        except AttributeError:
            pass

    # this method only pushes out the old background, and renders the
    # dynamic artists (which have presumably changed)
    def draw_dynamic(self):
        """
        Draw only dynamic actors, then restore the background.
        """
        if self.fig.canvas is None:
            return
        # detect a resize
        new_size = (self.fig.canvas.width(), self.fig.canvas.height())
        if new_size != self._old_size:
            self.draw()
            return
        if self._bkgrnd is not None:
            self.fig.canvas.restore_region(self._bkgrnd)
        for artist in self.dynamic_artists:
            self.ax.draw_artist(artist)
        self.fig.canvas.blit(self.ax.bbox)

    def connect_live_interaction(self, extra_connections=()):
        """
        Make callback connections with the figure's canvas. The
        argument extra_connections is a sequence of (event-name, handler)
        pairs that can be provided externally
        """
        canvas = self.fig.canvas
        if canvas is None:
            print('Canvas not present, no connections made')
            return
        if self._mpl_connections:
            print('Connections already present,'
                  'may want to consider disconnecting them first')
        # standard mpl connection pattern
        connections = (('resize_event', self._resize_handler),) + \
            extra_connections
        for event, handler in connections:
            id = canvas.mpl_connect(event, handler)
            self._mpl_connections.append(id)

    def disconnect_live_interaction(self):
        canvas = self.fig.canvas
        if canvas is None:
            print('Canvas not present')
        else:
            for id in self._mpl_connections:
                canvas.mpl_disconnect(id)
        self._mpl_connections = []

    def _resize_handler(self, ev):
        # print ev.name
        pass


class LongNarrowPlot(BlitPlot):
    """
    A BlitPlot with less tick clutter on the y-axis.
    """
    n_yticks = Int(2)

    def __init__(self, *args, **traits):
        super(LongNarrowPlot, self).__init__(*args, **traits)

    @on_trait_change('n_yticks')
    def _change_yticks(self):
        self.set_ylim()

    @on_trait_change('ylim')
    def set_ylim(self, *ylim):
        if not ylim:
            ylim = self.ylim
        else:
            self.trait_setq(ylim=ylim)
        ## md = 0.5 * (ylim[0] + ylim[1])
        ## rng = 0.5 * (ylim[1] - ylim[0])
        ## y_ticks = np.linspace(md - 0.8*rng, md + 0.8*rng, self.n_yticks)
        # self.ax.yaxis.set_ticks(y_ticks)
        self.ax.yaxis.set_major_locator(
            matplotlib.ticker.LinearLocator(numticks=self.n_yticks)
        )
        super(LongNarrowPlot, self).set_ylim(*ylim)

##############################################################################
########## Classes To Define Plot Functionality ##############################


class PlotInteraction(object):
    "Basis for plot-interaction factories"
    # List of event types to associate with this object's callable
    events = ()
    strict_type = None

    @classmethod
    def gen_event_associations(cls, plot_obj, *args, **kwargs):
        if cls.strict_type and not isinstance(plot_obj, BlitPlot):
            raise ValueError(str(cls) + 'requires a' + str(cls.strict_type))
        obj = cls(plot_obj, *args, **kwargs)
        objs = [obj] * len(cls.events)
        return tuple(zip(cls.events, objs))


class MouseScrubber(PlotInteraction):
    events = ('button_press_event',
              'button_release_event',
              'motion_notify_event')


class TimeBar(MouseScrubber):

    def __init__(self, obj, link_var, sense_button=1):
        self.obj = obj
        self.link_var = link_var
        self.sense_button = sense_button
        self.scrolling = False

    def __call__(self, ev):
        if not ev.inaxes or ev.button != self.sense_button:
            return
        if not self.scrolling and ev.name == 'button_press_event':
            self.scrolling = True
            #self.time = ev.xdata
            setattr(self.obj, self.link_var, ev.xdata)
        elif ev.name == 'button_release_event':
            self.scrolling = False
        elif self.scrolling and ev.name == 'motion_notify_event':
            #self.time = ev.xdata
            setattr(self.obj, self.link_var, ev.xdata)


class AxesScrubber(MouseScrubber):
    strict_type = BlitPlot

    def __init__(
            self, obj, link_lo, link_hi, sense_button=1,
            transient=True, scrub_x=True
    ):
        self.obj = obj
        self.link_lo = link_lo
        self.link_hi = link_hi
        self.sense_button = sense_button
        self.transient = transient
        self.scrub_x = scrub_x
        self.scrolling = False
        self.pressed = False
        # code is 0 for left/down (e.g. get_xlim()[0] and 1 for right/up
        self.scr_dir = 1

    def __call__(self, ev):
        if not hasattr(ev, 'button') or ev.button != self.sense_button:
            return

        yl = self.obj.ax.get_ylim()
        xl = self.obj.ax.get_xlim()
        lims = xl if self.scrub_x else yl
        if ev.name == 'button_press_event' and ev.inaxes:
            self.scrolling = True
            self._rect_start = ev.xdata if self.scrub_x else ev.ydata
            init_width = (lims[1] - lims[0]) / 100.0
            corner = (ev.xdata, yl[0]) if self.scrub_x else (xl[0], ev.ydata)
            x_len = init_width if self.scrub_x else xl[1] - xl[0]
            y_len = yl[1] - yl[0] if self.scrub_x else init_width
            # start a rectangle patch
            self._patch = Rectangle(
                corner, x_len, y_len, fc='k', ec='k', alpha=.25
            )
            self.obj.ax.add_artist(self._patch)
            self.obj.add_dynamic_artist(self._patch)
            self.pressed = True
            self.obj.draw()

        if ev.inaxes:
            evdata = ev.xdata if self.scrub_x else ev.ydata
        else:
            evdata = lims[self.scr_dir]
        self.scr_dir = int(evdata > self._rect_start)

        if ev.name == 'motion_notify_event' and self.scrolling:
            if self.scrub_x:
                self._patch.set_width(evdata - self._rect_start)
            else:
                self._patch.set_height(evdata - self._rect_start)
            self.obj.draw_dynamic()

        if ev.name == 'button_release_event' and self.scrolling:
            self.scrolling = False
            if self.transient:
                self.obj.remove_dynamic_artist(self._patch)
                self.obj.draw()

            lo_val, hi_val = sorted([self._rect_start, evdata])
            setattr(self.obj, self.link_lo, lo_val)
            setattr(self.obj, self.link_hi, hi_val)


class StaticFunctionPlot(LongNarrowPlot):
    """
    Plain vanilla x(t) plot, with a marker for the current time.
    """
    time = Float
    interactions = [(TimeBar, 'time')]

    # Caution: mixin-only. Supposes the method
    # * create_fn_image() which produces a particular plot
    #
    # This method is provided by the ProtoPlot types (e.g. StandardPlot)

    def __init__(
            self, t, x,
            t0=None,
            mark_line_props=dict(),
            plot_line_props=dict(),
            **bplot_kws
    ):
        # just do BlitPlot with defaults -- this gives us a figure and axes
        #bplot_kws['xlim'] = (t[0], t[-1])
        bplot_kws['xlim'] = np.nanmin(t), np.nanmax(t)
        figure = bplot_kws.pop('figure', None)
        if 'ylim' not in bplot_kws:
            bplot_kws['ylim'] = (np.nanmin(x), np.nanmax(x))
        super(StaticFunctionPlot, self).__init__(figure=figure, **bplot_kws)
        if t0 is None:
            t0 = t[0]
        ts_line = self.create_fn_image(x, t=t, **plot_line_props)
        self.ax.xaxis.get_major_formatter().set_useOffset(False)
        self.add_static_artist(ts_line)
        mark_line_props['color'] = mark_line_props.get('color', 'r')
        mark_line_props['ls'] = mark_line_props.get('ls', '-')
        self.time_mark = self.ax.axvline(x=t0, **mark_line_props)
        self.add_dynamic_artist(self.time_mark)

    @on_trait_change('time')
    def move_bar(self, *time):
        if time:
            time = time[0]
            self.trait_setq(time=time)
        else:
            time = self.time
        self.time_mark.set_data(([time, time], [0, 1]))
        self.draw_dynamic()

    def connect_live_interaction(self, extra_connections=(), sense_button=1):
        # connect a sequence of callbacks to
        # click -> enable scrolling
        # drag -> scroll time bar (if scrolling)
        # release click -> disable scrolling
        # Note: this is now handled by a PlotInteraction type
        connections = TimeBar.gen_event_associations(
            self, 'time', sense_button=sense_button
        )
        connections = connections + extra_connections
        super(StaticFunctionPlot, self).connect_live_interaction(
            extra_connections=connections
        )


class WindowedFunctionPlot(StaticFunctionPlot):
    """
    Same behavior as a static plot, but the plot view flips between
    windows of set duration.
    """

    _mx_window = Int
    _window = Range(low=0, high='_mx_window', value=0)
    overlap = Float(0.15)

    next_window = Button()
    prev_window = Button()

    def __init__(self, t, x, window_length=100, **traits):
        self._tmin = t[0]
        self._tmax = t[-1]
        # if window_length is not set, then choose appropriate length
        tspan = self._tmax - self._tmin
        if tspan < window_length:
            self.window_length = 1.05 * tspan
        else:
            self.window_length = window_length
        super(WindowedFunctionPlot, self).__init__(t, x, **traits)
        self._init_windows(t)

    def _init_windows(self, t):
        # set up the number of windows and the associated limits
        # each window is staggered at (1-overlap) * window_length points
        plen = (1 - self.overlap) * self.window_length
        full_window = float(t[-1] - t[0])
        self._mx_window = max(1, int(full_window / plen))
        self._time_insensitive = False
        self.change_window(self._window)

    @on_trait_change('_window')
    def change_window(self, *window):
        if window:
            window = window[0]
            self.trait_setq(_window=window)
        else:
            window = self._window

        # the window length is set, but the window stride is
        # (1-overlap) * window_length
        stride = (1 - self.overlap) * self.window_length
        x_start = window * stride + self._tmin
        x_stop = x_start + self.window_length
        # this will triger a redraw
        self.xlim = (x_start, x_stop)
        # if a window was switched while the current time is somewhere
        # outside the current interval, then enter a mode where subsequent
        # time updates won't change the window back. This will last until
        # the current time comes back into the current window
        if self.time < x_start or self.time > x_stop:
            self._time_insensitive = True

    def _next_window_fired(self):
        self._window = min(self._mx_window, self._window + 1)

    def _prev_window_fired(self):
        self._window = max(0, self._window - 1)

    def window_from_time(self, time):
        stride = (1 - self.overlap) * self.window_length
        # (n+1)*stride > time > n*stride
        n = int(time / stride)
        return n

    @on_trait_change('time')
    def move_bar(self, *time):
        if time:
            time = time[0]
            self.trait_setq(time=time)
        else:
            time = self.time
        too_lo = self.time < self.xlim[0]
        too_hi = self.time > self.xlim[1]
        within = not too_lo and not too_hi
        if not within:
            if self._time_insensitive:
                return
            self._window = self.window_from_time(time)
        else:
            # if time is within the interval then we can always
            # be sensitive to it
            self._time_insensitive = False
        super(WindowedFunctionPlot, self).move_bar(time)

    def connect_live_interaction(self, extra_connections=()):
        # connect a right-mouse-button triggered paging
        connections = (
            ('button_press_event', self._window_handler),
        )
        connections = connections + extra_connections
        super(
            WindowedFunctionPlot, self
        ).connect_live_interaction(extra_connections=connections)

    def _window_handler(self, ev):
        if ev.button != 3 or not ev.inaxes:
            return
        x = ev.xdata
        x_lo, x_hi = self.xlim
        if np.abs(x - x_lo) < np.abs(x - x_hi):
            self.prev_window = True
        else:
            self.next_window = True


class ScrollingFunctionPlot(StaticFunctionPlot):

    winsize = Float

    def __init__(
            self, t, x, winsize, t0=None, plot_line_props=dict(), **bplot_kws
    ):
        super(ScrollingFunctionPlot, self).__init__(
            t, x, t0=t0, plot_line_props=plot_line_props, **bplot_kws
        )
        if t0 is None:
            t0 = t[0]
        self.winsize = float(winsize)
        self.move_bar(t0)

    @on_trait_change('time')
    def move_bar(self, *time):
        if time:
            time = time[0]
            self.trait_setq(time=time)
        else:
            time = self.time
        self.time_mark.set_data(([time, time], [0, 1]))
        self.xlim = (time - self.winsize / 2, time + self.winsize / 2)

    @on_trait_change('winsize')
    def change_window(self):
        time = self.time
        self.xlim = (time - self.winsize / 2, time + self.winsize / 2)


class PagedFunctionPlot(StaticFunctionPlot):

    page = Int(0)
    page_length = Int
    stack_spacing = Float(-1)

    def __init__(self, t, x, page_length, stack_traces=True, **traits):
        self.lims = (x.min(), x.max())
        self._zooming = False
        self.page_length = page_length
        self.page = 0
        self.stack_traces = stack_traces
        self.x = x
        self.t = t
        t_init, x_init = self._data_page()
        super(PagedFunctionPlot, self).__init__(t_init, x_init, **traits)
        self.trait_setq(page_length=page_length)
        # Rename the list of Line objects as traces
        self.traces = self.static_artists[:]
        self.page_in(self.page)

    def _data_page(self):
        # page with +/- 1 page length buffer
        start = (self.page - 1) * self.page_length
        data_page = safe_slice(self.x, start, 3 * self.page_length, fill=0)
        tx_page = safe_slice(self.t, start, 3 * self.page_length, fill='extend')
        if self.x.ndim > 1 and self.stack_traces:
            if self.stack_spacing < 0:
                window = data_page[self.page_length:2 * self.page_length]
                spacing = np.median(np.ptp(window, axis=0))
            else:
                spacing = self.stack_spacing
            data_page = data_page + np.arange(self.x.shape[1]) * spacing
        return tx_page, data_page

    @on_trait_change('page')
    def page_in(self, *page):
        if not hasattr(self, 'traces'):
            return
        if page:
            self.trait_setq(page=page[0])

        # all vertical scaling is controlled w/in _data_page()
        tx, data_page = self._data_page()
        for fn, line_obj in zip(data_page.T, self.traces):
            line_obj.set_data(tx, fn)
        window = data_page[self.page_length:2 * self.page_length]
        self.center_page(quiet=True)
        self.ax.set_xlim(self.xlim)
        self.ylim = (np.nanmin(window), np.nanmax(window))

    def center_page(self, t_off=0, quiet=False):
        # set axis range to the middle segment of the window
        t = self.traces[0].get_data()[0]
        twid = t[self.page_length] - t[0]
        t0 = t[int(1.5 * self.page_length)]
        t0 = t0 + t_off
        t_min = t0 - twid / 2
        t_max = t0 + twid / 2
        if quiet:
            self.trait_setq(xlim=(t_min, t_max))
        else:
            self.xlim = (t_min, t_max)

    @on_trait_change('page_length')
    def _repage(self, x, y, old, new):
        # try to maintain the start of the window
        old_start = old * self.page
        #self.trait_setq(page = old_start // new)
        self.page_in(old_start // new)

    @on_trait_change('stack_spacing')
    def _change_stack(self):
        self.page_in()

    @property
    def current_spacing(self):
        if self.x.ndim < 2:
            return 0
        if self.stack_spacing:
            return self.stack_spacing
        ylim = self.ylim
        appx_spacing = (ylim[1] - ylim[0]) / (self.x.shape[1] - 1)
        return appx_spacing

    # def connect_live_interaction(self, *extra_connections):
    # connect a right-mouse-button triggered paging
    # connections = (
    ##         ('button_press_event', self._zoom_handler),
    ##         ('motion_notify_event', self._zoom_handler),
    ##         ('button_release_event', self._zoom_ender)
    # )
    ##     connections = connections + extra_connections
    # super(
    ##         PagedFunctionPlot, self
    # ).connect_live_interaction(*connections)

    # def _zoom_handler(self, ev):
    # if ev.button != 3 or not ev.inaxes:
    # return
    # if not self._zooming:
    # start zooming
    ##         self._saved_lims = self.xlim + self.ylim
    ##         self._x0 = (ev.xdata, ev.ydata)
    # print 'saving state:', self._saved_lims, self._x0
    ##         self._zooming = True
    # return
    ##     x0, y0 = self._x0
    ##     dx = 4*float(ev.xdata - x0)
    ##     dy = 4*float(ev.ydata - y0)
    ##     x_span = self._saved_lims[1] - self._saved_lims[0]
    ##     y_span = self._saved_lims[3] - self._saved_lims[2]
    # print dx, dy
    # apply a saturating curve to the mouse motion
    ##     new_y_span = y_span*(np.pi/2 - np.arctan(dy/y_span))
    ##     new_x_span = x_span*(np.pi/2 - np.arctan(dx/x_span))
    ##     print (0.5 - np.arctan(dy/y_span)/np.pi),
    ##     print (0.5 - np.arctan(dx/x_span)/np.pi)
    # print (y0 - new_y_span/2, y0 + new_y_span/2),
    # print (x0 - new_x_span/2, x0 + new_x_span/2)
    ##     self.ylim = (y0 - new_y_span/2, y0 + new_y_span/2)
    # self.trait_setq(ylim = (y0 - new_y_span/2, y0 + new_y_span/2))
    # self.trait_setq(xlim = (x0 - new_x_span/2, x0 + new_x_span/2))

    # def _zoom_ender(self, ev):
    # if ev.button != 3:
    # return
    # print 'ending zoom state'
    ##     self._zooming = False
    ##     self.xlim = self._saved_lims[:2]
    ##     self.ylim = self._saved_lims[2:]
##############################################################################
########## Classes To Define Plot Styles #####################################


class ProtoPlot(object):
    """
    An abstract prototype of the Plot types
    """

    def create_fn_image(self, *args, **kwargs):
        raise NotImplementedError('Abstract class: does not plot')


class StandardPlot(ProtoPlot):
    """
    A mixin type that plots a simple times series line.
    """

    # Caution: mixin-only. Supposes the attribute
    # * ax, an MPL Axes

    # this signature should be pretty generic
    def create_fn_image(self, x, t=None, **plot_line_props):
        # in this case, just a plot
        if t is None:
            t = np.arange(x.shape[0])
        lines = self.ax.plot(t, x, **plot_line_props)
        return lines


class MaskedPlot(StandardPlot):
    """
    A modified mixin that applies a "mask" to certain traces
    by plotting them with a masking color
    """

    mask_color = '#E0E0E0'

    def create_fn_image(self, x, t=None, **plot_line_props):
        channel_mask = plot_line_props.pop('channel_mask', None)
        lines = super(MaskedPlot, self).create_fn_image(
            x, t=t, **plot_line_props
        )
        if channel_mask is None or not len(channel_mask):
            return lines
        channel_mask = channel_mask.astype('?')
        for i in np.where(~channel_mask)[0]:
            lines[i].set_color(self.mask_color)
        return lines


class ColorCodedPlot(ProtoPlot):
    """
    A mixin-type whose image is a color-coded time series lines
    """

    # Caution: mixin-only. Supposes the attributes
    # * ax, an MPL Axes
    # * cx, a color-coding function
    # * cx_limits, the (possibly clipped) dynamic range of "cx"

    def create_fn_image(self, x, t=None, **plot_line_props):
        if t is None:
            t = np.arange(len(x))
        if not hasattr(self, 'cx'):
            raise RuntimeError(
                'Object should have been instantiated with color code'
            )
        cx = self.cx
        if not self.cx_limits:
            # try 95% confidence, so that hi/lo clipping is more likely
            #eps = stochastic_limits(cx, conf=95.0)
            mx = cx.max()
            mn = cx.min()
            limits = (mn, mx)
        else:
            limits = self.cx_limits
        #self.norm = pp.normalize(vmin=limits[0], vmax=limits[1])
        norm = Normalize(vmin=limits[0], vmax=limits[1])
        cmap = plot_line_props.pop('cmap', cm.jet)
        #colors = cmap( norm(cx) )
        plot_line_props['edgecolors'] = 'none'
        pc = self.ax.scatter(
            t, x, 14.0, c=cx, norm=norm, cmap=cmap, **plot_line_props
        )
        return pc


class ClassSegmentedPlot(ProtoPlot):
    """
    A mixin-type whose timeseries image is segmented into k classes
    """
    # e.g. for a line-plot, can
    # Caution: mixin-only. Supposes the attributes
    # * ax, an MPL Axes
    # * labels, a segmentation map
    # * n_classes -- total number of classes (included potentially
    #                outside this plot)

    def create_fn_image(self, x, t=None, labels=None, **plot_line_props):
        # Caution! Returns a sequence of separate lines for each data
        # segment
        #cmap = plot_line_props.pop('cmap', cm.jet)
        cmap = plot_line_props.pop('cmap', cm.hsv)
        if t is None:
            t = np.arange(len(x))
        if not hasattr(self, 'labels'):
            raise RuntimeError(
                'Object should have been instantiated with color code'
            )
        if labels is None:
            labels = self.labels
        # for each class sequentially fill out-of-class pts with nan,
        # and line plot in-class points with the appropriate color
        seg_line = np.empty_like(x)
        seg_lines = []
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        label_to_idx = dict(zip(unique_labels, range(n_labels)))
        if 0 in unique_labels:
            colors = cmap(np.linspace(0, 1, n_labels - 1))
            colors = np.row_stack(([0, 0, 0, 0.15], colors))
        else:
            colors = cmap(np.linspace(0, 1, n_labels))
        for seg in unique_labels:
            # -1 codes for out of bounds
            if seg < 0:
                continue
            # for seg, color in zip(xrange(mx_label+1), colors):
            seg_line.fill(np.nan)
            idx = np.where(labels == seg)[0]
            np.put(seg_line, idx, np.take(x, idx))
            cidx = label_to_idx[seg]
            color = colors[cidx]
            # if seg==0:
            ##     color[-1] = 0.15
            line = self.ax.plot(t, seg_line, c=color, **plot_line_props)
            seg_lines.extend(line)
        return seg_lines

##############################################################################
########## Classes Implementing Function and Style Combinations ##############


class StaticTimeSeriesPlot(StaticFunctionPlot, StandardPlot):
    """
    A static plot of a simple 1D timeseries graph.
    """
    # do defaults for both classes
    pass


class StaticColorCodedPlot(StaticFunctionPlot, ColorCodedPlot):
    """
    A static plot, but points are color-coded by a co-function c(t)
    """

    def __init__(
            self, t, x, cx, t0=None, cx_limits=(),
            plot_line_props=dict(), **bplot_kws
    ):
        self.cx = cx
        if not cx_limits:
            #eps = stochastic_limits(cx, conf=95.)
            mx = cx.max()
            mn = cx.min()
            cx_limits = (mn, mx)
        self.cx_limits = cx_limits
        super(StaticColorCodedPlot, self).__init__(
            t, x, t0=t0, plot_line_props=plot_line_props, **bplot_kws
        )


class StaticSegmentedPlot(StaticFunctionPlot, ClassSegmentedPlot):
    """
    A static plot, but the line plot is colored by k different classes.
    This is distinct from a ColorCodedPlot in that there are only
    a low number of discrete classes.
    """

    def __init__(
            self, t, x, labels, t0=None, plot_line_props=dict(), **bplot_kws
    ):
        self.labels = labels
        unique_labels = np.unique(labels)
        self.n_classes = len(unique_labels >= 0)
        super(StaticSegmentedPlot, self).__init__(
            t, x, t0=t0, plot_line_props=plot_line_props, **bplot_kws
        )


class WindowedTimeSeriesPlot(WindowedFunctionPlot, StandardPlot):
    """
    A static plot that is flipped between windows.
    """
    # defaults for both classes
    pass


class PagedTimeSeriesPlot(PagedFunctionPlot, MaskedPlot):
    """
    A static plot that is flipped between windows.
    """
    # defaults for both classes
    pass


class WindowedColorCodedPlot(WindowedFunctionPlot, ColorCodedPlot):
    """
    A static color-coded plot that is flipped between intervals
    """

    def __init__(
            self, t, x, cx, t0=None, cx_limits=(),
            plot_line_props=dict(), **bplot_kws
    ):
        self.cx = cx
        if not cx_limits:
            #eps = stochastic_limits(cx, conf=95.)
            mx = cx.max()
            mn = cx.min()
            cx_limits = (mn, mx)
        self.cx_limits = cx_limits
        super(WindowedColorCodedPlot, self).__init__(
            t, x, t0=t0, plot_line_props=plot_line_props, **bplot_kws
        )


class WindowedClassSegmentedPlot(WindowedFunctionPlot, ClassSegmentedPlot):
    """
    A static class-colored plot that is flipped between intervals
    """

    def __init__(
            self, t, x, labels, t0=None, plot_line_props=dict(), **bplot_kws
    ):
        self.labels = labels
        unique_labels = np.unique(labels)
        self.n_classes = len(unique_labels >= 0)
        super(WindowedClassSegmentedPlot, self).__init__(
            t, x, t0=t0, plot_line_props=plot_line_props, **bplot_kws
        )


class ScrollingTimeSeriesPlot(ScrollingFunctionPlot, StandardPlot):
    """
    A scrolling plot of a simple 1D time series graph.
    """
    # do defaults for both classes
    pass


class ScrollingColorCodedPlot(ScrollingFunctionPlot, ColorCodedPlot):
    """
    A scrolling plot, but points are color-coded by a co-function c(t)
    """

    def __init__(
            self, t, x, winsize, cx, t0=None, cx_limits=(),
            plot_line_props=dict(), **bplot_kws
    ):
        # make sure to set the color code first
        self.cx = cx
        if not cx_limits:
            mx = cx.max()
            mn = cx.min()
            cx_limits = (mn, mx)
        self.cx_limits = cx_limits
        super(ScrollingColorCodedPlot, self).__init__(
            t, x, winsize, t0=t0, plot_line_props=plot_line_props, **bplot_kws
        )


class ScrollingClassSegmentedPlot(ScrollingFunctionPlot, ClassSegmentedPlot):
    """
    A scrolling plot, but points are color-coded by class.
    """

    def __init__(
            self, t, x, winsize, labels,
            plot_line_props=dict(), **bplot_kws
    ):
        # make sure to set the class code first
        self.labels = labels
        unique_labels = np.unique(labels)
        self.n_classes = len(unique_labels >= 0)
        # hold onto these
        self._lprops = plot_line_props
        super(ScrollingClassSegmentedPlot, self).__init__(
            t, x, winsize, plot_line_props=plot_line_props, **bplot_kws
        )


def safe_slice(x, start, num, fill=np.nan):
    """
    Slice array x contiguously (along 1st dimension) for num pts,
    starting from start. If all or part of the range lies outside
    of the actual bounds of x, then fill with NaN
    """
    lx = x.shape[0]
    sub_shape = (num,) + x.shape[1:]
    if start < 0 or start + num > lx:
        sx = np.empty(sub_shape, dtype=x.dtype)
        if start <= -num or start >= lx:
            sx.fill(fill)
            # range is entirely outside
            return sx
        if start < 0 and start + num > lx:
            # valid data is in the middle of the range
            if fill == 'extend':
                # only makes sense for regularly spaced pts
                dx = x[1] - x[0]
                bwd = np.arange(1, -start + 1)
                sx[:-start, ...] = x[0] - bwd[::-1] * dx
                fwd = np.arange(1, num - lx + start + 1)
                sx[-start + lx:num, ...] = x[-1] + fwd * dx
            else:
                sx.fill(fill)
            sx[-start:-start + lx] = x
            return sx
        if start < 0:
            if fill == 'extend':
                # extend time back
                sx[:-start, ...] = x[0] - x[1:-start + 1][::-1]
            else:
                # fill beginning ( where i < 0 ) with NaN
                sx[:-start, ...] = fill

            # fill the rest with x
            sx[-start:, ...] = x[:(num + start), ...]
        else:
            sx[:(lx - start), ...] = x[start:, ...]
            if fill == 'extend':
                # extend range with this many points
                n_fill = num - (lx - start)
                x_template = x[1:n_fill + 1] - x[0]
                sx[(lx - start):, ...] = x[-1] + x_template
            else:
                sx[(lx - start):, ...] = fill
    else:
        sx = x[start:start + num, ...]
    return sx
