import numpy as np

# ETS
from traits.api import \
     HasTraits, on_trait_change, Float, Int, Tuple, Range, Button

import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib import cm

from . import plot_tools as pt

# XXX: an incomplete decorator for the listen/set pattern of traits callbacks
## def set_or_listen(attr):

##     @on_trait_change(attr)
##     def

##############################################################################
########## Simple Figure Wrapper #############################################

class BlitPlot(HasTraits):
    """
    Provides a generic MPL plot that has static and dynamic components.
    Dynamic parts (MPL "actors") can be re-rendered quickly without
    having to re-render the entire figure.
    """

    xlim = Tuple
    ylim = Tuple

    def __init__(self, figure=None, figsize=(6,4), axes=None, **traits):
        self.fig = figure or Figure(figsize=figsize)
        if axes:
            if not axes in self.fig.axes:
                self.fig.add_axes(axes)
            self.ax = axes
        else:
            self.ax = self.fig.add_axes([.15, .12, .8, .85])
        xlim = traits.pop('xlim', self.ax.get_xlim())
        ylim = traits.pop('ylim', self.ax.get_ylim())
        self.static_artists = []
        self.dynamic_artists = []
        self._bkgrnd = None
        self._old_size = tuple(self.ax.bbox.size)
        self._mpl_connections = []
        traits = dict(xlim=xlim, ylim=ylim)
        #super(BlitPlot, self).__init__(**traits)
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
        #self.fig.canvas.draw_idle() # thread-safe??
        self._bkgrnd = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        for artist in self.dynamic_artists:
            artist.set_visible(True)
        self.fig.canvas.draw()
        #self.fig.canvas.draw_idle() # thread-safe??
        self._old_size = (self.fig.canvas.width(), self.fig.canvas.height())

    # this method only pushes out the old background, and renders the
    # dynamic artists (which have presumably changed)
    def draw_dynamic(self):
        """
        Draw only dynamic actors, then restore the background.
        """
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

    def connect_live_interaction(self, *extra_connections):
        """
        Make callback connections with the figure's canvas. The
        argument extra_connections is a sequence of (event-name, handler)
        pairs that can be provided externally
        """
        canvas = self.fig.canvas
        if canvas is None:
            print 'Canvas not present, no connections made'
            return
        if self._mpl_connections:
            print 'Connections already present,'\
              'may want to consider disconnecting them first'
        #standard mpl connection pattern
        connections = (('resize_event', self._resize_handler),) + \
          extra_connections
        for event, handler in connections:
            id = canvas.mpl_connect(event, handler)
            self._mpl_connections.append( id )

    def disconnect_live_interaction(self):
        canvas = self.fig.canvas
        if canvas is None:
            print 'Canvas not present'
        else:
            for id in self._mpl_connections:
                canvas.mpl_disconnect(id)
        self._mpl_connections = []

    def _resize_handler(self, ev):
        #print ev.name
        pass

class LongNarrowPlot(BlitPlot):
    """
    A BlitPlot with less tick clutter on the y-axis.
    """
    n_yticks = Int(2)

    def __init__(self, *args, **traits):
        super(LongNarrowPlot, self).__init__(*args, **traits)
    
    @on_trait_change('ylim')
    def set_ylim(self, *ylim):
        if not ylim:
            ylim = self.ylim
        else:
            self.trait_setq(ylim=ylim)
        ## md = 0.5 * (ylim[0] + ylim[1])
        ## rng = 0.5 * (ylim[1] - ylim[0])
        ## y_ticks = np.linspace(md - 0.8*rng, md + 0.8*rng, self.n_yticks)
        ## self.ax.yaxis.set_ticks(y_ticks)
        self.ax.yaxis.set_major_locator(
            matplotlib.ticker.LinearLocator(numticks=self.n_yticks)
            )
        super(LongNarrowPlot, self).set_ylim(*ylim)

##############################################################################
########## Classes To Define Plot Functionality ##############################

class StaticFunctionPlot(LongNarrowPlot):
    """
    Plain vanilla x(t) plot, with a marker for the current time.
    """
    time = Float

    # Caution: mixin-only. Supposes the method
    # * create_fn_image() which produces a particular plot
    #
    # This method is provided by the ProtoPlot types (e.g. StandardPlot)

    def __init__(
            self, t, x,
            t0=None, line_props=dict(),
            **bplot_kws
            ):
        # just do BlitPlot with defaults -- this gives us a figure and axes
        bplot_kws['xlim'] = (t[0], t[-1])
        figure = bplot_kws.pop('figure', None)
        if 'ylim' not in bplot_kws:
            bplot_kws['ylim'] = (x.min(), x.max())
        super(StaticFunctionPlot, self).__init__(figure=figure, **bplot_kws)
        # XXX: why setting traits after construction?
        #self.trait_set(**bplot_kws)
        if t0 is None:
            t0 = t[0]
        ts_line = self.create_fn_image(x, t=t, **line_props)
        self.ax.xaxis.get_major_formatter().set_useOffset(False)
        self.add_static_artist(ts_line)
        self.time_mark = self.ax.axvline(x=t0, color='r', ls='-')
        self.add_dynamic_artist(self.time_mark)

    @on_trait_change('time')
    def move_bar(self, *time):
        if time:
            time = time[0]
            self.trait_setq(time=time)
        else:
            time = self.time
        self.time_mark.set_data(( [time, time], [0, 1] ))
        self.draw_dynamic()

    def connect_live_interaction(self, *extra_connections):
        # connect a sequence of callbacks to
        # click -> enable scrolling
        # drag -> scroll time bar (if scrolling)
        # release click -> disable scrolling
        connections = (
            ('button_press_event', self._scroll_handler),
            ('button_release_event', self._scroll_handler),
            ('motion_notify_event', self._scroll_handler),
            )
        connections = connections + extra_connections
        self._scrolling = False
        super(StaticFunctionPlot, self).connect_live_interaction(*connections)

    def _scroll_handler(self, ev):
        # XXX: debug
        if not ev.inaxes or ev.button != 1:
            return
        if not self._scrolling and ev.name == 'button_press_event':
            self._scrolling = True
            self.time = ev.xdata
        elif ev.name == 'button_release_event':
            self._scrolling = False
        elif self._scrolling and ev.name == 'motion_notify_event':
            self.time = ev.xdata

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
            self.window_length = 1.05*tspan
        else:
            self.window_length = window_length
        super(WindowedFunctionPlot, self).__init__(t, x, **traits)
        self._init_windows(t)

    def _init_windows(self, t):
        # set up the number of windows and the associated limits
        # each window is staggered at (1-overlap) * window_length points
        plen = (1-self.overlap)*self.window_length
        full_window = float( t[-1] - t[0] )
        self._mx_window = max(1, int( full_window / plen ))
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
        stride = (1-self.overlap)*self.window_length
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
        self._window = min(self._mx_window, self._window+1)

    def _prev_window_fired(self):
        self._window = max(0, self._window-1)

    def window_from_time(self, time):
        stride = (1-self.overlap)*self.window_length
        # (n+1)*stride > time > n*stride
        n = int(time/stride)
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

    def connect_live_interaction(self, *extra_connections):
        # connect a right-mouse-button triggered paging
        connections = (
            ('button_press_event', self._window_handler),
            )
        connections = connections + extra_connections
        super(
            WindowedFunctionPlot, self
            ).connect_live_interaction(*connections)

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
            self, t, x, winsize, t0=None, line_props=dict(), **bplot_kws
            ):
        super(ScrollingFunctionPlot, self).__init__(
            t, x, t0=t0, line_props=line_props, **bplot_kws
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
        self.time_mark.set_data(( [time, time], [0, 1] ))
        self.xlim = (time - self.winsize/2, time + self.winsize/2)

    @on_trait_change('winsize')
    def change_window(self):
        time = self.time
        self.xlim = (time - self.winsize/2, time + self.winsize/2)

class PagedFunctionPlot(StaticFunctionPlot):

    page = Int(0)
    page_length = Int
    stack_spacing = Float
    
    def __init__(self, t, x, page_length, stack_traces=True, **traits):
        self.lims = (x.min(), x.max())
        #self._spacing = np.median( np.ptp(x, axis=0) )
        self._zooming = False
        self.page_length = page_length
        self.page = 0
        self.stack_traces = stack_traces
        self.x = x
        self.t = t
        t_init, x_init = self._data_page()
        ## x_init = pt.safe_slice(x, 0, page_length)
        ## t_init = pt.safe_slice(t, 0, page_length)
        super(PagedFunctionPlot, self).__init__(t_init, x_init, **traits)
        self.trait_setq(page_length=page_length)
        # xxx: probably bad form here
        self._traces = self.static_artists[:]
        #self.page_in(0)

    def _data_page(self):
        # page with +/- 1 page length buffer
        start = (self.page-1)*self.page_length
        data_page = pt.safe_slice(self.x, start, 3*self.page_length, fill=0)
        tx_page = pt.safe_slice(self.t, start, 3*self.page_length)
        if self.x.ndim > 1 and self.stack_traces:
            if not self.stack_spacing:
                window = data_page[self.page_length:2*self.page_length]
                spacing = np.median( np.ptp(window, axis=0) )
            else:
                spacing = self.stack_spacing
            data_page = data_page + np.arange(self.x.shape[1]) * spacing
        return tx_page, data_page

    @on_trait_change('page')
    def page_in(self, *page):
        if page:
            self.trait_setq(page=page[0])

        tx, data_page = self._data_page()
        for fn, line_obj in zip(data_page.T, self._traces):
            line_obj.set_data(tx, fn)
        #self.trait_setq(ylim=(data_page.min(), data_page.max()))
        window = data_page[self.page_length:2*self.page_length]
        self.ylim = (np.nanmin(window), np.nanmax(window))
        self.center_page()

    def center_page(self, t_off=0):
        # set axis range to the middle segment of the window
        t = self._traces[0].get_data()[0]
        mn = np.nanmin(t); mx = np.nanmax(t)
        # ??
        twid = self.t[self.page_length] - self.t[0]
        t0 = t[int(1.5*self.page_length)]
        t0 = t0 + t_off
        t_min = max(mn, t0 - twid/2)
        t_max = min(mx, t0 + twid/2)
        ## t_min = max(mn, t[self.page_length])
        ## t_max = min(mx, t[2*self.page_length-1])
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

    ## def connect_live_interaction(self, *extra_connections):
    ##     # connect a right-mouse-button triggered paging
    ##     connections = (
    ##         ('button_press_event', self._zoom_handler),
    ##         ('motion_notify_event', self._zoom_handler),
    ##         ('button_release_event', self._zoom_ender)
    ##         )
    ##     connections = connections + extra_connections
    ##     super(
    ##         PagedFunctionPlot, self
    ##         ).connect_live_interaction(*connections)

    ## def _zoom_handler(self, ev):
    ##     if ev.button != 3 or not ev.inaxes:
    ##         return
    ##     if not self._zooming:
    ##         # start zooming
    ##         self._saved_lims = self.xlim + self.ylim
    ##         self._x0 = (ev.xdata, ev.ydata)
    ##         print 'saving state:', self._saved_lims, self._x0
    ##         self._zooming = True
    ##         return
    ##     x0, y0 = self._x0
    ##     dx = 4*float(ev.xdata - x0)
    ##     dy = 4*float(ev.ydata - y0)
    ##     x_span = self._saved_lims[1] - self._saved_lims[0]
    ##     y_span = self._saved_lims[3] - self._saved_lims[2]
    ##     print dx, dy
    ##     # apply a saturating curve to the mouse motion
    ##     new_y_span = y_span*(np.pi/2 - np.arctan(dy/y_span))
    ##     new_x_span = x_span*(np.pi/2 - np.arctan(dx/x_span))
    ##     print (0.5 - np.arctan(dy/y_span)/np.pi),
    ##     print (0.5 - np.arctan(dx/x_span)/np.pi)
    ##     ## print (y0 - new_y_span/2, y0 + new_y_span/2), 
    ##     ## print (x0 - new_x_span/2, x0 + new_x_span/2)
    ##     self.ylim = (y0 - new_y_span/2, y0 + new_y_span/2)
    ##     #self.trait_setq(ylim = (y0 - new_y_span/2, y0 + new_y_span/2))
    ##     #self.trait_setq(xlim = (x0 - new_x_span/2, x0 + new_x_span/2))

    ## def _zoom_ender(self, ev):
    ##     if ev.button != 3:
    ##         return
    ##     print 'ending zoom state'
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
        raise NotImpelentedError('Abstract class: does not plot')

class StandardPlot(ProtoPlot):
    """
    A mixin type that plots a simple times series line.
    """

    # Caution: mixin-only. Supposes the attribute
    # * ax, an MPL Axes

    # this signature should be pretty generic
    def create_fn_image(self, x, t=None, **line_props):
        print 'CREATING STANDARD PLOT'
        # in this case, just a plot
        if t is None:
            t = np.arange(x.shape[0])
        lines = self.ax.plot(t, x, **line_props)
        return lines

class ColorCodedPlot(ProtoPlot):
    """
    A mixin-type whose image is a color-coded time series lines
    """

    # Caution: mixin-only. Supposes the attributes
    # * ax, an MPL Axes
    # * cx, a color-coding function
    # * cx_limits, the (possibly clipped) dynamic range of "cx"

    def create_fn_image(self, x, t=None, **line_props):
        print 'CREATING COLOR CODED PLOT'
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
        cmap = line_props.pop('cmap', cm.jet)
        #colors = cmap( norm(cx) )
        line_props['edgecolors'] = 'none'
        pc = self.ax.scatter(
            t, x, 14.0, c=cx, norm=norm, cmap=cmap, **line_props
            )
        return pc

# XXX: would be nice to have an option to write class labels,
# possible with "annotate" fn.
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

    def create_fn_image(self, x, t=None, labels=None, **line_props):
        # Caution! Returns a sequence of separate lines for each data
        # segment
        #cmap = line_props.pop('cmap', cm.jet)
        cmap = line_props.pop('cmap', cm.hsv)
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
        label_to_idx = dict( zip( unique_labels, range(n_labels) ) )
        if 0 in unique_labels:
            colors = cmap(np.linspace(0, 1, n_labels-1))
            colors = np.row_stack( ([0, 0, 0, 0.15], colors) )
        else:
            colors = cmap(np.linspace(0, 1, n_labels))
        for seg in unique_labels:
            # -1 codes for out of bounds
            if seg < 0:
                continue
            #for seg, color in zip(xrange(mx_label+1), colors):
            seg_line.fill(np.nan)
            idx = np.where(labels==seg)[0]
            np.put(seg_line, idx, np.take(x, idx))
            cidx = label_to_idx[seg]
            color = colors[cidx]
            ## if seg==0:
            ##     color[-1] = 0.15
            line = self.ax.plot(t, seg_line, c=color, **line_props)
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
            line_props=dict(), **bplot_kws
            ):
        self.cx = cx
        if not cx_limits:
            #eps = stochastic_limits(cx, conf=95.)
            mx = cx.max()
            mn = cx.min()
            cx_limits = (mn, mx)
        self.cx_limits = cx_limits
        super(StaticColorCodedPlot, self).__init__(
            t, x, t0=t0, line_props=line_props, **bplot_kws
            )

class StaticSegmentedPlot(StaticFunctionPlot, ClassSegmentedPlot):
    """
    A static plot, but the line plot is colored by k different classes.
    This is distinct from a ColorCodedPlot in that there are only
    a low number of discrete classes.
    """

    def __init__(
            self, t, x, labels, t0=None, line_props=dict(), **bplot_kws
            ):
        self.labels = labels
        unique_labels = np.unique(labels)
        self.n_classes = len( unique_labels >= 0 )
        super(StaticSegmentedPlot, self).__init__(
            t, x, t0=t0, line_props=line_props, **bplot_kws
            )

class WindowedTimeSeriesPlot(WindowedFunctionPlot, StandardPlot):
    """
    A static plot that is flipped between windows.
    """
    # defaults for both classes
    pass

class PagedTimeSeriesPlot(PagedFunctionPlot, StandardPlot):
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
            line_props=dict(), **bplot_kws
            ):
        self.cx = cx
        if not cx_limits:
            #eps = stochastic_limits(cx, conf=95.)
            mx = cx.max()
            mn = cx.min()
            cx_limits = (mn, mx)
        self.cx_limits = cx_limits
        super(WindowedColorCodedPlot, self).__init__(
            t, x, t0=t0, line_props=line_props, **bplot_kws
            )

class WindowedClassSegmentedPlot(WindowedFunctionPlot, ClassSegmentedPlot):
    """
    A static class-colored plot that is flipped between intervals
    """
    def __init__(
            self, t, x, labels, t0=None, line_props=dict(), **bplot_kws
            ):
        self.labels = labels
        unique_labels = np.unique(labels)
        self.n_classes = len( unique_labels >= 0 )
        super(WindowedClassSegmentedPlot, self).__init__(
            t, x, t0=t0, line_props=line_props, **bplot_kws
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
            line_props=dict(), **bplot_kws
            ):
        # make sure to set the color code first
        self.cx = cx
        if not cx_limits:
            mx = cx.max()
            mn = cx.min()
            cx_limits = (mn, mx)
        self.cx_limits = cx_limits
        super(ScrollingColorCodedPlot, self).__init__(
            t, x, winsize, t0=t0, line_props=line_props, **bplot_kws
            )

class ScrollingClassSegmentedPlot(ScrollingFunctionPlot, ClassSegmentedPlot):
    """
    A scrolling plot, but points are color-coded by class.
    """

    def __init__(
            self, t, x, winsize, labels, 
            line_props=dict(), **bplot_kws
            ):
        # make sure to set the class code first
        self.labels = labels
        unique_labels = np.unique(labels)
        self.n_classes = len( unique_labels >= 0 )
        # hold onto these
        self._lprops = line_props
        super(ScrollingClassSegmentedPlot, self).__init__(
            t, x, winsize, line_props=line_props, **bplot_kws
            )
