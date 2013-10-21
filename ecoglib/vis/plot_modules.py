import numpy as np

# ETS
from traits.api import \
     HasTraits, on_trait_change, Float, Int, Tuple, Range, Button
from traitsui.qt4.editor import Editor
from traitsui.basic_editor_factory import BasicEditorFactory

# Matplotlib
import matplotlib
# We want matplotlib to use a QT backend
try:
    matplotlib.use('QtAgg')
except ValueError:
    matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import \
     FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib import cm

# XXX: an incomplete decorator for the listen/set pattern of traits callbacks
## def set_or_listen(attr):

##     @on_trait_change(attr)
##     def

##############################################################################
########## Matplotlib to Traits Panel Integration ############################

class _MPLFigureEditor(Editor):
    """
    This class provides a QT canvas to all MPL figures when drawn
    under the TraitsUI framework.
    """

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # matplotlib commands to create a canvas
        mpl_canvas = FigureCanvas(self.value.fig)
        return mpl_canvas

class MPLFigureEditor(BasicEditorFactory):

   klass = _MPLFigureEditor

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
        traits = dict(xlim=xlim, ylim=ylim)
        super(BlitPlot, self).__init__(**traits)

    def add_static_artist(self, a):
        if not np.iterable(a):
            a = (a,)
        self.static_artists.extend(a)

    def add_dynamic_artist(self, a):
        if not np.iterable(a):
            a = (a,)
        self.dynamic_artists.extend(a)

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
        print ev.name

class LongNarrowPlot(BlitPlot):
    """
    A BlitPlot with less tick clutter on the y-axis.
    """
    n_yticks = Int(2)
    @on_trait_change('ylim')
    def set_ylim(self, *ylim):
        if not ylim:
            ylim = self.ylim
        else:
            self.trait_setq(ylim=ylim)
        y_ticks = np.linspace(ylim[0], ylim[1], self.n_yticks)
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
        super(StaticFunctionPlot, self).__init__(figure=figure)
        # XXX: why setting traits after construction?
        self.trait_set(**bplot_kws)
        if t0 is None:
            t0 = t[0]
        ts_line = self.create_fn_image(x, t=t, **line_props)
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

class PagedFunctionPlot(StaticFunctionPlot):
    """
    Same behavior as a static plot, but the plot window flips between
    "pages" of set duration.
    """

    #page_length = Int(10000) # default 10000 point window
    page_length = Float(100) # units are arbitrary, as long as they match
                             # the time axis

    _mx_page = Int
    _page = Range(low=0, high='_mx_page', value=0)
    _overlap = Float(0.15)

    next_page = Button()
    prev_page = Button()

    def __init__(self, t, x, **traits):
        super(PagedFunctionPlot, self).__init__(t, x, **traits)
        self._init_pages(t)

    def _init_pages(self, t):
        # set up the number of pages and the associated limits
        # each page is staggered at (1-overlap) * page_length points
        plen = (1-self._overlap)*self.page_length
        full_window = float( t[-1] - t[0] )
        self._mx_page = int( full_window / plen )
        self._time_insensitive = False
        self.change_page(self._page)

    @on_trait_change('_page')
    def change_page(self, *page):
        if page:
            page = page[0]
            self.trait_setq(_page=page)
        else:
            page = self._page

        # the page length is set, but the page stride is
        # (1-overlap) * page_length
        stride = (1-self._overlap)*self.page_length
        x_start = page * stride
        x_stop = x_start + self.page_length
        # this will triger a redraw
        self.xlim = (x_start, x_stop)
        # if a page was switched while the current time is somewhere
        # outside the current interval, then enter a mode where subsequent
        # time updates won't change the page back. This will last until
        # the current time comes back into the current page
        if self.time < x_start or self.time > x_stop:
            self._time_insensitive = True

    def _next_page_fired(self):
        self._page = min(self._mx_page, self._page+1)

    def _prev_page_fired(self):
        self._page = max(0, self._page-1)

    def page_from_time(self, time):
        stride = (1-self._overlap)*self.page_length
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
            self._page = self.page_from_time(time)
        else:
            # if time is within the interval then we can always
            # be sensitive to it
            self._time_insensitive = False
        super(PagedFunctionPlot, self).move_bar(time)

    def connect_live_interaction(self, *extra_connections):
        # connect a right-mouse-button triggered paging
        connections = (
            ('button_press_event', self._page_handler),
            )
        connections = connections + extra_connections
        super(PagedFunctionPlot, self).connect_live_interaction(*connections)


    def _page_handler(self, ev):
        if ev.button != 3 or not ev.inaxes:
            return
        x = ev.xdata
        x_lo, x_hi = self.xlim
        if np.abs(x - x_lo) < np.abs(x - x_hi):
            self.prev_page = True
        else:
            self.next_page = True


class ScrollingFunctionPlot(LongNarrowPlot):
    """
    Plain vanilla x(t) plot, but only over a given interval. A time
    marker is placed at the current time, which by default is the
    center of the window.
    """

    # Caution: mixin-only. Supposes the method
    # * create_fn_image() which produces a particular plot
    #
    # This method is provided by the ProtoPlot types (e.g. StandardPlot)

    def __init__(self, x, line_props=dict(), **bplot_kws):
        """
        Parameters
        ----------

        x: ndarray
          x is a restriction of some function over an initial interval

        """
        self.winsize = len(x)
        bplot_kws['xlim'] = (-1, self.winsize)
        super(ScrollingFunctionPlot, self).__init__()
        self.trait_set(**bplot_kws)
        self.ax.xaxis.set_visible(False)
        self.zoom_element = self.create_fn_image(x, **line_props)
        self.add_dynamic_artist(self.zoom_element)
        t0 = np.floor(self.winsize/2.)
        self.time_mark = self.ax.axvline(x=t0, color='r', ls=':')
        self.add_static_artist(self.time_mark)

    # XXX: should reconsider resetting the data in this plot at every
    # step -- would be more self-contained to just move the x-limits,
    # or somehow handle the transition within this object
    def set_window(self, x, tc=None):
        """
        Update the interval of the function (x is restriction to
        the new interval). If given, "tc" is the current time defining
        the interval.
        """
        winsize = x.shape[0]
        if winsize == self.winsize:
            t, old_x = self.zoom_element.get_data()
            self.zoom_element.set_data(t, x)
            self.draw_dynamic()
            return
        t = np.arange(winsize)
        self.zoom_element.set_data(t, x)
        if tc is None:
            tc = np.round(winsize/2.)
        self.time_mark.set_data([tc, tc], [0, 1])
        self.winsize = winsize
        # setting xlim triggers draw
        self.xlim = (-1, self.winsize)

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
        line = self.ax.plot(t, x, **line_props)[0]
        return line

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
        cmap = line_props.pop('cmap', cm.jet)
        if t is None:
            t = np.arange(len(x))
        if not hasattr(self, 'labels'):
            raise RuntimeError(
                'Object should have been instantiated with color code'
                )
        if labels is None:
            labels = self.labels
        colors = cmap(np.linspace(0, 1, self.n_classes))
        # for each class sequentially fill out-of-class pts with nan,
        # and line plot in-class points with the appropriate color
        seg_line = np.empty_like(x)
        seg_lines = []
        unique_labels = np.unique(labels)
        for seg in unique_labels:
            # -1 codes for out of bounds
            if seg < 0:
                continue
            #for seg, color in zip(xrange(mx_label+1), colors):
            seg_line.fill(np.nan)
            idx = np.where(labels==seg)[0]
            np.put(seg_line, idx, np.take(x, idx))
            color = colors[seg]
            if seg==0:
                color[-1] = 0.15
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

class PagedTimeSeriesPlot(PagedFunctionPlot, StandardPlot):
    """
    A static plot that is flipped between pages.
    """
    # defaults for both classes
    pass

class PagedColorCodedPlot(PagedFunctionPlot, ColorCodedPlot):
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
        super(PagedColorCodedPlot, self).__init__(
            t, x, t0=t0, line_props=line_props, **bplot_kws
            )

class PagedClassSegmentedPlot(PagedFunctionPlot, ClassSegmentedPlot):
    """
    A static class-colored plot that is flipped between intervals
    """
    def __init__(
            self, t, x, labels, t0=None, line_props=dict(), **bplot_kws
            ):
        self.labels = labels
        unique_labels = np.unique(labels)
        self.n_classes = len( unique_labels >= 0 )
        super(PagedClassSegmentedPlot, self).__init__(
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
    def __init__(self, x, cx, cx_limits, line_props=dict(), **bplot_kws):
        # make sure to set the color code first
        self.cx = cx
        # cx_limits *must* be provided, since it is unreliable to estimate
        # them from the short window of cx provided here
        self.cx_limits = cx_limits
        super(ScrollingColorCodedPlot, self).__init__(
            x, line_props=line_props, **bplot_kws
            )

    # the signature of set_window changes now (XXX: dangerous?)
    def set_window(self, x, cx):
        winsize = len(x)
        norm = self.zoom_element.norm
        cmap = self.zoom_element.cmap
        if winsize == self.winsize:
            old_offset = self.zoom_element.get_offsets()
            # possibility that the previous set of points included
            # NaNs, in which case the offset data itself was truncated --
            # thus make a new pseudo-time axis if necessary
            if old_offset.shape[0] != winsize:
                t = np.arange(winsize)
            else:
                t = old_offset[:,0]
            self.zoom_element.set_offsets( np.c_[t, x] )
            self.zoom_element.set_color( cmap( norm(cx) ) )
            self.draw_dynamic()
            return
        t = np.arange(winsize)
        self.zoom_element.set_offsets( np.c_[t, x] )
        self.zoom_element.set_color( cmap( norm(cx) ) )
        t0 = np.round(winsize/2.)
        self.time_mark.set_data([t0, t0], [0, 1])
        self.winsize = winsize
        self.xlim = (-1, self.winsize)

class ScrollingClassSegmentedPlot(ScrollingFunctionPlot, ClassSegmentedPlot):
    """
    A scrolling plot, but points are color-coded by class.
    """

    def __init__(self, x, labels, n_classes, line_props=dict(), **bplot_kws):
        # make sure to set the class code first
        self.labels = labels
        self.n_classes = n_classes
        # hold onto these
        self._lprops = line_props
        super(ScrollingClassSegmentedPlot, self).__init__(
            x, line_props=line_props, **bplot_kws
            )

    def set_window(self, x, labels):
        winsize = len(x)
        # want to flush previous lines and re-generate class coded lines
        while self.dynamic_artists:
            line = self.dynamic_artists.pop()
            line.remove()
        self.zoom_element = self.create_fn_image(
            x, labels=labels, **self._lprops
            )
        self.add_dynamic_artist(self.zoom_element)
        t0 = np.round(winsize/2.)
        self.time_mark.set_data([t0, t0], [0, 1])
        self.winsize = winsize
        self.xlim = (-1, self.winsize)
        self.draw_dynamic()

