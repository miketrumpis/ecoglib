import numpy as np

# ETS
from traits.api import \
     HasTraits, on_trait_change, Float, Int, Tuple, Range, Button
from traitsui.qt4.editor import Editor
from traitsui.basic_editor_factory import BasicEditorFactory

# Matplotlib
import matplotlib
# We want matplotlib to use a QT backend
matplotlib.use('QtAgg')
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
        traits = dict(xlim=xlim, ylim=ylim)
        super(BlitPlot, self).__init__(**traits)

    def add_static_artist(self, a):
        self.static_artists.append(a)

    def add_dynamic_artist(self, a):
        self.dynamic_artists.append(a)

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
        self._bkgrnd = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        for artist in self.dynamic_artists:
            artist.set_visible(True)
        self.fig.canvas.draw()

    # this method only pushes out the old background, and renders the
    # dynamic artists (which have presumably changed)
    def draw_dynamic(self):
        """
        Draw only dynamic actors, then restore the background.
        """
        if self._bkgrnd is not None:
            self.fig.canvas.restore_region(self._bkgrnd)
        for artist in self.dynamic_artists:
            self.ax.draw_artist(artist)
        self.fig.canvas.blit(self.ax.bbox)

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
        super(StaticFunctionPlot, self).__init__()
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

    def set_window(self, x, tc=None):
        """
        Update the interval of the function (x is restriction to 
        the new interval). If given, "tc" is the current time defining
        the interval.
        """
        winsize = len(x)
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
            t = np.arange(len(x))
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
            eps = np.abs(cx).max()
            limits = (-eps, eps)
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
            eps = np.abs(cx).max()
            cx_limits = (-eps, eps)
        self.cx_limits = cx_limits
        super(StaticColorCodedPlot, self).__init__(
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
            eps = np.abs(cx).max()
            cx_limits = (-eps, eps)
        self.cx_limits = cx_limits
        super(PagedColorCodedPlot, self).__init__(
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