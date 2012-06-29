import numpy as np

# ETS
from traits.api import HasTraits, on_trait_change, Float, Int, Tuple
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


#### Matplotlib to Traits Panel Integration ####
  
class _MPLFigureEditor(Editor):

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

#### Simple Figure Wrapper ####

class BlitPlot(HasTraits):

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
        if self._bkgrnd is not None:
            self.fig.canvas.restore_region(self._bkgrnd)
        for artist in self.dynamic_artists:
            self.ax.draw_artist(artist)
        self.fig.canvas.blit(self.ax.bbox)

class LongNarrowPlot(BlitPlot):
    n_yticks = Int(2)
    @on_trait_change('ylim')
    def set_ylim(self, *ylim):
        if not ylim:
            ylim = self.ylim
        else:
            self.trait_setq(ylim=ylim)
        y_ticks = np.linspace(ylim[0], ylim[1], self.n_yticks)
        super(LongNarrowPlot, self).set_ylim(*ylim)
    
class StandardPlot(object):
    """A long-and-narrow BlitPlot that plots a simple times series line
    Prototype only! 
    """

    # this signature should be pretty generic
    def create_fn_image(self, x, t=None, **line_props):
        print 'CREATING STANDARD PLOT'
        # in this case, just a plot
        if t is None:
            t = np.arange(len(x))
        line = self.ax.plot(t, x, **line_props)[0]
        return line

class ColorCodedPlot(object):
    """A long-and-narrow BlitPlot whose image is a 
    color-coded time series lines --
    Prototype only! 
    """

    # expects two attributes to have been set: 'cx' and 'cx_limits'
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
        pc = self.ax.scatter(
            t, x, 14.0, c=cx, norm=norm, cmap=cmap, **line_props
            )
        return pc
        
class StaticFunctionPlot(LongNarrowPlot):
    """
    Plain vanilla x(t) plot, with a marker for the current time.
    """
    time = Float
    
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

class ScrollingFunctionPlot(LongNarrowPlot):
    """
    Plain vanilla x(t) plot, but only over a given interval
    """

    def __init__(self, x, line_props=dict(), **bplot_kws):
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

    def set_window(self, x):
        winsize = len(x)
        if winsize == self.winsize:
            t, old_x = self.zoom_element.get_data()
            self.zoom_element.set_data(t, x)
            self.draw_dynamic()
            return
        t = np.arange(winsize)
        self.zoom_element.set_data(t, x)
        t0 = np.round(winsize/2.)
        self.time_mark.set_data([t0, t0], [0, 1])
        self.winsize = winsize
        self.xlim = (-1, self.winsize)

class StaticTimeSeriesPlot(StaticFunctionPlot, StandardPlot):
    # do defaults for both classes
    pass
    
class StaticColorCodedPlot(StaticFunctionPlot, ColorCodedPlot):
    """
    x(t) plot, but points are color-coded by a co-function c(t)
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

class ScrollingTimeSeriesPlot(ScrollingFunctionPlot, StandardPlot):
    # do defaults for both classes
    pass
        
class ScrollingColorCodedPlot(ScrollingFunctionPlot, ColorCodedPlot):

    def __init__(self, x, cx, cx_limits, line_props=dict(), **bplot_kws):
        # make sure to set the color code first
        self.cx = cx
        # cx_limits *must* be provided, since it is unreliable to estimate
        # them from the short window of cx provided here
        self.cx_limits = cx_limits
        super(ScrollingColorCodedPlot, self).__init__(
            x, line_props=line_props, **bplot_kws
            )

    # the signature of set_window changes now
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
