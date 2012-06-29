## from pyface.qt import QtGui, QtCore
import numpy as np
import matplotlib
from threading import Thread
from time import sleep, time
import random

# We want matplotlib to use a QT backend
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.mlab import prctile
from matplotlib.colors import Normalize
from matplotlib import cm

from traitsui.qt4.editor import Editor
#from traitsui.qt4.basic_editor_factory import BasicEditorFactory
from traitsui.basic_editor_factory import BasicEditorFactory

from traits.api \
    import HasTraits, HasPrivateTraits, Instance, Enum, Dict, Constant, Str, \
    List, on_trait_change, Float, File, Array, Button, Range, Property, \
    cached_property, Event, Bool, Color, Int, String, Any, Tuple
    
from traitsui.api \
  import Item, Group, View, VGroup, HGroup, HSplit, \
  EnumEditor, CheckListEditor, ListEditor, message, ButtonEditor, RangeEditor

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk
from mayavi.sources.api import ArraySource


#### Utility for quick range finding XXX: way too liberal of a bound!
def stochastic_limits(x, n_samps=100, conf=98.0):
    """
    Use Markov's inequality to estimate a bound on the
    absolute values in the array x.
    """
    n = len(x)
    r_pts = random.sample(xrange(n), n_samps)
    r_samps = np.take(x, r_pts)
    # unbiased estimator??
    e_abs = np.abs(r_samps).mean()
    # Pr{ |X| > t } <= E{|X|}/t = 1 - conf/100
    # so find the threshold for which there is only a 
    # (100-conf)% chance that |X| is greater
    return e_abs/(1.0-conf/100.0)

#### Utility for safe sub-slicing
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
            sx.fill(np.nan)
            # range is entirely outside
            return sx
        if start < 0:
            # fill beginning ( where i < 0 ) with NaN
            sx[:-start, ...] = fill
            # fill the rest with x
            sx[-start:, ...] = x[:(num + start), ...]
        else:
            sx[:(num-start), ...] = x[start:, ...]
            sx[(num-start):, ...] = fill
    else:
        sx = x[start:start+num, ...]
    return sx

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
    
#### Data Scrolling App ####
#### (out of order) ####
## class ClockRunner(Thread):

##     def __init__(self, set_time, rate, incr, **thread_kws):
##         # pretty hacky.. but since the "time" attribute of the
##         # DataScroller looks like just a float in this context,
##         # we'll use a method that gets and sets the time
##         # -- set_time() returns the current time
##         # -- set_time(t) sets the current time
##         self.set_time = set_time
##         self.time = set_time()
##         self.stop_time = self.time + 10*incr ### throw away later
##         self.rate = rate # fps -- so sleep 1/rate between each update
##         self.incr = incr
##         self.abort = False
##         Thread.__init__(self, **thread_kws)
    
##     def run(self):
##         while not self.abort:
##             self.time += self.incr
##             self.set_time(self.time)
##             print 'would set to:', self.time
##             sleep(1.0/self.rate)
##             if self.time >= self.stop_time:
##                 abort = True
##         return

class DataScroller(HasTraits):

    ## these may need to be more specialized for handling 1D/2D timeseries
    zoom_plot = Instance(ScrollingTimeSeriesPlot)
    ts_plot = Instance(StaticTimeSeriesPlot)

    ## array scene, image, and data (Mayavi components)
    array_scene = Instance(MlabSceneModel, ())

    #arr_img_data = Array()
    arr_img_dsource = Instance(ArraySource, (), transpose_input_array=False)

    array_ipw = Instance(PipelineBase)
    
    ## view controls

    # interval for zoom plot (units sec)
    tau = Range(low=1.0, high=50.0, value=1.0)

    # limits for abs-amplitude (auto tuned)

    # going to map this to max_amp*(sin(pi*(eps-1/2)) + 1)/2 to
    # prevent blowing up the ylim too easily with the range slider
    eps = Range(
        low=0.0, high=1.0, value=0.5,
        editor=RangeEditor(
            format='%1.1f', low_label='tight', high_label='wide'
            )
        )
    
    ## scroller (auto tuned)
    _t0 = Float(0.0)
    _tf = Float
    time = Range(
        low='_t0', high='_tf',
        editor=RangeEditor(low_name='_t0', high_name='_tf',
                           format='%1.2f', mode='slider')
        )

    ## Animation control
    fps = Float(20.0)
    count = Button()
    counter = Instance(Thread)

    def __init__(self, d_array, ts_array, nrow, ncol, Fs, **traits):
        npts = d_array.shape[1]
        self._tf = float(npts-1) / Fs
        self.Fs = Fs
        # XXX: should set max_amp -- could stochastically sample to
        # estimate mean and variance
        #self.max_amp = stochastic_limits(ts_array)
        self.max_amp = ts_array.max()
        
        self.ts_arr = ts_array

        # Reshape the data as (ncol, nrow, ntime) to keep it contiguous...
        # this will also correspond to (k,j,i) indexing. Then flatten
        # and reshape the array to (ntime, nrow, ncol) (z,y,x) to
        # satisfy the VTKImageData column-major format
        new_shape = (npts, nrow, ncol)
        vtk_arr = np.reshape(d_array.transpose(), new_shape, order='F')
        # make the time dimension unit length, and put the origin at -1/2
        self.arr_img_dsource.spacing = 1./npts, 1., 1.
        self.arr_img_dsource.origin = (-0.5, 0.0, 0.0)
        self.arr_img_dsource.scalar_data = vtk_arr

        # pop out some traits that should be set after initialization
        time = traits.pop('time', 0)
        tau = traits.pop('tau', 1.0)
        i_eps = traits.pop('eps', 0.5)
        HasTraits.__init__(self, **traits)
        self._scrolling = False

        # configure the ts_plot
        n = self.ts_arr.shape[-1]
        t = np.linspace(self._t0, self._tf, n)
        eps = self.__map_eps(i_eps)
        figsize=(6,.25)
        self.ts_plot = self.construct_ts_plot(
            t, figsize, eps, time, linewidth=1
            )
        self.sync_trait('time', self.ts_plot, mutual=False)

        # configure the zoomed plot
        figsize=(4,2)
        self.zoom_plot = self.construct_zoom_plot(figsize, eps)

        self.trait_setq(tau=tau)
        self.trait_setq(time=time)
        self.trait_setq(eps=i_eps)

    def construct_ts_plot(self, t, figsize, eps, t0, **lprops):
        return StaticTimeSeriesPlot(
            t, self.ts_arr, figsize=figsize, ylim=(-eps, eps), t0=t0,
            line_props=lprops
            )

    def construct_zoom_plot(self, figsize, eps, **lprops):
        x = self.zoom_data()
        return ScrollingTimeSeriesPlot(
            x, figsize=figsize, ylim=(-eps, eps)
            )

    def configure_traits(self, *args, **kwargs):
        super(DataScroller, self).configure_traits(*args, **kwargs)
        self._post_canvas_hook()

    def edit_traits(self, *args, **kwargs):
        super(DataScroller, self).edit_traits(*args, **kwargs)
        self._post_canvas_hook()

    def _post_canvas_hook(self):
        self._connect_mpl_events()
        self.ts_plot.fig.tight_layout()
        self.ts_plot.draw()
        self.zoom_plot.fig.tight_layout()
        self.zoom_plot.draw()

    def zoom_data(self):
        d = self.ts_arr
        n_pts = d.shape[-1]
        n_zoom_pts = int(np.round(self.tau*self.Fs))
        zoom_start = int(np.round(self.Fs*(self.time - self.tau/2)))
        return safe_slice(d, zoom_start, n_zoom_pts)

    def _connect_mpl_events(self):
        # connect a sequence of callbacks to
        # click -> enable scrolling
        # drag -> scroll time bar (if scrolling)
        # unclick -> disable scrolling
        self.ts_plot.fig.canvas.mpl_connect(
            'button_press_event', self._scroll_handler
            )
        self.ts_plot.fig.canvas.mpl_connect(
            'button_release_event', self._scroll_handler
            )
        self.ts_plot.fig.canvas.mpl_connect(
            'motion_notify_event', self._scroll_handler
            )

    def _scroll_handler(self, ev):
        if not ev.inaxes:
            return
        if ev.name == 'button_press_event':
            self._scrolling = True
        elif ev.name == 'button_release_event':
            self._scrolling = False
        elif self._scrolling and ev.name == 'motion_notify_event':
            self.time = ev.xdata
        
    @on_trait_change('time')
    def _update_time(self):
        # 1) long ts_plot should be automatically linked to 'time'
        # 2) zoomed ts_plot needs new x data
        x = self.zoom_data()
        # manipulate x to accomodate overloaded input/output
        if type(x) != tuple:
            x = (x,)
        self.zoom_plot.set_window(*x) # works with overloaded zoom_data()
        # 3) VTK ImagePlaneWidget needs to be resliced
        if self.array_ipw:
            self.array_ipw.ipw.slice_index = int( np.round(self.time*self.Fs) )
        
    def __map_eps(self, eps):
        return self.max_amp*((np.sin(np.pi*(self.eps-1/2.0))+1)/2.0)**2
        
    @on_trait_change('eps')
    def _update_eps(self):
        max_amp = 1.0
        true_eps = self.__map_eps(self.eps)
        lim = (-true_eps, true_eps)
        self.ts_plot.ylim = lim
        self.zoom_plot.ylim = lim
        if self.array_ipw:
            self.array_ipw.module_manager.scalar_lut_manager.data_range = lim

    @on_trait_change('tau')
    def _plot_new_zoom(self):
        x = self.zoom_data()
        # manipulate x to accomodate overloaded input/output
        if type(x) != tuple:
            x = (x,)
        self.zoom_plot.set_window(*x)
    
    ## animation
    def __set_time(self, *args):
        if not args:
            return self.time
        t = args[0]
        self.time = t
    
    ## def _count_fired(self):
    ##     if self.counter and self.counter.isAlive():
    ##         self.counter.abort = True
    ##     else:
    ##         self.counter = ClockRunner(self.__set_time, self.fps, 1.0/self.Fs)
    ##         self.counter.start()

    def _count_fired(self):
        if self.counter and self.counter.isAlive():
            self._quit_counting = True
        else:
            self._quit_counting = False
            self.counter = Thread(target=self._count)
            self.counter.start()

    def _count(self):
        while not self._quit_counting:
            t = self.time + 1.0/self.Fs
            self.trait_setq(time=t)
            self.ts_plot.move_bar(t)
            t1 = time()
            self._update_time()
            t2 = time()
            s_time = max(0., 1/self.fps - t2 + t1)
            print 'time to draw:', (t2-t1), 'sleep time:', s_time
            sleep(s_time)
            
    ## mayavi components
    
    @on_trait_change('array_scene.activated')
    def _display_image(self):
        scene = self.array_scene
        eps = self.__map_eps(self.eps)
        ipw = mlab.pipeline.image_plane_widget(
            self.arr_img_dsource,
            plane_orientation='x_axes',
            figure=scene.mayavi_scene,
            vmin=-eps, vmax=eps
            )

        ipw.ipw.slice_position = np.round(self.time * self.Fs)
        ipw.ipw.interaction = 0

        scene.mlab.view(azimuth=0, elevation=90, distance=50)
        scene.scene.interactor.interactor_style = \
          tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)

        self.array_ipw = ipw
        
                
    view = View(
        VGroup(
            HGroup(
                Item(
                    'array_scene', editor=SceneEditor(scene_class=Scene),
                    height=200, width=200, show_label=False
                    ),
                Item(
                    'zoom_plot', editor=MPLFigureEditor(),
                    show_label=False, width=500, height=200, resizable=True
                    )
                ),
            HGroup(
                Item(
                    'ts_plot', editor=MPLFigureEditor(),
                    show_label=False, width=700, height=100, resizable=True
                    ),
                Item('fps', label='FPS'),
                Item('count', label='Run Clock')
                ),
            HGroup(
                VGroup(
                    Item('tau', label='Zoom Interval'),
                    Item('eps', label='Amplitude Interval')
                    ),
                Item('time', label='Time Slice', style='custom')
                )
            ),
        resizable=True,
        title='Data Scroller'
    )

class ColorCodedDataScroller(DataScroller):

    zoom_plot = Instance(ScrollingColorCodedPlot)
    ts_plot = Instance(StaticColorCodedPlot)

    def __init__(self, d_array, ts_array, cx_array, nrow, ncol, Fs, **traits):
        self.cx_arr = cx_array
        DataScroller.__init__(
            self, d_array, ts_array, nrow, ncol, Fs, **traits
            )

    def construct_ts_plot(self, t, figsize, eps, t0, **lprops):
        dfac = 1
        t = t[::dfac]
        ts_arr = self.ts_arr[::dfac]
        cx_arr = self.cx_arr[::dfac]
        return StaticColorCodedPlot(
            t, ts_arr, cx_arr,
            figsize=figsize, ylim=(-eps, eps), t0=t0,
            line_props=lprops
            )
    
    def construct_zoom_plot(self, figsize, eps, **lprops):
        x, cx = self.zoom_data()
        cx_lim = self.ts_plot.cx_limits # ooh! hacky
        return ScrollingColorCodedPlot(
            x, cx, cx_lim, 
            figsize=figsize, ylim=(-eps, eps), line_props=lprops
            )

    def zoom_data(self):
        d = self.ts_arr
        n_pts = d.shape[-1]
        n_zoom_pts = int(np.round(self.tau*self.Fs))
        zoom_start = int(np.round(self.Fs*(self.time - self.tau/2)))
        x = safe_slice(d, zoom_start, n_zoom_pts)
        cx = safe_slice(self.cx_arr, zoom_start, n_zoom_pts, fill=0)
        return x, cx

    
if __name__ == "__main__":
    pass
   
