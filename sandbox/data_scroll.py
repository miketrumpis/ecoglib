## from pyface.qt import QtGui, QtCore
import numpy as np
import matplotlib
from threading import Thread
from time import sleep, time

# We want matplotlib to use a QT backend
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.mlab import prctile

from traitsui.qt4.editor import Editor
#from traitsui.qt4.basic_editor_factory import BasicEditorFactory
from traitsui.basic_editor_factory import BasicEditorFactory

from traits.api \
    import HasTraits, HasPrivateTraits, Instance, Enum, Dict, Constant, Str, \
    List, on_trait_change, Float, File, Array, Button, Range, Property, \
    cached_property, Event, Bool, Color, Int, String, Any
    
from traitsui.api \
  import Item, Group, View, VGroup, HGroup, HSplit, \
  EnumEditor, CheckListEditor, ListEditor, message, ButtonEditor, RangeEditor

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk
from mayavi.sources.api import ArraySource

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
       mpl_canvas = FigureCanvas(self.value)
       return mpl_canvas

class MPLFigureEditor(BasicEditorFactory):

   klass = _MPLFigureEditor

#### Simple Figure Wrapper ####

# ???

#### Data Scrolling App ####

class ClockRunner(Thread):

    def __init__(self, set_time, rate, incr, **thread_kws):
        # pretty hacky.. but since the "time" attribute of the
        # DataScroller looks like just a float in this context,
        # we'll use a method that gets and sets the time
        # -- set_time() returns the current time
        # -- set_time(t) sets the current time
        self.set_time = set_time
        self.time = set_time()
        self.stop_time = self.time + 10*incr ### throw away later
        self.rate = rate # fps -- so sleep 1/rate between each update
        self.incr = incr
        self.abort = False
        Thread.__init__(self, **thread_kws)
    
    def run(self):
        while not self.abort:
            self.time += self.incr
            self.set_time(self.time)
            print 'would set to:', self.time
            sleep(1.0/self.rate)
            if self.time >= self.stop_time:
                abort = True
        return

class DataScroller(HasTraits):

    ## these may need to be more specialized for handling 1D/2D timeseries
    zoom_plot = Instance(Figure, (), figsize=(6,2))
    ts_plot = Instance(Figure, (), figsize=(7,0.5))

    ## array scene, image, and data (Mayavi components)
    array_scene = Instance(MlabSceneModel, ())

    arr_img_data = Array()
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

    def __init__(self, d_array, nrow, ncol, Fs, ts_xform=None, **traits):
        npts = d_array.shape[1]
        self._tf = float(npts-1) / Fs
        self.Fs = Fs
        # XXX: should set max_amp 
        
        # ts_xform maps the (n_chan, n_time) data array to a timeseries
        # E.G. lambda x: np.mean(x, axis=0)
        self.ts_arr = ts_xform(d_array)

        # Reshape the data as (ncol, nrow, ntime) to keep it contiguous...
        # this will also correspond to (k,j,i) indexing. Then flatten
        # and reshape the array to (ntime, nrow, ncol) (z,y,x) to
        # satisfy the VTKImageData column-major format
        new_shape = (npts, nrow, ncol)
        vtk_arr = np.reshape(d_array.transpose(), new_shape, order='F')
        # make the time dimension unit length, and put the origin at -1/2
        self.arr_img_dsource.spacing = 1./npts, 1., 1.
        self.arr_img_dsource.origin = (-0.5, 0.0, 0.0)
        #self.arr_img_dsource.scalar_data = self.arr_img_data
        self.arr_img_dsource.scalar_data = vtk_arr

        # set up long scale plot and zoom plots
        self.long_axes = self.ts_plot.add_subplot(111)
        self._plot_timeseries(self.long_axes, self.ts_arr)
        self.ts_mark = self.long_axes.axvline(
            x=self._t0, color='r', ls='-'
            )

        ## self.zoom_axes = self.zoom_plot.add_subplot(111)
        ## self._plot_timeseries(self.zoom_axes, ts_arr)
        ## self.zoom_mark = self.zoom_axes.axvline(
        ##     x=self._t0, color='r', ls=':'
        ##     )
        # set up zoom window
        self.zoom_axes = self.zoom_plot.add_subplot(111)

        # now that all elements are set, hit some traits callbacks
        ## self.time = 0
        ## #self.eps = 0 # 0 radians
        ## self.tau = 1.0
        if 'time' not in traits:
            traits['time'] = 0
        if 'tau' not in traits:
            traits['tau'] = 1.0
        HasTraits.__init__(self, arr_img_data=vtk_arr, **traits)
        self._scrolling = False

        self._plot_new_zoom()
        self.zoom_axes.xaxis.set_visible(False)
        self.zoom_mark = self.zoom_axes.axvline(x=0, color='r', ls=':')
        
    def _plot_timeseries(self, axes, arr):
        n = arr.shape[-1]
        if arr.ndim == 1:
            tx = np.linspace(self._t0, self._tf, n)
            axes.plot(tx, arr)
            true_eps = self.__map_radians(self.eps)
            lim = (-true_eps, true_eps)
            axes.set_ylim(lim)
        elif arr.ndim == 2:
            raise NotImplementedError

    def configure_traits(self, *args, **kwargs):
        super(DataScroller, self).configure_traits(*args, **kwargs)
        self._connect_mpl_events()
        self._blit_state()

    def edit_traits(self, *args, **kwargs):
        super(DataScroller, self).edit_traits(*args, **kwargs)
        self._connect_mpl_events()
        self._blit_state()

    def zoom_data(self):
        d = self.ts_arr
        n_pts = d.shape[-1]
        n_zoom_pts = int(np.round(self.tau*self.Fs))
        zoom_start = int(np.round(self.Fs*(self.time - self.tau/2)))
        if zoom_start < 0:
            # create an array padded in front with NaNs
            zoom_arr = np.empty((n_zoom_pts,), 'd')
            zoom_arr.fill(np.nan)
            zoom_arr[..., -zoom_start:] = \
              d[..., :n_zoom_pts + zoom_start]
        elif n_pts > zoom_start and zoom_start > n_pts - n_zoom_pts:
            # create an array padded at end with NaNs
            zoom_arr = np.empty((n_zoom_pts,), 'd')
            zoom_arr.fill(np.nan)
            zoom_arr[..., :(n_pts-zoom_start)] = \
              d[..., zoom_start:]
        elif zoom_start >= n_pts:
            zoom_arr = np.empty((n_zoom_pts,), 'd')
            zoom_arr.fill(np.nan)
        else:
            zoom_arr = d[...,zoom_start:(zoom_start+n_zoom_pts)]
        return zoom_arr

    def _connect_mpl_events(self):
        # connect a sequence of callbacks to
        # click -> enable scrolling
        # drag -> scroll time bar (if scrolling)
        # unclick -> disable scrolling
        self.ts_plot.canvas.mpl_connect(
            'button_press_event', self._scroll_handler
            )
        self.ts_plot.canvas.mpl_connect(
            'button_release_event', self._scroll_handler
            )
        self.ts_plot.canvas.mpl_connect(
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
        # zoom plot limits -- same logic as update zoom interval
        #self._update_tau()
        self.ts_mark.set_data(( [self.time, self.time], [0, 1] ))
        #self.zoom_mark.set_data(( [self.time, self.time], [0, 1] ))
        self._plot_new_zoom()
        # array image
        if self.array_ipw:
            self.array_ipw.ipw.slice_index = int( np.round(self.time*self.Fs) )

        # XXX: questionable to check the status of only one figure --
        # which is why the whole drawing mechanism should be encapsulated
        # in a separate class
        if (self._saved_ts_size == self.long_axes.bbox.size).all():
            self._draw_zoom_line()
            self._draw_ts_mark()
        else:
            # fully draw both if the figure has been resized
            self._blit_state()

    def __map_radians(self, eps):
        max_amp = 1.0
        return max_amp*((np.sin(np.pi*(self.eps-1/2.0))+1)/2.0)**2
        
    @on_trait_change('eps')
    def _update_eps(self):
        max_amp = 1.0
        true_eps = self.__map_radians(self.eps)
        lim = (-true_eps, true_eps)
        self.zoom_axes.set_ylim(lim)
        self.long_axes.set_ylim(lim)
        # Change color LUT of array image
        if self.array_ipw:
            self.array_ipw.module_manager.scalar_lut_manager.data_range = lim
        # draw the figures and save the new background
        self._blit_state()

    ## @on_trait_change('tau')
    ## def _update_tau(self):
    ##     lim = (self.time - self.tau/2, self.time + self.tau/2)
    ##     self.zoom_axes.set_xlim(lim)
    ##     self._blit_state(self.zoom_plot)

    @on_trait_change('tau')
    def _plot_new_zoom(self):
        ax = self.zoom_axes
        zdata = self.zoom_data()
        if hasattr(self, 'zoom_line'):
            x, y = self.zoom_line.get_data()
            if zdata.shape == y.shape:
                self.zoom_line.set_data(x, zdata)
                return
        t0 = np.round(-self.tau*self.Fs/2)
        n_zoom_pts = np.round(self.tau*self.Fs)
        tx = np.arange(t0, t0+n_zoom_pts)
        if not hasattr(self, 'zoom_line'):
            self.zoom_line = ax.plot(tx, zdata)[0]
        else:
            self.zoom_line.set_data(tx, zdata)
        ax.set_xlim(t0, t0+n_zoom_pts)
        self._blit_state()

    def _blit_state(self):
        if not (self.ts_plot.canvas and self.zoom_plot.canvas):
            return
        # time series plot
        ax = self.long_axes
        self._saved_ts_size = ax.bbox.size
        # set time mark invisible then save background
        self.ts_mark.set_visible(False)
        self.ts_plot.canvas.draw()
        self.ts_bkgrnd = self.ts_plot.canvas.copy_from_bbox(ax.bbox)
        self.ts_mark.set_visible(True)
        self.ts_plot.canvas.draw()
        # zoom plot
        ax = self.zoom_axes
        self._saved_zoom_size = ax.bbox.size
        self.zoom_line.set_visible(False)
        self.zoom_plot.canvas.draw()
        self.zoom_bkgrnd = self.zoom_plot.canvas.copy_from_bbox(ax.bbox)
        self.zoom_line.set_visible(True)
        self.zoom_plot.canvas.draw()
        
    # call this for an update of the vertical line
    def _draw_ts_mark(self):
        ax = self.long_axes
        canvas = self.ts_plot.canvas
        if self.ts_bkgrnd is not None:
            canvas.restore_region(self.ts_bkgrnd)
        ax.draw_artist(self.ts_mark)
        canvas.blit(ax.bbox)

    # call this for an update of the zoom plot
    def _draw_zoom_line(self):
        ax = self.zoom_axes
        canvas = self.zoom_plot.canvas
        if self.zoom_bkgrnd is not None:
            canvas.restore_region(self.zoom_bkgrnd)
        ax.draw_artist(self.zoom_line)
        canvas.blit(ax.bbox)
        
    # call this for a full draw
    ## def _draw_figures(self, *args):
    ##     if not len(args):
    ##         args = (self.zoom_plot, self.ts_plot)
    ##     for plot in args:
    ##         if plot.canvas:
    ##             if plot is self.ts_plot:
    ##                 self._blit_state()
    ##             else:
    ##                 plot.canvas.draw()


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
            t1 = time()
            self._update_time()
            t2 = time()
            s_time = max(0., 1/self.fps - t2 + t1)
            print 'time to draw:', (t2-t1)
            sleep(s_time)
            
    ## mayavi components
    
    @on_trait_change('array_scene.activated')
    def _display_image(self):
        scene = self.array_scene
        eps = self.__map_radians(self.eps)
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
                    show_label=False, width=700, height=50, resizable=True
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

    @on_trait_change('view')
    def _test(self):
        print 'foo'
        self._connect_mpl_events()
    
if __name__ == "__main__":
    pass
   
