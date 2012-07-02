import numpy as np

# std lib
from threading import Thread
from time import sleep, time
import random


# ETS Traits
from traits.api import \
     HasTraits, Instance, on_trait_change, Float, Button, Range, Int 
from traitsui.api import Item, View, VGroup, HGroup, RangeEditor

# Mayavi/TVTK
from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk
from mayavi.sources.api import ArraySource

import plot_modules as pm


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
    zoom_plot = Instance(pm.ScrollingTimeSeriesPlot)
    ts_plot = Instance(pm.StaticTimeSeriesPlot)

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
            format='%1.2f', low_label='tight', high_label='wide'
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

    def __init__(self, d_array, ts_array, rowcol=(), Fs=1.0, **traits):
        """
        Display a channel array in a 3-plot format:

        * array image in native array geometry
        * long-scale time series navigator plot
        * short-scale time series zoomed plot

        Parameters
        ----------

        d_array: ndarray, 2D or 3D
          the array recording, in either (n_chan, n_time) or
          (n_row, n_col, n_time) format

        ts_array: ndarray, 1D
          a (currently single) timeseries description of the recording

        rowcol: tuple
          the array geometry, if it cannot be inferred from d_array

        Fs: float
          sampling rate

        traits: dict
          other keyword parameters
        
        """
        npts = d_array.shape[-1]
        if len(d_array.shape) < 3:
            nrow, ncol = rowcol
        else:
            nrow, ncol = d_array.shape[:2]
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
        return pm.StaticTimeSeriesPlot(
            t, self.ts_arr, figsize=figsize, ylim=(-eps, eps), t0=t0,
            line_props=lprops
            )

    def construct_zoom_plot(self, figsize, eps, **lprops):
        x = self.zoom_data()
        return pm.ScrollingTimeSeriesPlot(
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
        print 'events connected'

    def _scroll_handler(self, ev):
        if not ev.inaxes:
            return
        if not self._scrolling and ev.name == 'button_press_event':
            self._scrolling = True
            self._scroll_handler(ev)
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
                    'zoom_plot', editor=pm.MPLFigureEditor(),
                    show_label=False, width=500, height=200, resizable=True
                    )
                ),
            HGroup(
                Item(
                    'ts_plot', editor=pm.MPLFigureEditor(),
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

    zoom_plot = Instance(pm.ScrollingColorCodedPlot)
    ts_plot = Instance(pm.StaticColorCodedPlot)

    def __init__(
            self, d_array, ts_array, cx_array, 
            rowcol=(), Fs=1.0, **traits
            ):
        """
        Display a channel array in a 3-plot format:

        * array image in native array geometry
        * long-scale time series navigator color-coded plot
        * short-scale time series zoomed, color-coded plot

        The timeseries plots are structured as points located at
        (time, amplitude) points, but color-coded based on values in
        the co-function cx_array.

        Parameters
        ----------

        d_array: ndarray, 2D or 3D
          the array recording, in either (n_chan, n_time) or
          (n_row, n_col, n_time) format

        ts_array: ndarray, 1D
          a (currently single) timeseries description of the recording

        cx_array: ndarray, 1D
          a secondary timeseries which color-codes the ts_array plots

        rowcol: tuple
          the array geometry, if it cannot be inferred from d_array

        Fs: float
          sampling rate

        traits: dict
          other keyword parameters
        
        """
        self.cx_arr = cx_array
        DataScroller.__init__(
            self, d_array, ts_array, rowcol=rowcol, Fs=Fs, **traits
            )

    def construct_ts_plot(self, t, figsize, eps, t0, **lprops):
        dfac = 1
        t = t[::dfac]
        ts_arr = self.ts_arr[::dfac]
        cx_arr = self.cx_arr[::dfac]
        return pm.StaticColorCodedPlot(
            t, ts_arr, cx_arr,
            figsize=figsize, ylim=(-eps, eps), t0=t0,
            line_props=lprops
            )
    
    def construct_zoom_plot(self, figsize, eps, **lprops):
        x, cx = self.zoom_data()
        cx_lim = self.ts_plot.cx_limits # ooh! hacky
        return pm.ScrollingColorCodedPlot(
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
    import sys
    nrow = 10; ncol = 15; n_pts = 1000
    d = np.random.randn(nrow*ncol, n_pts)
    d_mx = d.max(axis=0)
    if len(sys.argv) < 2:
        dscroll = DataScroller(d, d_mx, rowcol=(nrow, ncol), Fs=1.0)
        dscroll.configure_traits()
    else:
        # if ANY args, do color coded test
        cx = np.random.randn(len(d_mx))
        dscroll = ColorCodedDataScroller(
            d, d_mx, cx, rowcol=(nrow, ncol), Fs=1.0
            )
        dscroll.configure_traits()
