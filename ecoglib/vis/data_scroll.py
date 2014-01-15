import numpy as np

# std lib
import random


# ETS Traits
from traits.api import \
     HasTraits, Instance, on_trait_change, Float, Button, Range, Int, Any
from traitsui.api import Item, View, VGroup, HGroup, RangeEditor

# Mayavi/TVTK
from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk
from mayavi.sources.api import ArraySource

# Pyface Timer
from pyface.timer.api import Timer

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
            sx[:(lx-start), ...] = x[start:, ...]
            sx[(lx-start):, ...] = fill
    else:
        sx = x[start:start+num, ...]
    return sx

#### Utility to prepare volumetric data for VTK without
#### resorting to mem copying (if possible)
def volumetric_data(data, rowcol):

    # quick conditions
    # data is C-contiguous and shaped (npt, nchan) or (npt, ncol, nrow)
    # -- data may have come from MATLAB
    # data is C-contiguous and shaped (nchan, npt) or (ncol, nrow, npt)
    # -- data is probably from a HDF5 table format
    #
    #   (Other quick conditions exist for F-contiguous, deal with later)
    #

    shape = data.shape
    t_ax = np.argmax(shape)
    npt = shape[t_ax]
    if (t_ax == 0) and data.flags.c_contiguous:
        if len(shape) > 2:
            assert (shape[1] == rowcol[1]) and (shape[0] == rowcol[0])
        else:
            assert shape[1] == rowcol[0]*rowcol[1]
        # If the array is at base shaped (nsamp, nsite) and c-contiguous,
        # then the memory layout corresponds to axes (in ascending order):
        # (nrow, ncol, nsamp) -- this is a column-major MATLAB artifact.
        #
        # Now, set the shape to reflect the shape/stride relationship:
        # (Z, Y, X) <--> (ncol*nrow, ncol, 1)
        # Reverse the ordering of axes to create this (column-major)
        # relationship, which satisfies VTK
        # (X, Y, Z) <--> (1, ncol, ncol*nrow)
        #
        # The final transpose does not copy memory but just changes
        # the view to be F-contiguous
        final_shape_t = (npt, rowcol[1], rowcol[0])
        # return a F-contiguous array in the correct shape (no copy)
        return np.reshape(data, final_shape_t).transpose()
    if (t_ax > 0) and data.flags.c_contiguous:
        # simpler case.. just unpack the 0th axis if it is packed
        if len(shape) > 2:
            return data
        else:
            return data.reshape( rowcol + (npt,) )
    else:
        raise NotImplementedError('work in progress')

class DataScroller(HasTraits):

    ## these may need to be more specialized for handling 1D/2D timeseries

    zoom_plot = Instance(pm.ScrollingTimeSeriesPlot)
    ts_plot = Instance(pm.PagedTimeSeriesPlot)
    # XXX: this should probably not be a trait
    ts_page_length = Float(50.)

    ## array scene, image, and data (Mayavi components)

    array_scene = Instance(MlabSceneModel, ())
    arr_img_dsource = Instance(ArraySource, (), transpose_input_array=False)
    array_ipw = Instance(PipelineBase)
    arr_eps = Range(
        low=0.0, high=1.0, value=0.5,
        editor=RangeEditor(
            format='%1.2f', low_label='tight', high_label='wide'
            )
        )

    ## view controls

    # interval for zoom plot (units sec)
    tau = Range(low=1.0, high=50.0, value=1.0)

    # limits for abs-amplitude (auto tuned)

    # going to map this to max_amp*(sin(pi*(eps-1/2)) + 1)/2 to
    # prevent blowing up the ylim too easily with the range slider
    eps = Range(
        low=0.0, high=1.0, value=1.0,
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
    t_counter = Any()

    def __init__(
            self, d_array, ts_array,
            rowcol=(), Fs=1.0, tx=None, **traits
            ):
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
        if len(d_array.shape) < 3:
            nrow, ncol = rowcol
        else:
            ncol, nrow = d_array.shape[1:]
        vtk_arr = volumetric_data(d_array, (nrow, ncol))
        npts = vtk_arr.shape[-1]
        self._tf = float(npts-1) / Fs
        self.Fs = Fs
        # XXX: should set max_amp -- could stochastically sample to
        # estimate mean and variance
        #self.max_amp = stochastic_limits(ts_array)
        self.max_ts_amp = ts_array.max()
        self.min_ts_amp = ts_array.min()
        # the timeseries peaks are probably a good starting
        # point for the array peaks
        #self.base_arr_amp = max( self.max_ts_amp, -self.min_ts_amp )
        # can spare a couple O(N) ops here
        self.base_arr_amp = 1.1*max( d_array.max(), -d_array.min() )

        self.ts_arr = ts_array

        ## new_shape = (npts, ncol, nrow)
        ## vtk_arr = np.reshape(d_array, new_shape).transpose()
        # make the time dimension unit length, and put the origin at -1/2
        self.arr_img_dsource.spacing = 1., 1., 1./npts
        self.arr_img_dsource.origin = (0.0, 0.0, -0.5)
        self.arr_img_dsource.scalar_data = vtk_arr

        # pop out some traits that should be set after initialization
        time = traits.pop('time', 0)
        tau = traits.pop('tau', 1.0)
        i_eps = traits.pop('eps', 1.0)
        HasTraits.__init__(self, **traits)
        self._scrolling = False

        # configure the ts_plot
        n = self.ts_arr.shape[0]
        # XXX: disable tx for now.. too hairy
        if tx is not None:
            tx = None
        if tx is None:
            tx = np.linspace(self._t0, self._tf, n)
        else:
            assert len(tx)==n, 'provided time axis has wrong length'
        lim = self.__map_eps(i_eps, (self.min_ts_amp, self.max_ts_amp))
        figsize=(6,1)
        self.ts_plot = self.construct_ts_plot(
            tx, figsize, lim, time, linewidth=1
            )
        self.sync_trait('time', self.ts_plot, mutual=True)

        # configure the zoomed plot
        figsize=(5,2)
        self.zoom_plot = self.construct_zoom_plot(tx, figsize, lim)
        self.sync_trait('time', self.zoom_plot, mutual=True)

        self.trait_setq(tau=tau)
        self.trait_setq(time=time)
        self.trait_setq(eps=i_eps)

    def construct_ts_plot(self, t, figsize, lim, t0, **lprops):
        return pm.PagedTimeSeriesPlot(
            t, self.ts_arr, figsize=figsize, ylim=lim, t0=t0,
            page_length=self.ts_page_length,
            line_props=lprops
            )

    def construct_zoom_plot(self, t, figsize, lim, **lprops):
        return pm.ScrollingTimeSeriesPlot(
            t, self.ts_arr, self.tau, 
            line_props=lprops, figsize=figsize, ylim=lim
            )

    def configure_traits(self, *args, **kwargs):
        ui = super(DataScroller, self).configure_traits(*args, **kwargs)
        self._post_canvas_hook()
        return ui

    def edit_traits(self, *args, **kwargs):
        ui = super(DataScroller, self).edit_traits(*args, **kwargs)
        self._post_canvas_hook()
        return ui

    def _post_canvas_hook(self):
        self.ts_plot.connect_live_interaction()
        self.zoom_plot.connect_live_interaction()

    @on_trait_change('time')
    def _update_time(self):
        # 1) long ts_plot should be automatically linked to 'time'
        # 2) zoomed ts_plot needs new x data (**no longer**)
        # 3) VTK ImagePlaneWidget needs to be resliced
        if self.array_ipw:
            self.array_ipw.ipw.slice_index = int( np.round(self.time*self.Fs) )
            # It looks like ipw.ipw.reslice.output_{extent,spacing}
            # need to be set and enforced each time slice_index changes
            ipw = self.array_ipw
            xyz = self.arr_img_dsource.scalar_data.shape
            ipw.ipw.reslice.output_extent = \
              np.array([0, xyz[0], 0, xyz[1], 0, 0], 'd') - 0.5
            ipw.ipw.reslice.output_spacing = 1., 1., 1./xyz[2]

    def __map_eps(self, eps, limits):
        p = ((np.sin(np.pi*(eps-1/2.0))+1)/2.0)**2
        mn, mx = limits
        half_width = (mx - mn)*p / 2.0
        md = (mx + mn)/2.0
        return (md - half_width, md + half_width)

    @on_trait_change('eps')
    def _update_eps(self):
        lim = self.__map_eps(self.eps, (self.min_ts_amp, self.max_ts_amp))
        self.ts_plot.ylim = lim
        self.zoom_plot.ylim = lim

    @on_trait_change('arr_eps')
    def _update_arr_eps(self):
        lim = self.__map_eps(
            self.arr_eps, (-2*self.base_arr_amp, 2*self.base_arr_amp)
            )
        if self.array_ipw:
            self.array_ipw.module_manager.scalar_lut_manager.data_range = lim

    @on_trait_change('tau')
    def _plot_new_zoom(self):
        self.zoom_plot.winsize = self.tau

    def _count_fired(self):
        if self.t_counter is not None and self.t_counter.IsRunning():
            self.t_counter.Stop()
        else:
            self.t_counter = Timer(1000.0/self.fps, self._count)

    def _count(self):
        t = self.time + 1.0/self.Fs
        if t > self._tf:
            self.t_counter.Stop()
            return
        self.time = t

    ## mayavi components

    @on_trait_change('array_scene.activated')
    def _display_image(self):
        scene = self.array_scene
        lim = self.__map_eps(
            self.arr_eps, (-2*self.base_arr_amp, 2*self.base_arr_amp)
            )
        ipw = mlab.pipeline.image_plane_widget(
            self.arr_img_dsource,
            plane_orientation='z_axes',
            figure=scene.mayavi_scene,
            vmin=lim[0], vmax=lim[1],
            colormap='jet'
            )

        ipw.ipw.texture_interpolate = 0
        ipw.ipw.reslice_interpolate = 0
        ipw.ipw.slice_position = np.round(self.time * self.Fs)
        ipw.ipw.interaction = 0

        # It looks like ipw.ipw.reslice.output_{extent,spacing}
        # need to be set and enforced each time slice_index changes
        xyz = self.arr_img_dsource.scalar_data.shape
        ipw.ipw.reslice.output_extent = \
          np.array([0, xyz[0], 0, xyz[1], 0, 0], 'd') - 0.5
        ipw.ipw.reslice.output_spacing = 1., 1., 1./xyz[2]


        scene.mlab.view(distance=40)
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
                    show_label=False, width=600, height=100, resizable=True
                    ),
                Item('fps', label='FPS'),
                Item('count', label='Run Clock')
                ),
            HGroup(
                VGroup(
                    Item('tau', label='Zoom Interval'),
                    Item('eps', label='Amplitude Interval')
                    ),
                VGroup(
                    Item('arr_eps', label='Array Color'),
                    Item('time', label='Time Slice', style='custom')
                    )
                )
            ),
        resizable=True,
        title='Data Scroller'
    )

class ColorCodedDataScroller(DataScroller):

    zoom_plot = Instance(pm.ScrollingColorCodedPlot)
    ts_plot = Instance(pm.PagedColorCodedPlot)

    def __init__(
            self, d_array, ts_array, cx_array,
            rowcol=(), Fs=1.0, downsamp=10,
            **traits
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

        downsamp: int (Default 10)
          Downsample factor for the large-scale color coded plot (updates
          to page and scale limits can be quite slow with many color-coded
          points).

        traits: dict
          other keyword parameters

        """
        self.cx_arr = cx_array
        self._dfac = downsamp
        DataScroller.__init__(
            self, d_array, ts_array, rowcol=rowcol, Fs=Fs, **traits
            )

    def construct_ts_plot(self, t, figsize, lim, t0, **lprops):
        dfac = self._dfac
        t = t[::dfac]
        ts_arr = self.ts_arr[::dfac]
        cx_arr = self.cx_arr[::dfac]
        return pm.PagedColorCodedPlot(
            t, ts_arr, cx_arr,
            figsize=figsize, ylim=lim, t0=t0,
            page_length=self.ts_page_length,
            line_props=lprops
            )

    def construct_zoom_plot(self, t, figsize, lim, **lprops):
        cx_lim = self.ts_plot.cx_limits # ooh! hacky
        return pm.ScrollingColorCodedPlot(
            t, self.ts_arr, self.tau, self.cx_arr,
            cx_limits=cx_lim,
            figsize=figsize, ylim=lim, line_props=lprops
            )

class ClassCodedDataScroller(DataScroller):

    zoom_plot = Instance(pm.ScrollingClassSegmentedPlot)
    ts_plot = Instance(pm.PagedClassSegmentedPlot)

    def __init__(
            self, d_array, ts_array, labels,
            rowcol=(), Fs=1.0, **traits
            ):
        """
        Display a channel array in a 3-plot format:

        * array image in native array geometry
        * long-scale time series navigator class labeled plot
        * short-scale time series zoomed, class labeled plot

        Points in the timeseries plots are color coded to indicate
        classification according to the labels array.

        Parameters
        ----------

        d_array: ndarray, 2D or 3D
          the array recording, in either (n_chan, n_time) or
          (n_row, n_col, n_time) format

        ts_array: ndarray, 1D
          a (currently single) timeseries description of the recording

        labels: ndarray, 1D
          a secondary timeseries which color-codes the ts_array plots

        rowcol: tuple
          the array geometry, if it cannot be inferred from d_array

        Fs: float
          sampling rate

        traits: dict
          other keyword parameters

        """
        self.labels = labels
        DataScroller.__init__(
            self, d_array, ts_array, rowcol=rowcol, Fs=Fs, **traits
            )

    def construct_ts_plot(self, t, figsize, lim, t0, **lprops):
        ts_arr = self.ts_arr
        labels = self.labels
        return pm.PagedClassSegmentedPlot(
            t, ts_arr, labels,
            figsize=figsize, ylim=lim, t0=t0,
            page_length=self.ts_page_length,
            line_props=lprops
            )

    def construct_zoom_plot(self, t, figsize, lim, **lprops):
        return pm.ScrollingClassSegmentedPlot(
            t, self.ts_arr, self.tau, self.labels,
            figsize=figsize, ylim=lim, line_props=lprops
            )

if __name__ == "__main__":
    import sys
    from pyface.qt import QtGui
    nrow = 10; ncol = 15; n_pts = 1000
    d = np.random.randn(nrow*ncol, n_pts)
    d_mx = d.max(axis=0)
    if len(sys.argv) < 2:
        dscroll = DataScroller(d, d_mx, rowcol=(nrow, ncol), Fs=1.0)
        dscroll.edit_traits()
    else:
        # if ANY args, do color coded test
        cx = np.random.randn(len(d_mx))
        dscroll = ColorCodedDataScroller(
            d, d_mx, cx, rowcol=(nrow, ncol), Fs=1.0
            )
        dscroll.edit_traits()
    app = QtGui.QApplication.instance()
    app.exec_()
