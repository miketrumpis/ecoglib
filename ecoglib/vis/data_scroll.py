import numpy as np
from matplotlib.lines import Line2D

# ETS Traits
from traits.api import \
     HasTraits, Instance, on_trait_change, Float, Button, \
     Range, Int, Any, Bool
from traitsui.api import Item, View, VGroup, HGroup, \
     RangeEditor, HSplit, VSplit

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
import traitsui_bridge as tb
import ecoglib.util as ut

import ecogana.devices.units as units_tools

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
    ts_plot = Instance(pm.WindowedTimeSeriesPlot)
    # XXX: this should probably not be a trait
    ts_window_length = Float(50.)

    ## array scene, image, and data (Mayavi components)

    _has_video = Bool
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
    tau = Range(low=0.050, high=200.0, value=1.0)

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
        shape = list(d_array.shape)
        tdim = np.argmax(shape)
        npts = shape[tdim]
        shape.pop(tdim)
        if d_array.ndim == 3:
            rowcol = tuple(shape)
        else:
            # check that the shape is consistent
            if np.prod(rowcol) != shape[0]:
                rowcol = ()
        # Try to catch some cases where video is not intended
        if rowcol:
            try:
                vtk_arr = volumetric_data(d_array, rowcol)
                # make the time dimension unit length, 
                # and put the origin at -1/2
                self.arr_img_dsource.spacing = 1., 1., 1./npts
                self.arr_img_dsource.origin = (0.0, 0.0, -0.5)
                self.arr_img_dsource.scalar_data = vtk_arr
                self._has_video = True
            except NotImplementedError:
                self._has_video = False
                print 'no video due to implementation problem'
        else:
            # if still no shape info, then no video
            self._has_video = False
        
        self._tf = float(npts-1) / Fs
        self.Fs = Fs
        self.max_ts_amp = ts_array.max()
        self.min_ts_amp = ts_array.min()
        # the timeseries peaks are probably a good starting
        # point for the array peaks
        #self.base_arr_amp = max( self.max_ts_amp, -self.min_ts_amp )
        # can spare a couple O(N) ops here
        self.base_arr_amp = 1.1*max( d_array.max(), -d_array.min() )

        self.ts_arr = ts_array

        # pop out some traits that should be set after initialization
        time = traits.pop('time', 0)
        #tau = traits.pop('tau', 1.0)
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
        lim = self._map_eps(i_eps, (self.min_ts_amp, self.max_ts_amp))
        figsize=()
        self.ts_plot = self.construct_ts_plot(
            tx, figsize, lim, time, linewidth=1
            )
        self.sync_trait('time', self.ts_plot, mutual=True)

        # configure the zoomed plot
        figsize=()
        self.zoom_plot = self.construct_zoom_plot(tx, figsize, lim)
        self.sync_trait('time', self.zoom_plot, mutual=True)

        #self.trait_setq(tau=tau)
        self.trait_setq(time=time)
        self.trait_setq(eps=i_eps)

    def construct_ts_plot(self, t, figsize, lim, t0, **lprops):
        return pm.WindowedTimeSeriesPlot(
            t, self.ts_arr, ylim=lim, t0=t0,
            window_length=self.ts_window_length,
            plot_line_props=lprops
            )

    def construct_zoom_plot(self, t, figsize, lim, **lprops):
        return pm.ScrollingTimeSeriesPlot(
            t, self.ts_arr, self.tau, 
            plot_line_props=lprops, ylim=lim
            )

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

    def _map_eps(self, eps, limits):
        p = ((np.sin(np.pi*(eps-1/2.0))+1)/2.0)**2
        mn, mx = limits
        half_width = (mx - mn)*p / 2.0
        md = (mx + mn)/2.0
        return (md - half_width, md + half_width)

    @on_trait_change('eps')
    def _update_eps(self):
        full_limits = max(abs(self.min_ts_amp), abs(self.max_ts_amp))
        #lim = self._map_eps(self.eps, (self.min_ts_amp, self.max_ts_amp))
        lim = self._map_eps(self.eps, (-full_limits, full_limits))
        self.ts_plot.ylim = lim
        if self.zoom_plot:
            self.zoom_plot.ylim = lim

    @on_trait_change('arr_eps')
    def _update_arr_eps(self):
        lim = self._map_eps(
            self.arr_eps, (-2*self.base_arr_amp, 2*self.base_arr_amp)
            )
        if self.array_ipw:
            self.array_ipw.module_manager.scalar_lut_manager.data_range = lim

    @on_trait_change('tau')
    def _plot_new_zoom(self):
        if self.zoom_plot:
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
        if not self._has_video:
            return
        scene = self.array_scene
        lim = self._map_eps(
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
        mlab.axes(
            figure=scene.mayavi_scene, z_axis_visibility=False,
            xlabel='column', ylabel='row'
            )
        self.array_ipw = ipw

    def default_traits_view(self):
        view = View(
            VGroup(
                HGroup(
                    Item(
                        'array_scene', editor=SceneEditor(scene_class=Scene),
                        height=200, width=200, show_label=False,
                        enabled_when='_has_video'
                        ),
                    Item(
                        'zoom_plot', editor=tb.MPLFigureEditor(),
                        show_label=False, width=500, height=200, resizable=True
                        )
                    ),
                HGroup(
                    Item(
                        'ts_plot', editor=tb.MPLFigureEditor(),
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
            title='Data Scroller',
            handler=tb.PingPongStartup()
        )
        return view

class ColorCodedDataScroller(DataScroller):

    zoom_plot = Instance(pm.ScrollingColorCodedPlot)
    ts_plot = Instance(pm.WindowedColorCodedPlot)

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
          to window and scale limits can be quite slow with many color-coded
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
        return pm.WindowedColorCodedPlot(
            t, ts_arr, cx_arr,
            ylim=lim, t0=t0,
            window_length=self.ts_window_length,
            plot_line_props=lprops
            )

    def construct_zoom_plot(self, t, figsize, lim, **lprops):
        cx_lim = self.ts_plot.cx_limits # ooh! hacky
        return pm.ScrollingColorCodedPlot(
            t, self.ts_arr, self.tau, self.cx_arr,
            cx_limits=cx_lim,
            ylim=lim, plot_line_props=lprops
            )

class ClassCodedDataScroller(DataScroller):

    zoom_plot = Instance(pm.ScrollingClassSegmentedPlot)
    ts_plot = Instance(pm.WindowedClassSegmentedPlot)

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
        return pm.WindowedClassSegmentedPlot(
            t, ts_arr, labels,
            ylim=lim, t0=t0,
            window_length=self.ts_window_length,
            plot_line_props=lprops
            )

    def construct_zoom_plot(self, t, figsize, lim, **lprops):
        return pm.ScrollingClassSegmentedPlot(
            t, self.ts_arr, self.tau, self.labels,
            ylim=lim, plot_line_props=lprops
            )

class ChannelScroller(DataScroller):
    """View all array channels at once, with marked stimulation events
    """
    ts_plot = Instance(pm.PagedTimeSeriesPlot)
    _zero = Int(0)
    page = Range(low='_zero', high='_mx_page')
    page_length = Range(low=10, high=500000)
    _mx_page = Int
    page_up = Button()
    page_dn = Button()
    draw_stims = Bool(False)
    show_zoom = Bool(False)
    _has_stim = Bool
    channel_scale = Range(low=0.0, high=1.0, value=1.0)
    window_shift = Range(low=-1.0, high=1.0, value=0.0)
    auto_scale = Bool(True)
    
    def __init__(
            self, array_data, page_len, chans=(), exp=None,
            Fs=1.0, units='', **traits_n_kw
            ):

        t_axis = np.argmax(array_data.shape)
        npts = array_data.shape[t_axis]
        ts_arr = np.rollaxis(array_data, t_axis).reshape(npts, -1)
        # XXX: the following indexing relies on the array_data and
        # channel mapping being in the correct relative array ordering.
        # This indexing reverses the recording channel-to-array map 
        # back to recording channel order
        if chans:
            # for now, do this trick to get channels arranged as
            # (i,j) instead of (x,y)
            ## c_order = ut.mat_to_flat(
            ##     chans.geometry, *chans.to_mat(), 
            ##     col_major=not chans.col_major
            ##     )
            # in vtk, channel (i,j) gets plotted at cartesian (x,y)
            # 1) swap (i,j) for (j,i)
            # 2) stack the channels in raster-order

            # get the data channels, should be in col-major order 
            ts_arr = ts_arr[:,sorted(chans)]
        self.chans = chans
        self.page_length = int( round(Fs * page_len) )
        self._mx_page = int( npts // self.page_length )
        self.exp = exp
        self.units = units
        traits_n_kw['tau'] = page_len
        DataScroller.__init__(
            self, array_data, ts_arr, Fs=Fs, _has_stim = (exp is not None),
            **traits_n_kw
            )
        self.draw_events()
        self.sync_trait('page', self.ts_plot, mutual=True)
        self.sync_trait('page_length', self.ts_plot, mutual=True)
        self._mx_spacing = self.ts_plot.current_spacing

    @staticmethod
    def from_dataset_bunch(dset, window):
        # this trick is for the x,y indexing convention of VTK
        vtk_cmap = dset.chan_map.as_col_major()
        vtk_cmap.col_major = False
        vtk_cmap.geometry = vtk_cmap.geometry[::-1]
        exp = getattr(dset, 'exp', None)
        scr = ChannelScroller(
            vtk_cmap.embed(dset.data, axis=0, fill=0), 
            window, chans=vtk_cmap, exp=exp, Fs=dset.Fs,
            units=dset.units
            )
        return scr
        
    def construct_ts_plot(self, t, figsize, lim, t0, **lprops):
        lprops.setdefault('color', 'b')
        lprops['linewidth'] = 0.5
        n_lines = self.ts_arr.shape[1]
        plot = pm.PagedTimeSeriesPlot(
            t, self.ts_arr, self.page_length, stack_traces=True,
            t0=t0, plot_line_props=lprops
            )
        plot.n_yticks = int(n_lines)
        #plot.ax.set_yticks(np.arange(n_lines) * plot._spacing)
        ## plot.ax.set_yticks(
        ##     np.linspace(plot.ylim[0], plot.ylim[1], n_lines)
        ##     )
        #from matplotlib.ticker import LinearLocator
        #plot.ax.yaxis.set_major_locator(LinearLocator(numticks=n_lines))
        if isinstance(self.chans, ut.ChannelMap):
            # these indices are potentially transposed due to VTk hack
            # <but impossible to know!!>
            ii, jj = self.chans.to_mat()
            #jj, ii = zip(*sorted(zip(jj, ii)))
            ii, jj = zip(*sorted(zip(ii, jj)))
            c_num = self.chans.lookup(ii, jj)

            plot.ax.set_yticklabels(
                ['%d: (%d, %d)'%x for x in zip(c_num, jj, ii)],
                fontsize=8
                )
        else:
            plot.ax.set_yticklabels( 
                ['%s'%n for n in xrange(n_lines)], fontsize=8
                )
        if self.units:
            units = self.units
            pos = plot.ax.get_position()
            scl_ax = plot.fig.add_axes([0.8, pos.y0, 0.08, pos.y1-pos.y0])
            scl_ax.axis('off')

            ylim = plot.ax.get_ylim()
            y_scale = 2*float(ylim[1] - ylim[0]) / n_lines
            scale_step, scaling, units = \
              units_tools.best_scaling_step(y_scale, units, allow_up=True)
            bar_len = np.floor( scaling*y_scale / scale_step ) * scale_step
            bar_text = '%d %s'%(bar_len, units_tools.nice_unit_text(units))
            bar_len /= scaling
            #bar_len = 200e-6
            scl_ax.add_line(
                Line2D([0, 0], [bar_len, 2*bar_len], color='k', linewidth=3)
                )
            scl_ax.text(
                .25*bar_len, 1.5*bar_len, bar_text, ha='left', va='center'
                )
            scl_ax.set_ylim(ylim)
            scl_ax.set_xlim(-bar_len, bar_len)
            self.scl_ax = scl_ax
        else:
            self.scl_ax = None
        if not self.auto_scale:
            self._set_manual_scaling()
        return plot

    def construct_zoom_plot(self, t, figsize, lim, **lprops):
        return pm.ScrollingTimeSeriesPlot(
            t, self.ts_arr.mean(axis=1), self.tau, 
            plot_line_props=lprops, ylim=lim
            )
            
    def draw_events(self):
        self._event_lines = list()
        if not (self.draw_stims and self._has_stim):
            return
        ylim = self.ts_plot.ylim
        delta = (ylim[1] - ylim[0]) / 100
        page = self.page
        plen = self.page_length
        try:
            events = self.exp.time_stamps
        except AttributeError:
            events = self.exp.trig_times
        plotted_stims = (events >= page*plen) & (events < (page+1)*plen)
        stim_idx = np.where(plotted_stims)[0]
        for idx in stim_idx:
            time = events[idx]/self.Fs
            ln = self.ts_plot.ax.axvline(
                x=time, color=[.25, .25, .25, .8], 
                linestyle='--', linewidth=2
                )
            s = self.exp.stim_str(idx, mpl_text=True)
            s.set_position( (time, ylim[1]+delta) )
            s.set_transform(self.ts_plot.ax.transData)
            s.set_clip_on(False)
            s.update(dict(va='baseline', ha='center', fontsize=7))
            self.ts_plot.ax.add_artist(s)
            self._event_lines.extend( (ln, s) )
            self.ts_plot.add_static_artist( (ln, s) )
            
        self.ts_plot.draw()

    def undraw_events(self):
        self.ts_plot.remove_static_artist(self._event_lines)
        self._event_lines = list()

    def _change_page(self, page):
        self.undraw_events()
        self.page = int(page)
        if self.scl_ax:
            self.scl_ax.set_ylim(self.ts_plot.ax.get_ylim())
        self.trait_setq(window_shift=0)
        self.draw_events()
        
    def _page_up_fired(self):
        if self.page < self._mx_page:
            self._change_page(self.page+1)

    def _page_dn_fired(self):
        if self.page > 0:
            self._change_page(self.page-1)

    @on_trait_change('channel_scale')
    def _change_scale(self):
        if self.auto_scale:
            return
        a = self.channel_scale
        # map from [0,1] --> 25 to mx_spacing microvolts
        spacing = a * self._mx_spacing + (1-a) * 25e-6
        self.undraw_events()
        # XXX: nice to get a way to keep this from triggering a draw
        self.ts_plot.stack_spacing = spacing
        if self.scl_ax:
            self.scl_ax.set_ylim(self.ts_plot.ax.get_ylim())
        self.draw_events()
        
    @on_trait_change('window_shift')
    def _change_window(self):
        xlim = self.ts_plot.xlim
        twin = xlim[1] - xlim[0]
        # map from -1 to +1 twin
        t_off = self.window_shift * twin
        self.ts_plot.center_page(t_off)
        
    @on_trait_change('auto_scale')
    def _set_manual_scaling(self):
        if self.auto_scale:
            self.undraw_events()
            self.ts_plot.stack_spacing = 0
            if self.scl_ax:
                self.scl_ax.set_ylim(self.ts_plot.ax.get_ylim())
            self.draw_events()
            return
        self._mx_spacing = self.ts_plot.current_spacing
        self._change_scale()
        
    @on_trait_change('page_length')
    def _change_mx_page(self):
        self._mx_page = int( self.ts_arr.shape[0] // self.page_length ) + 1
        self.ts_plot.page_length = self.page_length
        self.page = self.ts_plot.page
        self._change_page(self.page)

    @on_trait_change('draw_stims')
    def _stim_handler(self):
        if self.draw_stims and not len(self._event_lines):
            self.draw_events()
        if not self.draw_stims:
            self.undraw_events()
            self.ts_plot.draw()
        
    @on_trait_change('eps')
    def _update_eps(self):
        full_limits = max(abs(self.min_ts_amp), abs(self.max_ts_amp))
        #lim = self._map_eps(self.eps, (self.min_ts_amp, self.max_ts_amp))
        lim = self._map_eps(self.eps, (-full_limits, full_limits))
        self.zoom_plot.ylim = lim

    def default_traits_view(self):
        view = View(
            HSplit(
                Item(
                    'ts_plot', editor=tb.MPLFigureEditor(), show_label=False,
                    width=400, height=900, resizable=True
                    ),
                VSplit(
                    Item(
                        'array_scene', editor=SceneEditor(scene_class=Scene),
                        height=500, width=500, show_label=False,
                        visible_when='_has_video'
                        ),
                    VSplit(
                        Item(
                            'zoom_plot', editor=tb.MPLFigureEditor(), 
                            show_label=False,
                            width=400, height=150, resizable=True
                            ),
                        HGroup(
                            Item('tau', label='Zoom Width'),
                            Item('eps', label='Array Limits')
                            ),
                        visible_when='show_zoom'
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
                            Item('window_shift', label='Shift Win'),
                            Item('draw_stims', label='Draw Stim Events',
                                 enabled_when='_has_stim>0'),
                            Item('show_zoom', label='Plot CAR')
                            ),
                        VGroup(
                            Item('arr_eps', label='Array Color'),
                            Item('count', label='Run Clock'),
                            Item('fps', label='FPS')
                            )
                        )
                    )
                ),
            resizable=True, title='ChannelScroller',
            handler=tb.PingPongStartup()
        )
        return view
                
                
                 
        
    
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
