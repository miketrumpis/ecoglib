import numpy as np

import ecoglib.vis.plot_modules as pm

from traits.api import \
     HasTraits, Range, Float, on_trait_change, Instance, Button, Int, Any
from traitsui.api import \
     RangeEditor, View, HGroup, VGroup, Item

from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk
from mayavi.sources.api import VTKDataSource

# Pyface Timer
from pyface.timer.api import Timer


class ScatterPlot(HasTraits):
    """
    This class manages the pipeline which plots and scrolls
    through the scatter points
    """

    scatter_src = Instance(VTKDataSource)
    scatter_pts = Instance(PipelineBase)

    inst_src = Instance(VTKDataSource)
    inst_pt = Instance(PipelineBase)

    trail_src = Instance(VTKDataSource)
    trail_pts = Instance(PipelineBase)
    trail_line = Instance(PipelineBase)
    _trail_length = Int(200)

    Fs = Float(1.0)
    time = Float(0.0)
    # Set this relative to the sampling period of the time series
    # -- e.g. if there is one scatter space point per 25 samples,
    #    then set scatter_time_scale = 25
    scatter_time_scale = Float(1.0)

    def __init__(self, scatter_pts, trail_length=200, **traits):
        traits['_trail_length'] = trail_length
        super(ScatterPlot, self).__init__(**traits)
        self.scatter_array = scatter_pts

    def setup_scatters(self, fig, scl_fn=None):
        s_array = self.scatter_array
        x, y, z = s_array.T

        # --- quick and dirty scale ---
        # this is approximately the number of points on each axis
        f_idx = np.isfinite(s_array[:,0])
        n_pt = len(np.unique(s_array[f_idx,0]))
        n_pt = len(f_idx)
        density = np.power(n_pt, -1/3.0)
        # this is approximately the length of each axis
        bb = s_array[f_idx].ptp(axis=0)
        relative_scale = np.power(np.prod(bb), 1/3.0)
        scale = relative_scale * density

        if scl_fn is None:
            self.scatter_src = mlab.pipeline.scalar_scatter(
                x, y, z, np.ones_like(x), figure=fig
                )
            self.scatter_pts = mlab.pipeline.glyph(
                self.scatter_src, mode='2dcircle', color=(0,0,1),
                scale_mode='none', scale_factor=scale/3
                )
        else:
            self.scatter_src = mlab.pipeline.scalar_scatter(
                x, y, z, scl_fn, figure=fig
                )
            self.scatter_pts = mlab.pipeline.glyph(
                self.scatter_src, mode='2dcircle',
                colormap='jet', scale_mode='none', scale_factor=scale/3
                )
        self.scatter_pts.glyph.glyph_source.glyph_source.filled = 1
        self.scatter_pts.actor.property.opacity = 0.75

        n = round(self.time*self.Fs)
        t_point = s_array[n][:,None]
        x, y, z = t_point
        # The scalar value used makes a light green in the Greens colormap.
        # This value will be the maximal value for the scalars representing
        # the trailing lines
        init_scl = np.array([0.6])
        self.inst_src = mlab.pipeline.scalar_scatter(
            x, y, z, init_scl, figure=fig
            )
        self.inst_pt = mlab.pipeline.glyph(
            self.inst_src, mode='sphere', colormap='Greens',
            scale_mode='none', scale_factor=scale*1.5, vmin=0.0, vmax=1.0
            )

        # copy the same for "trail" scatter
        # XXX: Could replace using a more light-weight point source.
        self.trail_src = mlab.pipeline.line_source(
            x, y, z, init_scl, figure=fig
            )
        self.trail_pts = mlab.pipeline.glyph(
            self.trail_src, mode='sphere', color=(0.7, 0.2, 0.2),
            scale_mode='none', scale_factor=scale
            )
        #self.trail_pts.actor.property.opacity = 0.4

        # and indeed use the same points for trail tube
        s = mlab.pipeline.stripper(self.trail_pts)
        t = mlab.pipeline.tube(s, tube_radius=scale/4)
        self.trail_line = mlab.pipeline.surface(
            t, colormap='Greens', opacity=0.3, vmin=0.0, vmax=1.0
            )

    @on_trait_change('time')
    def _update_time(self):
        n = round(self.time*self.Fs/self.scatter_time_scale)
        n = max( min( len(self.scatter_array)-1, n ), 0 )
        t_point = self.scatter_array[n][None,:]
        #self.inst_src.mlab_source.points = t_point
        self.inst_src.mlab_source.set(points = t_point)
        t_len = self._trail_length
        if not t_len:
            return
        trail_points = self.scatter_array[max(0,n+1-t_len):n+1]
        update_dict = dict(points = trail_points)
        if len(self.trail_src.mlab_source.scalars) < t_len or n < t_len:
            #if n < t_len:
            # XXX: referencing "maximum scalar"
            scalars = np.sqrt(np.linspace(0, 1.0, t_len))[-(n+1):]
            scalars *= 0.6 # maximum
            #print len(scalars), len(trail_points)
            update_dict['scalars'] = scalars
        #this is a safe and more slow update of the underlying VTK source
        self.trail_src.mlab_source.reset(**update_dict)
        #self.trail_src.mlab_source.reset(points = trail_points)


        ## else:
        ##     # this is faster and potentially bad update -- also try trail_src.set()
        ##     self.trail_src.mlab_source.dataset.set(points = trail_points)
        ##     self.trail_src.mlab_source.dataset.update()



class ScatterScroller(HasTraits):
    ts_plot = Instance(pm.PagedTimeSeriesPlot)
    ts_page_length = Float(50.0)

    scatter = Instance(MlabSceneModel, ())
    scatter_plot = Instance(ScatterPlot, ())

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
            self, scatter_pts, ts_array, Fs = 1.0,
            trailing = 0.5, scatter_time_scale = 1.0, **traits
            ):
        #self.scatter_array = scatter_array
        self.ts_array = ts_array
        self.Fs = Fs
        self._scrolling = False
        super(ScatterScroller, self).__init__(**traits)
        trail_length = int(
            np.round(trailing * self.Fs / scatter_time_scale)
            )
        self._tf = len(self.ts_array)/self.Fs
        self.ts_plot # trigger default
        self.scatter_plot = ScatterPlot(
            scatter_pts, trail_length = trail_length, Fs=Fs,
            scatter_time_scale = scatter_time_scale
            )
        self.sync_trait('time', self.ts_plot, mutual=True)
        self.sync_trait('time', self.scatter_plot, mutual=True)

    def _ts_plot_default(self):
        figsize = (6, .25)
        n = len(self.ts_array)
        t = np.linspace(self._t0, self._tf, n)
        d_range = (self.ts_array.min(), self.ts_array.max())
        mid = (d_range[1] + d_range[0])/2.0
        extent = (d_range[1] - d_range[0]) * 1.05
        return pm.PagedTimeSeriesPlot(
            t, self.ts_array, figsize=figsize, t0=0,
            page_length=self.ts_page_length,
            ylim=(mid-extent/2, mid+extent/2), linewidth=1
            )

    @on_trait_change('scatter.activated')
    def _plot_scatters(self):
        self.scatter_plot.setup_scatters(self.scatter.mayavi_scene)

    def configure_traits(self, *args, **kwargs):
        ui = super(ScatterScroller, self).configure_traits(*args, **kwargs)
        self._post_canvas_hook()
        return ui

    def edit_traits(self, *args, **kwargs):
        ui = super(ScatterScroller, self).edit_traits(*args, **kwargs)
        self._post_canvas_hook()
        return ui

    def _post_canvas_hook(self):
        #self._connect_mpl_events()
        self.ts_plot.connect_live_interaction()
        self.ts_plot.fig.tight_layout()
        self.ts_plot.draw()

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

    view = View(
        VGroup(
            Item(
                'scatter', editor=SceneEditor(scene_class=Scene),
                height=600, width=600, show_label=False, resizable=True
                ),
            Item(
                'ts_plot', editor=pm.MPLFigureEditor(),
                show_label=False, width=600, height=200, resizable=True
                ),
            HGroup(
                Item('time', label='Time Slice', style='custom'),
                Item('fps', label='FPS'),
                ),
            Item('count', label='Run Clock')
            ),
        resizable=True,
        title='Scatter Scroller'

        )

class ClassCodedScatterScroller(ScatterScroller):
    ts_plot = Instance(pm.PagedClassSegmentedPlot)

    def __init__(
            self, scatter_array, ts_array, labels,
            Fs=1.0, trailing=0.5, **traits
            ):
        self.labels = labels
        super(ClassCodedScatterScroller, self).__init__(
            scatter_array, ts_array, Fs=Fs, trailing=trailing, **traits
            )

    def _ts_plot_default(self):
        figsize = (6, .25)
        n = len(self.ts_array)
        t = np.linspace(self._t0, self._tf, n)
        d_range = (self.ts_array.min(), self.ts_array.max())
        mid = (d_range[1] + d_range[0])/2.0
        extent = (d_range[1] - d_range[0]) * 1.05
        ts_labels = self.labels
        return pm.PagedClassSegmentedPlot(
            t, self.ts_array, ts_labels, figsize=figsize, t0=0,
            page_length=self.ts_page_length,
            ylim=(mid-extent/2, mid+extent/2), linewidth=1
            )

    @on_trait_change('scatter.activated')
    def _plot_scatters(self):
        self.scatter_plot.setup_scatters(
            self.scatter.mayavi_scene, scl_fn=self.labels
            )

if __name__ == '__main__':
    # copied from test_plot3() in Mayavi
    n_mer, n_long = 6, 11
    pi = np.pi
    dphi = pi/1000.0
    phi = np.arange(0.0, 2*pi + 0.5*dphi, dphi)
    mu = phi*n_mer
    x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
    y = np.sin(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
    z = np.sin(n_long*mu/n_mer)*0.5

    ts = x**2 + y**2 + z**2
    sct = np.c_[x, y, z]
    scroller = ScatterScroller(sct, ts)
    scroller.configure_traits()

