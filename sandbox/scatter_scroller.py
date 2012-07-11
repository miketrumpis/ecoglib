import numpy as np
from threading import Thread
from time import sleep, time

import ecoglib.vis.plot_modules as pm

from traits.api import \
     HasTraits, Range, Float, on_trait_change, Instance, Button
from traitsui.api import \
     RangeEditor, View, HGroup, VGroup, Item

from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk
from mayavi.sources.api import VTKDataSource

class ScatterScroller(HasTraits):
    ts_plot = Instance(pm.StaticTimeSeriesPlot)

    scatter = Instance(MlabSceneModel, ())

    scatter_src = Instance(VTKDataSource)
    scatter_pts = Instance(PipelineBase)

    inst_src = Instance(VTKDataSource)
    inst_pt = Instance(PipelineBase)

    trail_src = Instance(VTKDataSource)
    trail_pts = Instance(PipelineBase)
    trail_line = Instance(PipelineBase)

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

    def __init__(self, scatter_array, ts_array, Fs=1.0, **traits):
        self.scatter_array = scatter_array
        self.ts_array = ts_array
        self.Fs = Fs
        self._scrolling = False
        super(ScatterScroller, self).__init__(**traits)
        self._tf = len(self.ts_array)/self.Fs
        self.ts_plot # trigger default
        self.sync_trait('time', self.ts_plot, mutual=False)
        #self.trails = deque( list(), 200 )

    def _ts_plot_default(self):
        figsize = (6, .25)
        n = len(self.ts_array)
        t = np.linspace(self._t0, self._tf, n)
        d_range = (self.ts_array.min(), self.ts_array.max())
        mid = (d_range[1] + d_range[0])/2.0
        extent = (d_range[1] - d_range[0]) * 1.05
        return pm.StaticTimeSeriesPlot(
            t, self.ts_array, figsize=figsize, t0=0,
            ylim=(mid-extent/2, mid+extent/2), linewidth=1
            )

    @on_trait_change('scatter.activated')
    def _setup_scatters(self):
        fig = self.scatter.mayavi_scene
        s_array = self.scatter_array
        x, y, z = s_array.T
        self.scatter_src = mlab.pipeline.scalar_scatter(
            x, y, z, np.ones_like(x), figure=fig
            )
        self.scatter_pts = mlab.pipeline.glyph(
            self.scatter_src, mode='2dvertex', color=(0,0,1)
            )
        self.scatter_pts.actor.property.opacity = 0.25

        n = round(self.time*self.Fs)
        t_point = s_array[n][:,None]
        x, y, z = t_point
        self.inst_src = mlab.pipeline.scalar_scatter(
            x, y, z, np.array([1]), figure=fig
            )
        # quick and dirty scale
        bb = np.array(self.scatter_src.data.bounds)
        relative_scale = np.power(np.prod(bb[1::2] - bb[0::2]), 1/3.0)
        scale = 2e-2 * relative_scale
        self.inst_pt = mlab.pipeline.glyph(
            self.inst_src, mode='sphere', color=(0,1,0),
            scale_mode='none', scale_factor=scale
            )

        # copy the same for "trail" scatter
        self.trail_src = mlab.pipeline.line_source(
            x, y, z, figure=fig
            )
        self.trail_pts = mlab.pipeline.glyph(
            self.trail_src, mode='sphere', color=(0.7, 0.2, 0.2),
            scale_mode='none', scale_factor=scale*0.4
            )
        #self.trail_pts.actor.property.opacity = 0.4

        # and indeed use the same points for trail tube
        s = mlab.pipeline.stripper(self.trail_pts)
        t = mlab.pipeline.tube(s, tube_radius=scale*0.25)
        self.trail_line = mlab.pipeline.surface(
            t, color=(0.2, 0.7, 0.4), opacity = 0.3
            )


    @on_trait_change('time')
    def _update_time(self):
        n = round(self.time*self.Fs)
        t_point = self.scatter_array[n][None,:]
        self.inst_src.mlab_source.points = t_point
        trail_points = self.scatter_array[max(0,n-200):n+1]
        #self.trail_src.mlab_source.points = trail_points
        self.trail_src.mlab_source.reset(points = trail_points)

    def configure_traits(self, *args, **kwargs):
        super(ScatterScroller, self).configure_traits(*args, **kwargs)
        self._post_canvas_hook()

    def edit_traits(self, *args, **kwargs):
        super(ScatterScroller, self).edit_traits(*args, **kwargs)
        self._post_canvas_hook()

    def _post_canvas_hook(self):
        self._connect_mpl_events()
        self.ts_plot.fig.tight_layout()
        self.ts_plot.draw()

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

    view = View(
        VGroup(
            HGroup(
                Item(
                    'scatter', editor=SceneEditor(scene_class=Scene),
                    height=600, width=600, show_label=False
                    ),
                Item(
                    'ts_plot', editor=pm.MPLFigureEditor(),
                    show_label=False, width=600, height=200, resizable=True
                    )
                ),
            HGroup(
                Item('time', label='Time Slice', style='custom'),
                VGroup(
                    Item('fps', label='FPS'),
                    Item('count', label='Run Clock')
                    )
                )
            ),
        resizable=True,
        title='Scatter Scroller'
        
        )
            
if __name__ == '__main__':
    ## # copied from test_plot3() in Mayavi
    ## n_mer, n_long = 6, 11
    ## pi = np.pi
    ## dphi = pi/1000.0
    ## phi = np.arange(0.0, 2*pi + 0.5*dphi, dphi)
    ## mu = phi*n_mer
    ## x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
    ## y = np.sin(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
    ## z = np.sin(n_long*mu/n_mer)*0.5
    
    ## ts = x**2 + y**2 + z**2
    ## sct = np.c_[x, y, z]
    ## scroller = ScatterScroller(sct, ts)
    ## scroller.configure_traits()

    # try something real
    import scipy.io as sio
    m = sio.loadmat('../../mlab/diffgeo_new_ecog.mat')
    G = m['G'][0,0]
    V = G['EigenVecs']
    del G, m
    ix, iy, iz = (3,4,5)
    sct = np.c_[V[:,ix], V[:,iy], V[:,iz]]

    n_good_pts = 93000

    m = sio.loadmat('../../data/2010-05-19_test_41_filtered.mat')
    d = m.pop('data')
    Fs = float(m['Fs'][0,0])
    nrow = int(m['numRow'][0,0])
    ncol = int(m['numCol'][0,0])
    del m
    pruned_pts = range(0, 17430) + range(22910,93000)
    d = d[:nrow*ncol,pruned_pts] # make contiguous
    
    scroller = ScatterScroller(sct, d.mean(axis=0), Fs=Fs)
    scroller.configure_traits()
