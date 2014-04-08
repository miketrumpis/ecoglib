import numpy as np
from traits.api import Tuple

from data_scroll import *
from scatter_scroller import *

# XXX: these are some horrible hacks.. need to think of away around
# Traits rigidity
class dummy_plot1(pm.ScrollingTimeSeriesPlot):
    def set_window(*args, **kwargs):
        pass
class dummy_plot2(pm.ScrollingClassSegmentedPlot):
    def set_window(*args, **kwargs):
        pass

class ComboScroller(DataScroller):

    # This will essentially be a datascroller, but with the scatter plot
    # scene included. We will skip the zoom plot and not create a figure
    # for it.

    scatter = Instance(MlabSceneModel, ())
    scatter_plot = Instance(ScatterPlot, ())

    def __init__(
            self, d_array, scatter_pts, ts_array,
            rowcol = (), Fs = 1.0,
            trailing = 0.5, scatter_time_scale = 1.0,
            **traits
            ):
        super(ComboScroller, self).__init__(
            d_array, ts_array, rowcol=rowcol, Fs=Fs, **traits
            )
        self.zoom_plot = dummy_plot1(np.array([0.0]))
        trail_length = int(
            np.round(trailing * self.Fs / scatter_time_scale)
            )
        self.scatter_plot = ScatterPlot(
            scatter_pts, trail_length=trail_length, Fs=Fs,
            scatter_time_scale = scatter_time_scale
            )
        self.sync_trait('time', self.scatter_plot, mutual=True)

    def _post_canvas_hook(self):
        self.ts_plot.connect_live_interaction()
        self.ts_plot.fig.tight_layout()
        self.ts_plot.draw()

    @on_trait_change('scatter.activated')
    def _plot_scatters(self):
        self.scatter_plot.setup_scatters(self.scatter.mayavi_scene)

    view = View(
        VGroup(
            HGroup(
                Item(
                    'scatter', editor=SceneEditor(scene_class=Scene),
                    height=300, width=300, show_label=False, resizable=True
                    ),
                Item(
                    'array_scene', editor=SceneEditor(scene_class=Scene),
                    height=300, width=300, show_label=False, resizable=True
                    )
                ),
            Item(
                'ts_plot', editor=tb.MPLFigureEditor(),
                show_label=False, width=600, height=200, resizable=True
                ),
            HGroup(
                Item('time', label='Time Slice', style='custom'),
                Item('eps', label='Amplitude Limits')
                ),
            HGroup(
                Item('count', label='Run Clock'),
                Item('fps', label='FPS'),
                Item('arr_eps', label='Array Color Limits')
                )
            ),
        resizable=True,
        title='Combo Scroller'

        )


class ClassCodedComboScroller(ClassCodedDataScroller):

    scatter = Instance(MlabSceneModel, ())
    scatter_plot = Instance(ScatterPlot, ())

    def __init__(
            self, d_array, scatter_pts, ts_array, labels,
            rowcol = (), Fs = 1.0,
            trailing = 0.5, scatter_time_scale = 1.0,
            **traits
            ):
        super(ClassCodedComboScroller, self).__init__(
            d_array, ts_array, labels, rowcol=rowcol, Fs=Fs, **traits
            )
        self.zoom_plot = dummy_plot2(np.array([0.0]), np.array([0]), 1)
        trail_length = int(
            np.round(trailing * self.Fs / scatter_time_scale)
            )
        self.scatter_plot = ScatterPlot(
            scatter_pts, trail_length=trail_length, Fs=Fs,
            scatter_time_scale = scatter_time_scale
            )
        self.sync_trait('time', self.scatter_plot, mutual=True)

    def _post_canvas_hook(self):
        self.ts_plot.connect_live_interaction()
        self.ts_plot.fig.tight_layout()
        self.ts_plot.draw()

    @on_trait_change('scatter.activated')
    def _plot_scatters(self):
        self.scatter_plot.setup_scatters(
            self.scatter.mayavi_scene, scl_fn=self.labels
            )

    view = View(
        VGroup(
            HGroup(
                Item(
                    'scatter', editor=SceneEditor(scene_class=Scene),
                    height=300, width=300, show_label=False, resizable=True
                    ),
                Item(
                    'array_scene', editor=SceneEditor(scene_class=Scene),
                    height=300, width=300, show_label=False, resizable=True
                    )
                ),
            Item(
                'ts_plot', editor=tb.MPLFigureEditor(),
                show_label=False, width=600, height=200, resizable=True
                ),
            HGroup(
                Item('time', label='Time Slice', style='custom'),
                Item('eps', label='Amplitude Limits')
                ),
            HGroup(
                Item('count', label='Run Clock'),
                Item('fps', label='FPS'),
                Item('arr_eps', label='Array Color Limits')
                )
            ),
        resizable=True,
        title='Class Coded Combo Scroller'

        )
