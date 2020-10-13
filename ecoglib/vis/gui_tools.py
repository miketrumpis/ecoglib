import os
import os.path as osp
import numpy as np
from traits.api import *
from traitsui.api import *

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, BoxStyle
import matplotlib.cm as cm

from ecogdata.channel_map import CoordinateChannelMap
from ecogdata.util import mkdir_p

import ecoglib.vis.traitsui_bridge as tb
import ecoglib.vis.plot_modules as pm

__all__ = ['SavesFigure', 'ArrayMap', 'EvokedPlot', 'current_screen']


# Get current screen geometry -- hide qt4 import unless really needed!
class Screen(object):

    def _import_and_fetch_screen(self):
        # safe way to get QtGui?
        if not hasattr(self, '_qt4'):
            import traitsui.qt4 as qt4
            self._qt4 = qt4
        Qapp = self._qt4.pyface.qt.QtGui.QApplication.instance()
        return Qapp.desktop().screenGeometry()

    @property
    def x(self):
        V = self._import_and_fetch_screen()
        return V.width()

    @property
    def y(self):
        V = self._import_and_fetch_screen()
        return V.height()


current_screen = Screen()


class SavesFigure(HasTraits):
    sfile = Str
    spath = File(os.getcwd())
    path_button = Button('fig dir')
    update = Button('Refresh')

    fig = Instance(Figure)
    save = Button('Save plot')
    dpi = Enum(400, (100, 200, 300, 400, 500, 600))

    _extensions = ('pdf', 'svg', 'eps', 'png')
    format = Enum('pdf', _extensions)

    # internal axes management
    # has_images = Bool(False)
    has_images = Property
    c_lo = Float
    c_hi = Float
    cmap_name = Str
    # has_graphs = Bool(False)
    has_graphs = Property
    y_lo = Float
    y_hi = Float

    @classmethod
    def live_fig(cls, fig, **traits):
        sfig = cls(fig, **traits)
        v = sfig.default_traits_view()
        v.kind = 'live'
        # sfig.edit_traits(view=v)
        sfig.configure_traits(view=v)
        return sfig

    def __init__(self, fig, **traits):
        super(SavesFigure, self).__init__(**traits)
        self.fig = fig
        if self.has_images:
            self._search_image_props()
            ## clim = ()
            ## for ax in self.fig.axes:
            ##     if ax.images:
            ##         clim = ax.images[0].get_clim()
            ##         cm = ax.images[0].get_cmap().name
            ##         break
            ## if clim:
            ##     self.trait_setq(c_lo = clim[0], c_hi = clim[1], cmap_name=cm)

        self.on_trait_change(self._clip, 'c_hi')
        self.on_trait_change(self._clip, 'c_lo')
        self.on_trait_change(self._cmap, 'cmap_name')
        if self.has_graphs:
            self._search_graph_props()
            ## ylim = ()
            ## for ax in self.fig.axes:
            ##     # is this specific enough?
            ##     if not ax.images and ax.lines:
            ##         ylim = ax.get_ylim()
            ##         break
            ## if ylim:
            ##     self.trait_setq(y_lo = ylim[0], y_hi = ylim[1])
        self.on_trait_change(self._ylims, 'y_hi')
        self.on_trait_change(self._ylims, 'y_lo')

    def _search_image_props(self):
        if not self.has_images:
            return
        clim = None
        for ax in self.fig.axes:
            if ax.images:
                clim = ax.images[0].get_clim()
                cm = ax.images[0].get_cmap().name
                break
        if clim:
            self.trait_set(c_lo=clim[0], c_hi=clim[1], cmap_name=cm)

    def _search_graph_props(self):
        ylim = ()
        for ax in self.fig.axes:
            # is this specific enough?
            if not ax.images and ax.lines:
                ylim = ax.get_ylim()
                break
            if not ax.images and ax.collections:
                ylim = ax.get_ylim()
                break
        if ylim:
            self.trait_set(y_lo=ylim[0], y_hi=ylim[1])

    def _get_has_images(self):
        if not hasattr(self.fig, 'axes'):
            return False
        for ax in self.fig.axes:
            if hasattr(ax, 'images') and len(ax.images) > 0:
                return True
        return False

    def _get_has_graphs(self):
        if not hasattr(self.fig, 'axes'):
            return False
        for ax in self.fig.axes:
            if hasattr(ax, 'lines') and len(ax.lines) > 0:
                return True
            if hasattr(ax, 'collections') and len(ax.collections) > 0:
                return True
        return False

    def _ylims(self):
        for ax in self.fig.axes:
            if not ax.images and ax.lines:
                ax.set_ylim(self.y_lo, self.y_hi)
        self.fig.canvas.draw()

    def _clip(self):
        for ax in self.fig.axes:
            for im in ax.images:
                im.set_clim(self.c_lo, self.c_hi)
        self.fig.canvas.draw()

    def _cmap(self):
        name = self.cmap_name
        try:
            colors = cm.cmap_d[name]
        except KeyError:
            # try to evaluate the string as a function in colormaps module
            try:
                code = 'cmaps.' + name
                colors = eval(code)
            except:
                # try the same within seaborn
                try:
                    code = 'cmaps.sns.' + name
                    colors = eval(code)
                except:
                    return
        for ax in self.fig.axes:
            for im in ax.images:
                im.set_cmap(colors)
        self.fig.canvas.draw()

    def _save_fired(self):
        if not (self.spath or self.sfile):
            return
        if self.spath[-1] != '/':
            self.spath = self.spath + '/'
        pth = osp.dirname(self.spath)
        mkdir_p(pth)

        f, e = osp.splitext(self.sfile)
        if e in self._extensions:
            self.sfile = f

        ext_sfile = self.sfile + '.' + self.format
        self.fig.savefig(osp.join(self.spath, ext_sfile), dpi=self.dpi)

    def _update_fired(self):
        self._search_image_props()
        self._search_graph_props()

    def default_traits_view(self):
        # The figure is put in a panel with correct fig-width and fig-height.
        # Using negative numbers locks in the size. It appears that using
        # positive numbers enforces a minimum size.
        fig = self.fig
        fh = int(fig.get_figheight() * fig.get_dpi())
        fw = int(fig.get_figwidth() * fig.get_dpi())
        traits_view = View(
            VSplit(
                UItem(
                    'fig', editor=CustomEditor(tb.embedded_figure),
                    resizable=True, height=fh, width=fw
                ),
                Group(
                    HGroup(
                        Item('spath', label='Figure path'),
                        # UItem('path_button'),
                        Item('sfile', style='simple', label='Image File')
                    ),
                    HGroup(
                        HGroup(
                            Item('dpi', label='DPI'),
                            Item('format'),
                            UItem('save'),
                            label='Image format'
                        ),
                        HGroup(
                            UItem('update'),
                            label='Refresh properties'
                        )
                    ),

                    HGroup(
                        VGroup(
                            HGroup(
                                Item('c_lo', label='Clip lo',
                                     enabled_when='has_images', width=4),
                                Item('c_hi', label='Clip hi',
                                     enabled_when='has_images', width=4),
                                enabled_when='has_images'
                            ),
                            Item('cmap_name', label='Colormap',
                                 enabled_when='has_images'),
                            enabled_when='has_images',
                            label='Control image properties'
                        ),
                        VGroup(
                            Item('y_lo', label='y-ax lo', width=4),
                            Item('y_hi', label='y-ax hi', width=4),
                            enabled_when='has_graphs',
                            label='Control axis properties'
                        )
                    )
                )
            ),
            resizable=True
        )
        return traits_view


class ArrayMap(HasTraits):
    """
    A simple wrapper of an MPL figure. Has a .fig and works
    with MPLFigureEditor from ecoglib.vis.traitsui_bridge.

    The figure itself is a sensor vector embedded in the geometry
    of the electrode array. Sites can be clicked and selected to
    modify the "selected_site" Trait
    """
    selected_site = Int(-1)

    def __init__(self, chan_map, labels=None, vec=None, ax=None, map_units=None, cbar=True,
                 mark_site=True, **plot_kwargs):
        # the simplest instantiation is with a vector to plot
        self.labels = labels
        self._clim = plot_kwargs.pop('clim', None)
        chan_image = chan_map.image(vec, cbar=cbar, ax=ax, clim=self._clim, **plot_kwargs)
        if cbar:
            self.fig, self.cbar = chan_image
        else:
            self.fig = chan_image
            self.cbar = None
        self.ax = self.fig.axes[0]
        self.ax.axis('image')
        if self.cbar:
            if labels is not None:
                self.cbar.set_ticks(np.arange(0, len(labels)))
                self.cbar.set_ticklabels(labels)
            if map_units is not None:
                self.cbar.set_label(map_units)

        super(ArrayMap, self).__init__()

        self._coord_map = isinstance(chan_map, CoordinateChannelMap)
        # both these kinds of maps update in the same way :D
        if self._coord_map:
            self._map = self.ax.collections[-1]
        else:
            self._map = self.ax.images[-1]
        self.chan_map = chan_map
        self._box = None
        self._mark_site = mark_site

        # if (ax is None):
        #     if vec is None:
        #         vec = np.ones(len(chan_map))
        #     cmap = traits_n_kws.pop('cmap', cm.Blues)
        #     origin = traits_n_kws.pop('origin', 'upper')
        #     fsize = np.array(chan_map.geometry[::-1], 'd') / 3.0
        #     self.fig = Figure(figsize=tuple(fsize))
        #     self.ax = self.fig.add_subplot(111)
        #     self._map = self.ax.imshow(
        #         chan_map.embed(vec), cmap=cmap, origin=origin,
        #         clim=self._clim
        #     )
        #     self.ax.axis('image')
        #     self.cbar = self.fig.colorbar(
        #         self._map, ax=self.ax, use_gridspec=True
        #     )
        #     if labels is not None:
        #         self.cbar.set_ticks(np.arange(0, len(labels)))
        #         self.cbar.set_ticklabels(labels)
        #     elif map_units is not None:
        #         self.cbar.set_label(map_units)
        # elif ax:
        #     self.ax = ax
        #     self.fig = ax.figure


    def click_listen(self, ev):
        try:
            i, j = ev.ydata, ev.xdata
            if i is None or j is None:
                raise TypeError
            if not self._coord_map:
                i, j = list(map(round, (i, j)))
        except TypeError:
            if ev.inaxes is None:
                self.selected_site = -1
            return
        try:
            self.selected_site = self.chan_map.lookup(i, j)
        except ValueError:
            self.selected_site = -1

    @on_trait_change('selected_site')
    def _move_box(self):
        if not self._mark_site:
            return
        try:
            # negative index codes for outside array
            if self.selected_site < 0:
                raise IndexError
            i, j = self.chan_map.rlookup(self.selected_site)
            if self._box:
                self._box.remove()
            box_size = self.chan_map.min_pitch if self._coord_map else 1
            style = BoxStyle('Round', pad=0.3 * box_size, rounding_size=None)
            self._box = FancyBboxPatch(
                (j - box_size / 2.0, i - box_size / 2.0), box_size, box_size, boxstyle=style,
                fc='none', ec='k', transform=self.ax.transData,
                clip_on=False
            )
            self.ax.add_patch(self._box)
        except IndexError:
            if self._box:
                self._box.remove()
                self._box = None
            pass
        finally:
            self.fig.canvas.draw()

    def update_map(self, scores, c_label=None, **extra):
        "Update map image given new set of scalars from the sensor vector"
        if 'clim' not in extra:
            extra['clim'] = self._clim
        if not self._coord_map:
            if scores.shape != self.chan_map.geometry:
                scores = self.chan_map.embed(scores)
        elif len(scores) != len(self.chan_map):
            raise ValueError("Can't plot vector length {} to a coordinate map.".format(len(scores)))
        self._map.set_array(scores)
        self._map.update(extra)
        if self.cbar:
            if c_label is not None:
                self.cbar.set_label(c_label)
        try:
            if self.cbar:
                self.cbar.draw_all()
            self.fig.canvas.draw()
        except:
            # no canvas? no problem
            pass


class EvokedPlot(pm.StaticFunctionPlot):
    """
    This is an instance of a StaticFunctionPlot of plot_modules, meaning
    that it has many Traits-ified attributes (e.g. xlim and ylim have
    listeners to automatically redraw the view). It plots multiple
    timeseries and has GUI interaction.

    In this case, the GUI interaction defines an "interval of interest"
    in the timeseries. Analyses of evoked potentials can take advantage
    of this user-defined interval.
    """

    peak_min = Float(0.005)
    peak_max = Float(0.025)

    def create_fn_image(self, x, t=None, **plot_line_props):
        # operates in the following modes:
        # * plots all one color (1D or 2D arrays)
        #  - color argument is single item
        # * plot in a gradient of colors (for 2D arrays)
        #  - color argument is an array or a colormap
        # * plot groups of timeseries by color (for 3D arrays)
        #  - color argument should be an array or a colormap
        #  - timeseries are grouped in the last axis
        if t is None:
            t = np.arange(len(x))

        if x.ndim == 1:
            x = np.reshape(x, (len(x), 1))

        ndim = x.ndim
        color = plot_line_props.pop('color', 'k')
        if isinstance(color, cm.colors.Colormap):
            plot_colors = color(np.linspace(0, 1, x.shape[-1]))
        elif isinstance(color, (np.ndarray, list, tuple)):
            plot_colors = color
        elif isinstance(color, str):
            plot_colors = (color,) * x.shape[-1]
        lines = list()
        if ndim == 2:
            self.ax.set_prop_cycle(color=list(plot_colors))
            lines = self.ax.plot(t, x, **plot_line_props)
        if ndim == 3:
            for group, color in zip(x.transpose(2, 0, 1), plot_colors):
                ln = self.ax.plot(t, group, color=color, **plot_line_props)
                lines.extend(ln)

        self.ax.set_xlabel('time from stimulus')
        pos = self.ax.get_position()
        self.ax.set_position([pos.x0, 0.15, pos.x1 - pos.x0, pos.y1 - .15])
        return lines

    def move_bar(self, *time):
        pass

    def connect_live_interaction(self):
        connections = pm.AxesScrubber.gen_event_associations(
            self, 'peak_min', 'peak_max',
            scrub_x=True, sense_button=3
        )
        super(EvokedPlot, self).connect_live_interaction(
            extra_connections=connections
        )

