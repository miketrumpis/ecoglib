"""This module creates a bridge between the plot modules and a traits GUI
"""

import os
import matplotlib
use = matplotlib.get_backend()

# Only really tested for QT4Agg backend
if use.lower() == u'qt4agg':
    from matplotlib.backends.backend_qt4agg import \
     FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import \
     NavigationToolbar2QT as NavigationToolbar
elif use.lower() == u'qt5agg':
    from matplotlib.backends.backend_qt5agg import \
     FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import \
     NavigationToolbar2QT as NavigationToolbar
elif use.lower() == 'wxagg':
    from matplotlib.backends.backend_wxagg \
      import FigureCanvasWxAgg as FigureCanvas
    from matplotlib.backends.backend_wx import \
     NavigationToolbar2Wx as NavigationToolbar
else: #elif use.lower() == 'agg':
    # make this the fallback case
    from matplotlib.backends.backend_agg \
      import FigureCanvasAgg as FigureCanvas
    from matplotlib.backend_bases \
      import NavigationToolbar2 as NavigationToolbar
         
from matplotlib.figure import Figure

try:
    from traitsui.qt4.editor import Editor
except RuntimeError:
    class Editor(object):
        pass

from traitsui.basic_editor_factory import BasicEditorFactory
from traitsui.api import Handler


##############################################################################
########## Matplotlib to Traits Panel Integration ############################

class MiniNavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save', 'Subplots')]

def assign_canvas(editor):
    
    if isinstance(editor.object, Figure):
        mpl_fig = editor.object
    else:
        mpl_fig = editor.object.fig
    if hasattr(mpl_fig, 'canvas') and mpl_fig.canvas is not None:
        # strip this canvas, and close the originating figure?
        #num = mpl_fig.number
        #Gcf.destroy(num)
        return mpl_fig.canvas
    mpl_canvas = FigureCanvas(mpl_fig)
    return mpl_canvas

def _embedded_qt_figure(parent, editor, toolbar=True):
    if os.environ['QT_API'].lower() == 'pyqt5':
        from PyQt5.QtWidgets import QVBoxLayout, QWidget
    elif os.environ['QT_API'].lower() in ('pyside', 'pyqt'):
        try:
            from PySide.QtGui import QVBoxLayout, QWidget
        except ImportError:
            from PyQt4.QtGui import QVBoxLayout, QWidget

    panel = QWidget(parent.parentWidget())
    canvas = assign_canvas(editor)

    vbox = QVBoxLayout(panel)
    vbox.addWidget(canvas)
    if toolbar:
        toolbar = MiniNavigationToolbar(canvas, panel)
        vbox.addWidget(toolbar)
    panel.setLayout(vbox)
    return panel

def _embedded_wx_figure(parent, editor, toolbar=True):
    """
    Builds the Canvas window for displaying the mpl-figure
    from http://wiki.scipy.org/Cookbook/EmbeddingInTraitsGUI
    """
    import wx
    fig = editor.object.figure
    panel = wx.Panel(parent, -1)
    canvas = assign_canvas(editor)
    #toolbar.Realize()

    sizer = wx.BoxSizer(wx.VERTICAL)
    if toolbar:
        toolbar = MiniNavigationToolbar(canvas)
        sizer.Add(canvas,1,wx.EXPAND|wx.ALL,1)
        sizer.Add(toolbar,0,wx.EXPAND|wx.ALL,1)
    else:
        sizer.Add(canvas,0,wx.EXPAND|wx.ALL,1)
    panel.SetSizer(sizer)
    return panel

if use.lower() in ('qt4agg', 'qt5agg'):
    embedded_figure = _embedded_qt_figure  
else:
    embedded_figure = _embedded_wx_figure

class _MPLFigureEditor(Editor):
    """
    This class provides a QT canvas to all MPL figures when drawn
    under the TraitsUI framework.
    """

    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # matplotlib commands to create a canvas
        if isinstance(self.value, Figure):
            mpl_fig = self.value
        else:
            mpl_fig = self.value.fig
        if hasattr(mpl_fig, 'canvas') and mpl_fig.canvas is not None:
            return mpl_fig.canvas
        mpl_canvas = FigureCanvas(mpl_fig)
        # for Qt?
        # mpl_canvas.setParent(parent)
        return mpl_canvas

class MPLFigureEditor(BasicEditorFactory):

   klass = _MPLFigureEditor


class PingPongStartup(Handler):
    """
    This object can act as a View Handler for an HasTraits instance 
    that creates Matplotlib elements after the GUI canvas is created.
    This handler simply calls the _post_canvas_hook() method on the
    HasTraits instance, which applies any finishing touches to the MPL
    elements.
    """

    def init(self, info):
        info.object._post_canvas_hook()
