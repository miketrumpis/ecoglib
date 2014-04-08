"""This module creates a bridge between the plot modules and a traits GUI
"""

# Matplotlib
import matplotlib
# We want matplotlib to use a QT backend
try:
    matplotlib.use('QtAgg')
except ValueError:
    matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import \
     FigureCanvasQTAgg as FigureCanvas

from traitsui.qt4.editor import Editor
from traitsui.basic_editor_factory import BasicEditorFactory


##############################################################################
########## Matplotlib to Traits Panel Integration ############################

class _MPLFigureEditor(Editor):
    """
    This class provides a QT canvas to all MPL figures when drawn
    under the TraitsUI framework.
    """

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
