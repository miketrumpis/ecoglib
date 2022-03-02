from .signal_tools import *
from .signal_plots import *
from ..vis import plotters
# if runing jupyter, avoid importing traitsui stuff (which will launch a qt app)
# TODO: STILL NEED A THREE-WAY DISTINCTION:
#   1) JUPYTER RUNTIME (OBS NO TRAITSUI)
#   2 & 3) TRUE HEADLESS VERSUS AGG BACKEND IN GUI MODE
#  I guess if the GUI app sets MPL to Agg, then it can be responsible for explicitly importing chanenl picker
if not (plotters.headless or plotters.jupyter_runtime):
    from .channel_picker import *