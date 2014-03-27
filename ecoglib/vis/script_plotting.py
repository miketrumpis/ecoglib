import os

class ScriptPlotSkip(Exception):
    pass

class ScriptPlotter(object):
    """
    A context manager to handle saving figures in a plot
    """

    def __init__(
            self, path, name, 
            formats=('pdf',), 
            dpi=None,
            plotting=True,
            saving=True
            ):
        path = os.path.abspath(path)
        self.fig_path = os.path.join(path, name)
        self.fig_cache = list()
        self.formats = map(lambda x: x.strip('.'), formats)
        self.dpi = dpi
        self.plotting = plotting
        self.saving = saving

    def savefig(self, f, name):
        """Append a figure reference and a figure name to the
        list of figures to save"""
        for fmt in self.formats:
            name = name.split('.'+fmt)[0]
        self.fig_cache.append( (f, name) )

    def _save_cache(self):
        if not os.path.exists(self.fig_path):
            os.mkdir(self.fig_path)
        for ext in self.formats:
            e_path = os.path.join(self.fig_path, ext)
            if not os.path.exists(e_path):
                os.mkdir(e_path)
            for (f, name) in self.fig_cache:
                dpi = self.dpi or f.dpi
                f_file = os.path.join(e_path, name)+'.'+ext
                print 'saving', f_file
                f.savefig(f_file, dpi=dpi)

    def skip(self, explicit=None):
        # raise an exception that the exit function will skip
        if explicit is None:
            skip_block = not self.plotting
        else:
            skip_block = explicit
        if skip_block:
            raise ScriptPlotSkip
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is ScriptPlotSkip:
            return True
        
        if self.saving:
            self._save_cache()
        

# XXX: would be nice to be able to take an ScripPlotterMaker object and
# set certain flags on and off.. use case:
# spm = ScriptPlotterMaker(*stuff)
# spm.saving = False
# < old script >
# spm.saving = True
# < new script plots >
class ScriptPlotterMaker(object):
    """
    An object that quickly recreates ScriptPlotters with identical
    parameters
    """

    def __init__(self, path, name, **kwargs):
        self._params = (path, name, kwargs)

    def new_plotter(self, **kwargs):
        path, name = self._params[:2]
        inst_kwargs = self._params[2].copy()
        inst_kwargs.update(kwargs)
        return ScriptPlotter(path, name, **inst_kwargs)

        
        
    
