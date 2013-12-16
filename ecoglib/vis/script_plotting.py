import os

class ScriptPlotter(object):
    """
    A context manager to handle saving figures in a plot
    """

    def __init__(
            self, path, name, formats=('pdf',), dpi=None,
            saving=True
            ):
        path = os.path.abspath(path)
        self.fig_path = os.path.join(path, name)
        self.fig_cache = list()
        self.formats = map(lambda x: x.strip('.'), formats)
        self.dpi = dpi
        self.saving = saving

    def savefig(self, f, name):
        """Append a figure reference and a figure name to the
        list of figures to save"""
        name = os.path.splitext(name)[0]
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

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.saving:
            self._save_cache()
        

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

        
        
    
