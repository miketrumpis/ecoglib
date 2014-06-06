import os
import inspect

def rst_image_markup(relpath, name, extensions):

    lines = ['\n', 
             '.. image:: %s/%s/%s.*\n'%(relpath, 'png', name),
             '   :width: 500\n'
             '\n'
             ]
    extensions = list(extensions)
    if extensions:
        download = 'Download hi-res: '
        for ext in extensions:
            download = download + ':download:`%s <%s/%s/%s.%s>`, '%(
                ext.upper(), relpath, ext, name, ext
                )
        download = download[:-2] + '\n'
    lines.append(download)
    return ''.join(lines)

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
            saving=True,
            www=False
            ):
        path = os.path.abspath(path)
        self.fig_path = os.path.join(path, name)
        self.fig_cache = list()
        self.fig_text = list()
        self.formats = map(lambda x: x.strip('.'), formats)
        self.dpi = dpi
        self.plotting = plotting
        self.saving = saving
        self.www = www
        if www and 'png' not in self.formats:
            self.formats = self.formats + ('png',)

    def savefig(self, f, name, rst_text=''):
        """Append a figure reference and a figure name to the
        list of figures to save"""
        for fmt in self.formats:
            name = name.split('.'+fmt)[0]
        self.fig_cache.append( (f, name) )
        self.fig_text.append( rst_text )

    def _save_cache(self):
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
        for ext in self.formats:
            e_path = os.path.join(self.fig_path, ext)
            if not os.path.exists(e_path):
                os.mkdir(e_path)
            for (f, name) in self.fig_cache:
                dpi = self.dpi or f.dpi
                f_file = os.path.join(e_path, name)+'.'+ext
                print 'saving', f_file
                f.savefig(f_file, dpi=dpi)

    def _fixup_rst(self, fname, line, context):
        rst_source = open(fname).read()
        cstart = rst_source.find(context)
        if cstart < 0:
            raise RuntimeError('context not found in this file')


        prev = rst_source[:cstart+len(context)]
        cstart += len(context)
        post = rst_source[cstart:]

        magic_str = '\n\n.. post-hoc images\n'

        if post.find(magic_str)==0:
            print 'already fixed'
            return
        
        new_str = [prev, magic_str]

        pth, _ = os.path.split(fname)
        relpath = os.path.relpath(self.fig_path, pth)
        for (f, name), text in zip(self.fig_cache, self.fig_text):
            image_lines = rst_image_markup(relpath, name, self.formats)
            if text:
                new_str.extend(['\n', text+'\n'])
            new_str.append(image_lines)
        new_str.append(post)
            
        open(fname, 'w').write(''.join(new_str))
                   
        
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
        calling = inspect.getouterframes(inspect.currentframe())[1]
        fi = inspect.getframeinfo(calling[0], 5)
        line = fi.lineno
        context = '    '.join(fi.code_context)
        fname = fi.filename
        #print inspect.getsource(inspect.currentframe())
        if self.saving:
            self._save_cache()
            rst_file = os.path.splitext(fname)[0] + '.rst'
            if self.www and os.path.exists(rst_file):
                self._fixup_rst(rst_file, line, context)
            
        

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

        
        
    
