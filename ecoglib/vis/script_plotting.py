import os
import inspect
from matplotlib.backends.backend_pdf import PdfPages


def rst_image_markup(relpath, name, extensions):

    lines = ['\n',
             '.. image:: %s/%s/%s.*\n' % (relpath, 'png', name),
             '   :width: 500\n'
             '\n'
             ]
    extensions = list(extensions)
    if extensions:
        download = 'Download hi-res: '
        for ext in extensions:
            download = download + ':download:`%s <%s/%s/%s.%s>`, ' % (
                ext.upper(), relpath, ext, name, ext
            )
        download = download[:-2] + '\n'
    lines.append(download)
    return ''.join(lines)


def make_heading(txt, level='-'):
    return txt + '\n' + level * len(txt) + '\n\n'

# knicked from SO with modifications
# http://stackoverflow.com/questions/17870544/python-find-starting-and-ending-indices-of-sublist-in-list


def find_sub_source(sl, l):
    # Find the context of source lines in list sl
    # with respect to full source lines in l (ignoring whitespace).
    # Returned is the slicing-index of the sublist (i.e. index set
    # is exclusive at the upper limit).
    #
    results = []
    sll = len(sl)
    sl = [s.strip() for s in sl]
    for ind in (i for i, e in enumerate(l) if e.strip() == sl[0]):
        l_strip = [s.strip() for s in l[ind:ind + sll]]
        if l_strip == sl:
            # results.append((ind,ind+sll-1))
            return ind, ind + sll
    return ()


class ScriptPlotSkip(Exception):
    pass


class ScriptPlotter(object):
    """
    A context manager to handle saving figures.
    """

    def __init__(self, path, name, formats=('pdf',), dpi=None, plotting=True, saving=True):
        """
        Construct a plot saver

        Parameters
        ----------
        path: str
            Base path for figures
        name: str
            Subdirectory to save images in this context
        formats: sequence
            Save figures in all of these MPL-supported formats (will save under format-specific subdirectories)
        dpi: int
            If not None, then make bitmaps at this DPI
        plotting: bool
            If False, then skip code execution in this context
        saving: bool
            If False, skip saving plots

        """
        path = os.path.abspath(path)
        self.fig_path = os.path.join(path, name)
        self.fig_cache = list()
        self.pages_cache = list()
        self.fig_text = list()
        self.formats = [x.strip('.') for x in formats]
        self.dpi = dpi
        self.plotting = plotting
        self.saving = saving

    def savefig(self, f, name, rst_text='', **fig_kwargs):
        """Append a figure reference and a figure name to the
        list of figures to save"""
        for fmt in self.formats:
            name = name.split('.' + fmt)[0]
        self.fig_cache.append((f, name, fig_kwargs))
        self.fig_text.append(rst_text)

    def save_many(self, figs, name):
        """Save a PdfPages with the list of figures"""
        self.pages_cache.append((figs, name))

    def _save_cache(self):
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
        for ext in self.formats:
            e_path = os.path.join(self.fig_path, ext)
            if not os.path.exists(e_path):
                os.mkdir(e_path)
            for (f, name, kwargs) in self.fig_cache:
                dpi = self.dpi or f.dpi
                f_file = os.path.join(e_path, name) + '.' + ext
                print('saving', f_file)
                f.savefig(f_file, dpi=dpi, **kwargs)
        if len(self.pages_cache):
            e_path = os.path.join(self.fig_path, 'pdf')
            for pages in self.pages_cache:
                figs, name = pages
                with PdfPages(os.path.join(e_path, name) + '.pdf') as pdf:
                    for f in figs:
                        pdf.savefig(f)

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


class RstScriptPlotter(ScriptPlotter):
    """This script plotter provides some magical weaving of images into old ex2rst generated reStructuredText,
    to allow downloading the plots from an HTML server."""

    def __init__(self, path, name, www=False, heading='', hlevel='-', **kwargs):
        super(RstScriptPlotter, self).__init__(path, name, **kwargs)
        self.www = www
        self.heading = heading
        self.hlevel = hlevel
        self._heading_pending = len(heading) > 0
        if www and 'png' not in self.formats:
            self.formats.append('png')

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is ScriptPlotSkip:
            return True
        calling = inspect.getouterframes(inspect.currentframe())[1]
        fi = inspect.getframeinfo(calling[0], 5)
        line = fi.lineno
        code_context = [s for s in fi.code_context if s.strip() not in ('"""', "'''")]
        fname = fi.filename
        if self.saving:
            self._save_cache()
            rst_file = os.path.splitext(fname)[0] + '.rst'
            if self.www and os.path.exists(rst_file):
                self._fixup_rst(rst_file, line, code_context)

    def _fixup_rst(self, fname, line, context):
        rst_source = open(fname).readlines()
        # xxx: line gives trouble because the embedded py-source
        # is not guaranteed to be in the rst-source after the
        # given line (due to skipped header material).
        #
        # So let's assume the context only happens uniquely
        # within the rst file?
        line = 0
        sub_idx = find_sub_source(context, rst_source[line:])
        if not sub_idx:
            raise RuntimeError('context not found in this file')

        # this is the cutoff where we're going to
        # attempt to splice in code
        cstart = line + sub_idx[1]

        prev_lines = rst_source[:cstart]

        # We may be in the middle of a literal block or a hidden-code
        # block, so scan ahead until we're out of the block. Since
        # multiple script-plotter managers could be within this block,
        # we'll have to check for the end of the sequence of fixed-up
        # ReST.

        fig_names = [n[1] for n in self.fig_cache]
        fig_hash = '.. %s\n' % str(tuple(fig_names))

        magic_str = '\n.. post-hoc images\n'
        magic_end = '.. post-hoc images finished\n\n'

        new_str = prev_lines[:]

        inside_fixup = False
        inside_markup = False
        save_line = ''
        # for line in post_lines:
        for line in rst_source[cstart:]:
            if not inside_markup:
                if not line.startswith('   '):
                    inside_markup = True
                new_str.append(line)
                cstart += 1
                continue

            # now we're into markup
            if line.strip() == magic_str.strip():
                save_line = line
                inside_fixup = True
                continue
            if inside_fixup:
                # we hit a magic string, check if it's ours
                if line.strip() == fig_hash.strip():
                    print('already fixed')
                    return
                # if not, put back that last line (if not already done)
                if save_line:
                    new_str.append(save_line)
                    cstart += 1
                    save_line = ''
                new_str.append(line)
                cstart += 1
                if line.strip() == magic_end.strip():
                    inside_fixup = False
            else:
                if line.strip():
                    # if we've gotten here, then these conditions are true
                    # * not inside block-quotes or code-block
                    # * the previous line was not a magic start str
                    # * the next non-empty line is not a magic start str
                    break
                else:
                    # play out empties and see what comes along
                    new_str.append(line)
                    cstart += 1
        post = rst_source[cstart:]

        # if a new heading is to be written, see to it here
        if self.heading and not self._heading_pending:
            new_str.append(make_heading(self.heading, self.hlevel))
            self._heading_pending = False
        # now proceed to add figure directive and title
        new_str.extend([magic_str, fig_hash])

        pth, _ = os.path.split(fname)
        relpath = os.path.relpath(self.fig_path, pth)
        for (f, name, _), text in zip(self.fig_cache, self.fig_text):
            image_lines = rst_image_markup(relpath, name, self.formats)
            if text:
                new_str.extend(['\n', text + '\n'])
            new_str.append(image_lines)
        new_str.extend(['\n', fig_hash, magic_end])
        new_str.extend(post)

        open(fname, 'w').write(''.join(new_str))


class ScriptPlotterMaker(object):
    """
    An object that quickly recreates ScriptPlotters with identical
    parameters
    """

    def __init__(self, path, name, rst=False, **kwargs):
        self._params = (path, name, kwargs)
        self._use_rst = rst

    def new_plotter(self, **kwargs):
        path, name = self._params[:2]
        inst_kwargs = self._params[2].copy()
        inst_kwargs.update(kwargs)
        if self._use_rst:
            return RstScriptPlotter(path, name, **inst_kwargs)
        else:
            return ScriptPlotter(path, name, **inst_kwargs)
