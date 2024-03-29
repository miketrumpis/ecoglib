"""Visualization tools. Keep flat for now, but think towards organizing into subpackages for cohesive toolboxes.
"""


class _plot_libraries:

    _mpl = None
    _sns = None
    _plt = None

    def _onetime_load_modules(self):
        if self._mpl is None:
            import matplotlib as mpl
            self._mpl = mpl
        if self._plt is None:
            import matplotlib.pyplot as plt
            self._plt = plt
        if self._sns is None:
            import seaborn as sns
            sns.reset_orig()
            self._sns = sns

    @property
    def mpl(self):
        if self._mpl is None:
            self._onetime_load_modules()
        return self._mpl

    @property
    def plt(self):
        if self._plt is None:
            self._onetime_load_modules()
        return self._plt

    @property
    def sns(self):
        if self._sns is None:
            self._onetime_load_modules()
        return self._sns

    @property
    def jupyter_runtime(self):
        be = self.mpl.get_backend()
        # This likely can be simplified to "backend_inline" in mpl.get_backend()
        if be.lower() in ('ipympl', 'nbagg'):
            return True
        if 'ipykernel.pylab.backend_inline' in be or 'matplotlib_inline.backend_inline' in be:
            return True
        return False

    # Assume that if the agg backend is default, then this is a headless box
    @property
    def headless(self):
        return self.mpl.get_backend() == 'agg'


plotters = _plot_libraries()
