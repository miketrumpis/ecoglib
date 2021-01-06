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
            import warnings
            import seaborn as sns
            # Fix until MPL or seaborn gets straightened out
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', self._mpl.cbook.MatplotlibDeprecationWarning)
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
        return 'ipykernel.pylab.backend_inline' in self.mpl.get_backend()


plotters = _plot_libraries()
