# this module exposes plot types from plot_modules for use outside of
# a TraitsUI GUI.

import numpy as np
import matplotlib.pyplot as pp
import itertools
# XXX: can probably accomplish the stand-alone plot_module plotting
# with a decorator or a factory.


def subspace_scatters(x, labels=None, oneplot=False, **kwargs):
    """
    For samples in X, plot all n-choose-2 scatter plots
    of f_i vs f_j, i != j.

    Parameters
    ----------

    X: ndarray shape (m,n)
      a matrix of n-dimensional features.

    labels: ndarray shape (m,)
      optional labels for each feature

    kwargs: dict
      common options for pyplot.scatter

    """

    m, n = x.shape
    n_plots = n*(n-1)/2

    if oneplot:
        P = int( np.sqrt(float(n_plots)) + 0.5 )
        f = pp.figure()

    for p, ij in enumerate(itertools.combinations(range(n), 2)):
        if oneplot:
            pp.subplot(P,P,p)
        else:
            f = pp.figure()
        i, j = ij
        pp.scatter(x[:,i], x[:,j], c=labels, **kwargs)
        pp.title(r'$f_{%d}$ vs $f_{%d}$'%(j+1,i+1))
    
