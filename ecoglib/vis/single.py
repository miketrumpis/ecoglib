# this module exposes plot types from plot_modules for use outside of
# a TraitsUI GUI.

import numpy as np
import matplotlib.pyplot as pp
import itertools
# XXX: can probably accomplish the stand-alone plot_module plotting
# with a decorator or a factory.

def plot_lin_color(x, cmap='jet', tx=None):
    cmap = pp.cm.cmap_d.get(cmap, pp.cm.jet)
    npt, nline = x.shape
    colors = cmap( np.linspace(0, 1, nline) )
    f = pp.figure()
    ax = f.add_subplot(111)
    if tx is None:
        tx = np.arange(npt)
    for trace, c in zip(x.T, colors):
        ax.plot(tx, trace, color=c)
    return f

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
    if labels is None:
        c = 'b'
    else:
        c = labels
    if oneplot:
        P1 = int( np.ceil(np.sqrt(float(n_plots))) )
        P2 = int( np.ceil(n_plots / float(P1)) )
        f = pp.figure()
    else:
        plots = list()
    
    for p, ij in enumerate(itertools.combinations(range(n), 2)):
        if oneplot:
            pp.subplot(P2,P1,p+1)
        else:
            plots.append(pp.figure())
        i, j = ij
        pp.scatter(x[:,i], x[:,j], c=c, **kwargs)
        #pp.title(r'$f_{%d}$ vs $f_{%d}$'%(j+1,i+1))
        pp.gca().set_xlabel(r'$f_{%d}$'%(i+1,))
        pp.gca().set_ylabel(r'$f_{%d}$'%(j+1,))
        pp.gca().yaxis.set_ticks([])
        pp.gca().xaxis.set_ticks([])
        #if oneplot:
        #pp.gca().xaxis.set_visible(False)
        #pp.gca().yaxis.set_visible(False)
    if oneplot:
        f.tight_layout()
        return f
    return plots
