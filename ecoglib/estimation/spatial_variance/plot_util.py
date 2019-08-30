import itertools
import numpy as np
from matplotlib.collections import LineCollection
from networkx import Graph, draw

from ecoglib.vis.colormaps import diverging_cm
from .variogram import binned_variance
from .kernels import matern_correlation, matern_spectrum


__all__ = ['covar_to_lines', 'covar_to_iqr_lines', 'make_matern_label', 'matern_demo', 'plot_electrode_graph']


def covar_to_lines(x, y, cen_fn=np.mean, len_fn=np.std, binsize=None, **lc_kwargs):
    """
    Take a variogram/correlogram cloud (x, y) and return plot elements for binned averages and a LineCollection
    marking the y-extent of binned standard errors.

    Parameters
    ----------
    x: ndarray
        Cloud x-axis
    y: ndarray
        Cloud y-axis
    cen_fn: callable
        Find the central tendency of the bin (default mean)
    len_fn: callable
        Find the dispersion of the bin (default stdev)
    binsize: float or None
        Approximate bin spacing, or use every unique grid spacing.
    lc_kwargs: dict
        Style options for the LineCollection

    Returns
    -------
    binned_pts: tuple
        (x, y) points for the binned variance
    lc: LineCollection
        LineCollection visualizing dispersion

    """

    xb, yb = binned_variance(x, y, binsize=binsize)
    y_c = np.array([cen_fn(_y) for _y in yb])
    y_l = np.array([len_fn(_y) for _y in yb])

    bin_lines = [[(xi, yi-li), (xi, yi+li)] for xi, yi, li in zip(xb, y_c, y_l)]
    return (xb, y_c), LineCollection(bin_lines, **lc_kwargs)


def covar_to_iqr_lines(x, y, binsize=None, **lc_kwargs):
    """
    This method acts identically to `covar_to_lines`, but the dispersion metric is inter-quartile range.

    """

    xb, yb = binned_variance(x, y, binsize=binsize)
    iqr = [np.percentile(y_, [25, 50, 75]) for y_ in yb]
    bin_lines = [[(x, y[0]), (x, y[2])] for (x, y) in zip(xb, iqr)]
    lc = LineCollection(bin_lines, **lc_kwargs)
    med = [y[1] for y in iqr]
    return (xb, med), lc


def make_matern_label(**params):
    """Helper function for plot labels."""
    label = u''
    if 'nu' in params:
        label = label + '\u03BD {nu:.1f} '
    if 'theta' in params:
        label = label + '\u03B8 {theta:.1f} (mm) '
    if 'nugget' in params:
        label = label + '\u03C3 {nugget:.2f} (uV^2) '
    if 'sill' in params:
        label = label + '\u03B6 {sill:.2f} (uV^2) '

    return label.format(**params)


def matern_demo(
        nus=[0.2, 0.5, 1, 2], thetas=[1, 2, 3, 4], spectrum=False, paired=False,
        reparam=False, context='notebook', figsize=None, semivar=False
):
    import matplotlib.pyplot as pp
    import seaborn as sns
    # sns.set_style('dark')
    # sns.set_palette('muted')
    xx = np.linspace(0, 2, 200) if spectrum else np.linspace(0.001, 5, 200)
    sns_st = {u'font.sans-serif': [u'Helvetica', u'Arial']}
    with sns.plotting_context(context), sns.axes_style('dark', rc=sns_st), \
         sns.color_palette('muted'):
        if paired:
            f, ax = pp.subplots()
            axs = [ax]
        else:
            f, axs = pp.subplots(2, 1, sharex=True, sharey=True, figsize=figsize)
        ls = itertools.cycle(('-', '--', '-.', ':'))
        if paired:
            for nu, theta in zip(nus, thetas):
                label = r'$\theta$={1:.1f}, $\nu$={0:.1f}'.format(nu, theta)
                if spectrum:
                    y = matern_spectrum(xx, theta=theta, nu=nu)
                    ax.plot(
                        xx, y / y[0],
                        label=label, ls=ls.next()
                    )
                else:
                    if reparam:
                        # theta = theta * (2*nu)**0.5
                        theta = theta / (2 * nu) ** 0.5
                        # label=r'$\theta$={1:.1f}, $\nu$={0:.1f}'.format(nu, theta)
                    y = matern_correlation(xx, nu=nu, theta=theta)
                    if semivar:
                        y = 1 - y
                    ax.plot(
                        xx, y,
                        label=label, ls=ls.next()
                    )
            ax.legend(loc='upper right')
            axs = [ax]
        else:
            for nu in nus:
                label = r'$\theta$=2, $\nu$={0:.1f}'.format(nu)
                if spectrum:
                    y = matern_spectrum(xx, theta=1.0, nu=nu)
                    axs[0].plot(
                        xx, y / y[0],
                        label=label, ls=ls.next()
                    )
                else:
                    theta = 2.0
                    if reparam:
                        # theta = (2*nu)**0.5
                        theta = theta / (2 * nu) ** 0.5
                        # label=r'$\theta$=1, $\nu$={0:.1f}'.format(nu)
                    y = matern_correlation(xx, nu=nu, theta=theta)
                    if semivar:
                        y = 1 - y
                    axs[0].plot(
                        xx, y,
                        label=label, ls=ls.next()
                    )
            ls = itertools.cycle(('-', '--', '-.', ':'))
            for theta in thetas:
                label = r'$\theta$={0}, $\nu$=1'.format(theta)
                if spectrum:
                    y = matern_spectrum(xx, theta=theta, nu=1.0)
                    axs[1].plot(
                        xx, y / y[0],
                        label=label, ls=ls.next()
                    )
                else:
                    if reparam:
                        # theta = theta * 2**0.5
                        theta = theta / (2 * nu) ** 0.5
                        # label=r'$\theta$={0}, $\nu$=1'.format(theta)
                    y = matern_correlation(xx, nu=nu, theta=theta)
                    if semivar:
                        y = 1 - y
                    axs[1].plot(
                        xx, y,
                        label=label,
                        ls=ls.next()
                    )

        for ax in axs:
            ax.legend(loc='upper right')
            if not spectrum:
                ax.set_yticks([0, 0.5, 1.0])
        if spectrum:
            axs[-1].set_xlabel('Spatial freq (mm$^{-1}$)')
        else:
            axs[-1].set_xlabel('Distance (mm)')
        for ax in axs:
            ax.set_ylabel('power (normalized)' if spectrum else 'Correlation')
        f.tight_layout()
    return f


def plot_electrode_graph(
        graph, chan_map, scale='auto', edge_colors=('black', 'red'), ax=None, stagger_x=False, stagger_y=False
):
    import matplotlib.pyplot as pp
    if not ax:
        figsize = np.array(chan_map.geometry)
        if scale == 'auto':
            scale = 8.0 / max(figsize)
        figsize = tuple(scale * figsize)
        f = pp.figure(figsize=figsize)
        ax = f.add_subplot(111)
    else:
        f = ax.figure

    ii, jj = chan_map.to_mat()
    if stagger_x:
        jj = jj + (np.random.rand(len(jj)) * 0.7 + 0.1)
    if stagger_y:
        ii = ii + (np.random.rand(len(ii)) * 0.7 + 0.1)
    n = len(graph)
    # plot in (x, y) coords, which is (jj, ii)
    pos = dict(enumerate(zip(jj, ii)))
    graph = graph.copy()
    graph.flat[::n + 1] = 0
    graph[np.isnan(graph)] = 1e-3
    rank = np.nansum(np.abs(graph), axis=0) / n
    # make node size somewhere between 40-80 pts
    nsize = 80 * rank + 40 * (1 - rank)
    G = Graph(graph)
    ew = graph[np.triu_indices(n, k=1)]

    cm = diverging_cm(-1, 1, edge_colors)
    draw(G, pos, edge_color=ew, edge_cmap=cm, with_labels=False, node_size=nsize, node_color=rank, cmap='Blues',
         width=np.abs(ew) ** 4, linewidths=0, ax=ax, edge_vmin=-1, edge_vmax=1)

    ax.axis('equal')
    ax.set_ylim(chan_map.geometry[0] - 0.5, -0.5)
    ax.set_xlim(-0.5, chan_map.geometry[1] - 0.5)
    cbar = pp.colorbar(ax.collections[0], ax=ax, use_gridspec=True)
    cbar.set_label('graph avg rank')
    return f

