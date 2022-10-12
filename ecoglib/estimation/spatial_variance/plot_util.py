import itertools
import numpy as np
from matplotlib.collections import LineCollection

from ecogdata.channel_map import ChannelMap, CoordinateChannelMap
from ecoglib.vis.colormaps import diverging_cm
from .variogram import binned_variance, semivariogram, fast_semivariogram
from .kernels import matern_correlation, matern_spectrum
from ...vis import plotters


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

    bin_lines = [[(xi, yi - li), (xi, yi + li)] for xi, yi, li in zip(xb, y_c, y_l)]
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


def plot_variogram(field, sites, fast=True, binsize=None, ax=None, trimmed=None, weight_bins=False,
                   lc_kwargs=dict(), **line_kwargs):
    plt = plotters.plt
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.figure
    if fast:
        x, svar = fast_semivariogram(field, sites, cloud=True, trimmed=trimmed)
    else:
        x, svar = semivariogram(field, sites, cloud=True, trimmed=trimmed)
    line_kwargs.setdefault('marker', 's')
    line_kwargs.setdefault('linestyle', '--')
    (x_bin, sv_med), sv_iqr = covar_to_iqr_lines(x, svar, binsize=binsize, **lc_kwargs)
    ax.plot(x_bin, sv_med, **line_kwargs)
    ax.add_collection(sv_iqr)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Semivariance')
    ax.autoscale()
    yl = ax.get_ylim()
    ax.set_ylim(bottom=min(0, yl[0]))
    xl = ax.get_xlim()
    ax.set_xlim(left=min(0, xl[0]))
    return f



def make_matern_label(**params):
    """Helper function for plot labels."""
    label = u''
    if 'nu' in params:
        label = label + '\u03BD {nu:.1f} '
    if 'theta' in params:
        label = label + '\u03B8 {theta:.1f} (mm) '
    if 'nugget' in params:
        label = label + '\u03C3 {nugget:.2f} (\u03BCV\u00B2) '
    if 'sill' in params:
        label = label + '\u03B6 {sill:.2f} (\u03BCV\u00B2) '

    return label.format(**params)


def matern_demo(
        nus=[0.2, 0.5, 1, 2], thetas=[1, 2, 3, 4], spectrum=False, paired=False,
        reparam=False, context='notebook', figsize=None, semivar=False
):
    plt = plotters.plt
    sns = plotters.sns
    # sns.set_style('dark')
    # sns.set_palette('muted')
    xx = np.linspace(0, 2, 200) if spectrum else np.linspace(0.001, 5, 200)
    sns_st = {u'font.sans-serif': [u'Helvetica', u'Arial']}
    with sns.plotting_context(context), sns.axes_style('dark', rc=sns_st), sns.color_palette('muted'):
        if paired:
            f, ax = plt.subplots()
            axs = [ax]
        else:
            f, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=figsize)
        ls = itertools.cycle(('-', '--', '-.', ':'))
        if paired:
            for nu, theta in zip(nus, thetas):
                label = r'$\theta$={1:.1f}, $\nu$={0:.1f}'.format(nu, theta)
                if spectrum:
                    y = matern_spectrum(xx, theta=theta, nu=nu)
                    ax.plot(
                        xx, y / y[0],
                        label=label, ls=next(ls)
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
                        label=label, ls=next(ls)
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
                        label=label, ls=next(ls)
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
                        label=label, ls=next(ls)
                    )
            ls = itertools.cycle(('-', '--', '-.', ':'))
            for theta in thetas:
                label = r'$\theta$={0}, $\nu$=1'.format(theta)
                if spectrum:
                    y = matern_spectrum(xx, theta=theta, nu=1.0)
                    axs[1].plot(
                        xx, y / y[0],
                        label=label, ls=next(ls)
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
                        ls=next(ls)
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


def plot_electrode_graph(graph: np.ndarray, chan_map: ChannelMap, scale: str='auto', node_map='rank',
                         edge_colors: tuple=('black', 'red'), edge_clim: tuple=(), max_edge_width: float=1,
                         ax: plotters.mpl.axes.Axes=None, stagger_x: bool=False, stagger_y: bool=False):
    if not ax:
        plt = plotters.plt
        figsize = np.array(chan_map.geometry[::-1])
        if scale == 'auto':
            scale = 8.0 / max(figsize)
        figsize = tuple(scale * figsize)
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(111)
        ax.axis('equal')
    else:
        f = ax.figure
    mpl = plotters.mpl
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
    if isinstance(node_map, np.ndarray):
        node = node_map
        cb_label = ''
    elif node_map == 'rank':
        rank = np.nansum(np.abs(graph), axis=0) / (n - 1)
        node = rank
        cb_label = 'Average graph rank'
    else:
        node = None
    # make node size somewhere between 40-80 pts
    # nsize = 80 * rank + 40 * (1 - rank)
    if node is None:
        nsize = 60
    else:
        nsize = (80 * node + 40 * (node.max() - node)) / node.max()
    # G = Graph(graph)
    ew = graph[np.triu_indices(n, k=1)]

    if not len(edge_clim):
        amx = np.abs(ew).max()
        edge_clim = (-amx, amx)
    cm = diverging_cm(edge_clim[0], edge_clim[1], edge_colors)

    # Transpose these positions to (x, y)
    loc_a = chan_map.site_combinations.idx1[:, ::-1]
    loc_b = chan_map.site_combinations.idx2[:, ::-1]
    lines = [np.array([i1, i2]) for i1, i2 in zip(loc_a, loc_b)]
    norm = mpl.colors.Normalize(-amx, amx)
    edge_colors = cm(norm(ew))
    edge_width = max_edge_width * (np.abs(ew) / edge_clim[1]) ** 2
    lines = LineCollection(lines, colors=edge_colors, linewidths=edge_width)
    ax.add_collection(lines)
    if node is not None:
        sct = ax.scatter(jj, ii, s=nsize, c=node, cmap='Blues', edgecolors='none', zorder=10,
                         vmin=0, vmax=1)

    if isinstance(chan_map, CoordinateChannelMap):  # np.all(ii.astype('l') == ii):
        row_lim = ii.min(), ii.max()
        row_lim = [row_lim[0] - 0.025 * (row_lim[1] - row_lim[0]),
                   row_lim[1] + 0.025 * (row_lim[1] - row_lim[0])]
        col_lim = jj.min(), jj.max()
        col_lim = [col_lim[0] - 0.025 * (col_lim[1] - col_lim[0]),
                   col_lim[1] + 0.025 * (col_lim[1] - col_lim[0])]
        ax.set_xlim(col_lim)
        ax.set_ylim(row_lim)
    else:
        ax.set_ylim(chan_map.geometry[0] - 0.5, -0.5)
        ax.set_xlim(-0.5, chan_map.geometry[1] - 0.5)
    if node is not None:
        cbar = f.colorbar(sct, ax=ax, use_gridspec=True)
        cbar.set_label(cb_label)
    return f
