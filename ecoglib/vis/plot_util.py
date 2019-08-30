"""
Many visualization utilities
"""

import matplotlib as mpl
import matplotlib.cm
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from matplotlib.ticker import MaxNLocator
import numpy as np
from copy import copy

from ecogdata.numutil import bootstrap_stat

from .tile_images import quick_tiles, calibration_axes  # need to de-pyplot
from .colormaps import rgba_field  # need to de-pyplot

import seaborn as sns
sns.reset_orig()


# just use Figure rather than figure
def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
             subplot_kw=None, gridspec_kw=None, **fig_kw):
    """
    Create a figure with a set of subplots already made.
    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.
    Keyword arguments:
      *nrows* : int
        Number of rows of the subplot grid.  Defaults to 1.
      *ncols* : int
        Number of columns of the subplot grid.  Defaults to 1.
      *sharex* : string or bool
        If *True*, the X axis will be shared amongst all subplots.  If
        *True* and you have multiple rows, the x tick labels on all but
        the last row of plots will have visible set to *False*
        If a string must be one of "row", "col", "all", or "none".
        "all" has the same effect as *True*, "none" has the same effect
        as *False*.
        If "row", each subplot row will share a X axis.
        If "col", each subplot column will share a X axis and the x tick
        labels on all but the last row will have visible set to *False*.
      *sharey* : string or bool
        If *True*, the Y axis will be shared amongst all subplots. If
        *True* and you have multiple columns, the y tick labels on all but
        the first column of plots will have visible set to *False*
        If a string must be one of "row", "col", "all", or "none".
        "all" has the same effect as *True*, "none" has the same effect
        as *False*.
        If "row", each subplot row will share a Y axis and the y tick
        labels on all but the first column will have visible set to *False*.
        If "col", each subplot column will share a Y axis.
      *squeeze* : bool
        If *True*, extra dimensions are squeezed out from the
        returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the
          resulting single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy
          object array of Axis objects are returned as numpy 1-d
          arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d
          array.
        If *False*, no squeezing at all is done: the returned axis
        object is always a 2-d array containing Axis instances, even if it
        ends up being 1x1.
      *subplot_kw* : dict
        Dict with keywords passed to the
        :meth:`~matplotlib.figure.Figure.add_subplot` call used to
        create each subplots.
      *gridspec_kw* : dict
        Dict with keywords passed to the
        :class:`~matplotlib.gridspec.GridSpec` constructor used to create
        the grid the subplots are placed on.
      *fig_kw* : dict
        Dict with keywords passed to the :func:`figure` call.  Note that all
        keywords not recognized above will be automatically included here.
    Returns:
    fig, ax : tuple
      - *fig* is the :class:`matplotlib.figure.Figure` object
      - *ax* can be either a single axis object or an array of axis
        objects if more than one subplot was created.  The dimensions
        of the resulting array can be controlled with the squeeze
        keyword, see above.
    Examples::
        x = np.linspace(0, 2*np.pi, 400)
        y = np.sin(x**2)
        # Just a figure and one subplot
        f, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title('Simple plot')
        # Two subplots, unpack the output array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(x, y)
        ax1.set_title('Sharing Y axis')
        ax2.scatter(x, y)
        # Four polar axes
        plt.subplots(2, 2, subplot_kw=dict(polar=True))
        # Share a X axis with each column of subplots
        plt.subplots(2, 2, sharex='col')
        # Share a Y axis with each row of subplots
        plt.subplots(2, 2, sharey='row')
        # Share a X and Y axis with all subplots
        plt.subplots(2, 2, sharex='all', sharey='all')
        # same as
        plt.subplots(2, 2, sharex=True, sharey=True)
    """
    # for backwards compatibility
    if isinstance(sharex, bool):
        if sharex:
            sharex = "all"
        else:
            sharex = "none"
    if isinstance(sharey, bool):
        if sharey:
            sharey = "all"
        else:
            sharey = "none"
    share_values = ["all", "row", "col", "none"]
    if sharex not in share_values:
        # This check was added because it is very easy to type
        # `subplots(1, 2, 1)` when `subplot(1, 2, 1)` was intended.
        # In most cases, no error will ever occur, but mysterious behavior will
        # result because what was intended to be the subplot index is instead
        # treated as a bool for sharex.
        if isinstance(sharex, int):
            import warnings
            warnings.warn("sharex argument to subplots() was an integer."
                          " Did you intend to use subplot() (without 's')?")

        raise ValueError("sharex [%s] must be one of %s" %
                         (sharex, share_values))
    if sharey not in share_values:
        raise ValueError("sharey [%s] must be one of %s" %
                         (sharey, share_values))
    if subplot_kw is None:
        subplot_kw = {}
    if gridspec_kw is None:
        gridspec_kw = {}

    fig = Figure(**fig_kw)
    gs = GridSpec(nrows, ncols, **gridspec_kw)

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    nplots = nrows * ncols
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(gs[0, 0], **subplot_kw)
    axarr[0] = ax0

    r, c = np.mgrid[:nrows, :ncols]
    r = r.flatten() * ncols
    c = c.flatten()
    lookup = {
        "none": np.arange(nplots),
        "all": np.zeros(nplots, dtype=int),
        "row": r,
        "col": c,
    }
    sxs = lookup[sharex]
    sys = lookup[sharey]

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        if sxs[i] == i:
            subplot_kw['sharex'] = None
        else:
            subplot_kw['sharex'] = axarr[sxs[i]]
        if sys[i] == i:
            subplot_kw['sharey'] = None
        else:
            subplot_kw['sharey'] = axarr[sys[i]]
        axarr[i] = fig.add_subplot(gs[i // ncols, i % ncols], **subplot_kw)

    # returned axis array will be always 2-d, even if nrows=ncols=1
    axarr = axarr.reshape(nrows, ncols)

    # turn off redundant tick labeling
    if sharex in ["col", "all"] and nrows > 1:
        # turn off all but the bottom row
        for ax in axarr[:-1, :].flat:
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)

    if sharey in ["row", "all"] and ncols > 1:
        # turn off all but the first column
        for ax in axarr[:, 1:].flat:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            ret = fig, axarr[0, 0]
        else:
            ret = fig, axarr.squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        ret = fig, axarr.reshape(nrows, ncols)

    return ret


def embedded_frames(frames, geometry, gap=0.05, border=0.05, fill=np.nan):
    # frames is a sequence of (N x M [x T]) matrices
    # they will be arrayed in "c major" order -- ie rasterized left to right,
    # from top to bottom

    f_row, f_col = geometry

    m, n = frames[0].shape[:2]

    gap, border = [round(x * (f_row * m + f_col * n) / 2.0) for x in (gap, border)]
    gap, border = map(int, (gap, border))

    big_i = f_row * m + (f_row - 1) * gap + 2 * border
    big_j = f_col * n + (f_col - 1) * gap + 2 * border

    offsets_i = border + np.arange(f_row) * (m + gap)
    offsets_j = border + np.arange(f_col) * (n + gap)

    big_shape = (big_i, big_j)
    if len(frames[0].shape) > 2:
        big_shape = big_shape + (frames[0].shape[2],)

    big_frames = np.empty(big_shape, dtype=frames[0].dtype)
    # let this raise exception for integer types?
    big_frames.fill(fill)

    for i in range(f_row):
        for j in range(f_col):
            i_sl = slice(offsets_i[i], offsets_i[i] + m)
            j_sl = slice(offsets_j[j], offsets_j[j] + n)
            big_frames[i_sl, j_sl] = frames[j + i * f_col]
    return big_frames


def filled_interval(
        pfun, x, fx, f_itvl, color=None,
        ax=None, ec='none', alpha=0.2,
        fillx=False, **pfun_kwargs
):
    """Line plot with a shaded margin

    Parameters
    ----------
    pfun : callable
        Plotting function, i.e. pyplot.plot, pyplot.semilogy, ...
    x : array-like
        Abscissa of plot
    fx : array-like
        Ordinate of plot
    f_itvl : array-like
        Interval spec for shaded margin. If 1D, then fill the margin
	    everywhere between (fx - f_itvl, fx + f_itvl). If
    	len(f_itvl)==2, then fill the margin between these two levels.
    color : color spec
        Line color (margin is white-blended)
    ax : matplotlib axes (optional)
        axes to plot into. Note that the pfun argument would be
	    ax.plot, ax.semilogy, etc.
    ec : color (optional)
        Optional edgecolor for the margin (default 'none' shows no
	    distinct edge).
    alpha : float
        Mix the linecolor with (1 - alpha) parts white to get the
	    margin color.
    fillx : {True/False}
        If True, reverse the roles of x and fx and make a horizontal
    	margin.
    pfun_kwargs : dict
        Other keyword arguments for the plotting function.

    Returns
    -------
    f : matplotlib figure
    ln : list
        lines returned by the plotting function

    """

    if not ax:
        import matplotlib.pyplot as pp
        f = pp.figure()
        ax = f.add_subplot(111)
    else:
        f = ax.figure
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']

    (xx, fxx) = (fx, x) if fillx else (x, fx)

    ln = pfun(xx, fxx, color=color, **pfun_kwargs)
    if (isinstance(f_itvl, (list, tuple)) and len(f_itvl) == 2) or \
            (isinstance(f_itvl, np.ndarray) and f_itvl.ndim == 2):
        f_lo, f_hi = f_itvl
    else:
        f_lo = fx - f_itvl
        f_hi = fx + f_itvl

    fill_fun = ax.fill_betweenx if fillx else ax.fill_between
    elw = pfun_kwargs.get('linewidth',
                          pfun_kwargs.get('lw', mpl.rcParams['lines.linewidth']))
    fill_fun(
        x, f_hi, f_lo,
        facecolor=color, alpha=alpha, edgecolor=ec, lw=0.8 * elw
    )
    return f, ln


def plot_on_density(
        t, bkgrnd, frgrnd, ax, b_map='gray_r',
        traces=True, sem=True, outline_sigma=0,
        sigma_color='#005AFF', img_size=201,
        **plot_kws
):
    """
    Plot traces (or center & margin) on top of an image (heatmap)
    of the background variation. Background variation is depicted
    as a normal density with mu and sigma estimated for each time
    point.
    """

    from scipy.stats.distributions import norm
    if traces:
        ax.plot(t, frgrnd.T, **plot_kws)
    else:
        r_mn = frgrnd.mean(0)
        r_se = frgrnd.std(0)
        if sem:
            r_se /= np.sqrt(len(frgrnd))
        color = plot_kws.pop('color', 'orange')
        filled_interval(ax.plot, t, r_mn, r_se, color,
                        alpha=0.5, ax=ax, zorder=40, **plot_kws)

    b_mn = bkgrnd.mean(0)
    b_sigma = bkgrnd.std(0)
    # mx = np.abs(b_mn).max() + 3 * b_sigma.max()
    mx = np.nanmedian(np.abs(b_mn)) + 4 * np.nanmedian(b_sigma)
    d_samp = np.linspace(-mx, mx, img_size)
    bs_image = np.row_stack([norm.pdf(d_samp, mu, sig)
                             for mu, sig in zip(b_mn, b_sigma)])
    bs_image *= 0.65 / bs_image.max(axis=1, keepdims=1)
    bs_rgba, _ = rgba_field(b_map, bs_image.T, afield=bs_image.T,
                            clim=(0, 1), alim=(0, 0.1 * np.nanmax(bs_image)))
    # manually mix RGB levels for correct PDF output
    ax_bg = mpl.colors.colorConverter.to_rgb(mpl.rcParams['axes.facecolor'])
    ax_bg = (np.array(ax_bg) * 255.0)
    alpha = bs_rgba[..., -1:].astype('d') / 255.0
    bs_rgb = (bs_rgba[..., :3].astype('d') * alpha + (1 - alpha) * ax_bg).astype('B')

    ax.imshow(bs_rgb,  # cmap=b_map, clim=(0, 1),
              extent=[t.min(), t.max(), -mx, mx], origin='lower')
    ## ax.imshow(bs_image.T, cmap=b_map, clim=(0, 1),
    ##           extent=[t.min(), t.max(), -mx, mx])
    if outline_sigma > 0:
        if 'lw' in plot_kws:
            lw = plot_kws['lw'] * 0.25
        elif 'linewidth' in plot_kws:
            lw = plot_kws['linewidth'] * 0.25
        else:
            lw = mpl.rcParams['lines.linewidth'] * 0.25
        ax.plot(t, -outline_sigma * b_sigma,
                ls='-', lw=lw, color=sigma_color)
        ax.plot(t, outline_sigma * b_sigma,
                ls='-', lw=lw, color=sigma_color)
    ax.axis('auto')
    # ax.set_ylim(-mx, mx)
    ax.set_xlim(t.min(), t.max())
    return


def _median_ci(samps, boots=1000, ci=95.0):
    """
    Returns (med_ci_lo, median, med_ci_hi) and then 25th and 7th pctiles.
    """

    q1 = bootstrap_stat(samps, func=np.percentile, args=[25], n_boot=boots)
    md = bootstrap_stat(samps, func=np.percentile, args=[50], n_boot=boots)
    q3 = bootstrap_stat(samps, func=np.percentile, args=[75], n_boot=boots)

    margin = 100.0 - ci
    md_lo, md, md_hi = np.percentile(
        md, [margin / 2.0, 50, 100 - margin / 2.0], axis=0
    )

    q1 = q1.mean(0)
    q3 = q3.mean(0)
    return (md_lo, md, md_hi), (q1, q3)


def light_boxplot(
        samps, names=(), colors=(), desat=0.6, box_lw=1, box_ls='dotted',
        box_w=None, mark_outliers=True, ax=None, figheight=5.0,
        figwidth=None, whiskers=False, sampcount=False, mark_mean=False,
        notch_med=True, jitterx=False, horiz=False, outlier_range=3,
        **plot_kws
):
    """Box-and-whisker plots of one or more samples.

    The default style of these boxplots are a dotted hourglass
    outlining the interquartile range of a sample with a marked median
    level. The hourglass notch indicates the 95% confidence range of
    the median under bootstrap resampling. The "whisker" element is by
    default a scatter plot of the actual inliers in the sample.

    Many variations can be made!

    Parameters
    ----------
    samps : sequence
        Sequence of sample vectors (do not need to be the same length)
    names : sequence (optional)
        Sequence of sample labels
    colors : {sequence | string} (optional)
        Sequence of colors to use for each sample, or a single color
        for all samples
    mark_outliers : {True/False | string}
        Mark individual outliers or not. Can provide a matplotlib
        marker code.
    whiskers : {True/False}
        Draw whiskers rather than actual sample scatter
    sampcount : {True/False}
        Write the number of points per sample in the plot
    jitterx : {True/False}
        Add jitter to the sample strip scatter plot (helpful to
	    visualize larger samples).
    horiz : {True/False}
        Orient the plot horizontally if True (default False for
        vertical)
    outlier_range : float
        The IQR factor used for outlier detection.
    ax : matplotlib Axes
        If given, plot into this axes
    plot_kws : dict
        Additional keyword arguments used in plotting sample
        strips. Also applies to plotting outliers.

    Returns
    -------
    fig : matplotlib figure

    """

    if not np.iterable(samps[0]):
        samps = [samps]
    outlier_kws = copy(plot_kws)
    # outlier_kws['mec'] = 'red'
    outlier_kws['mew'] = 0.5 * box_lw
    if box_w is None and figwidth is None:
        box_w = 0.1
        x_sep = 5.0 * box_w
    elif box_w is not None and figwidth is None:
        x_sep = 5.0 * box_w
    elif figwidth is not None and box_w is None:
        x_sep = float(figwidth - 0.75) / len(samps)
        box_w = x_sep / 5.0
    else:
        raise ValueError('box_w and figwidth conflict')

    x_ax = np.arange(len(samps)) * x_sep
    if not ax:
        import matplotlib.pyplot as pp
        fs = (figheight, len(samps) * x_sep + 0.75) if horiz else \
            (len(samps) * x_sep + 0.75, figheight)
        f = pp.figure(figsize=fs)
        ax = f.add_subplot(111)
    else:
        f = ax.figure
    samp_sizes = list()
    notches = [notch_med] * len(samps)
    for n in range(len(samps)):
        s = np.ma.compressed(np.ma.masked_invalid(samps[n]))
        samp_sizes.append(len(s))
        if not len(colors):
            c = 'k'
        elif isinstance(colors, str):
            c = colors
        else:
            c = colors[n]
        dc = sns.desaturate(c, desat)
        notch = notches[n]
        if notch:
            med, qs = _median_ci(s)
            med_lo, med, med_hi = med
            if med_lo < qs[0] or med_hi > qs[1]:
                notch = False
        else:
            q0, med, q1 = np.percentile(s, [25, 50, 75])
            qs = (q0, q1)
        if mark_outliers:
            iqr = qs[1] - qs[0]
            k = outlier_range / 2.0
            mask = (s >= qs[0] - k * iqr) & (s <= qs[1] + k * iqr)
            inliers = s[mask]
            outliers = s[~mask]
        else:
            inliers = s
            outliers = ()
        if len(inliers):
            if whiskers:
                x = [x_ax[n], x_ax[n]]
                # y = [ qs[0] - 1.5 * iqr, qs[1] + 1.5 * iqr ]
                y = [inliers.min(), inliers.max()]
                x_, y_ = (y, x) if horiz else (x, y)
                ax.plot(x_, y_, ls='-', color=c, lw=box_lw)
            else:
                x = np.ones(len(inliers)) * x_ax[n]
                if jitterx:
                    # add left-right jitter at 45% of the "box_w"
                    x += (np.random.rand(len(x)) - 0.5) * 0.45 * box_w
                x_, y_ = (inliers, x) if horiz else (x, inliers)
                ax.plot(
                    x_, y_,
                    linestyle='none', marker='.',
                    color=c, **plot_kws
                )
        if len(outliers):
            marker = mark_outliers if isinstance(mark_outliers, str) else '+'
            if marker != 'none':
                outlier_kws['mec'] = c
                x = np.ones(len(outliers)) * x_ax[n]
                ## if jitterx:
                ##     # add left-right jitter at 45% of the "box_w"
                ##     x += ( np.random.rand(len(x)) - 0.5 ) * 0.45 * box_w
                x_, y_ = (outliers, x) if horiz else (x, outliers)
                ax.plot(
                    x_, y_,
                    linestyle='none', marker=marker,
                    color=c, **outlier_kws
                )

        ## med = np.median(s)
        ## iqr = np.percentile(s, [25, 75])
        # x0, x1 = (n+1-box_w), (n+1+box_w)
        if notch:
            # Draw an hour-glass shaped path.
            # The notch part of the hour glass encompasses the 95%
            # confidence interval of the mean. The rest of the
            # hour glass encompasses the interquartile range.
            # This path starts from the bottom left
            x0, x1 = (x_ax[n] - box_w), (x_ax[n] + box_w)
            path = [(x0 - box_w, qs[0]), (x1 + box_w, qs[0]),
                    (x1 + box_w, med_lo), (x1, med), (x1 + box_w, med_hi),
                    (x1 + box_w, qs[1]), (x0 - box_w, qs[1]),
                    (x0 - box_w, med_hi), (x0, med), (x0 - box_w, med_lo)
                    ]
        else:
            # Draw a rectangle encompassing the 25-75 percentiles
            x0, x1 = (x_ax[n] - 2 * box_w), (x_ax[n] + 2 * box_w)
            path = [(x0, qs[0]), (x1, qs[0]),
                    (x1, qs[1]), (x0, qs[1])]
        if horiz:
            path = [p[::-1] for p in path]
            ax.plot([med, med], [x0, x1], color=dc, lw=2 * box_lw)
        else:
            ax.plot([x0, x1], [med, med], color=dc, lw=2 * box_lw)
        poly = Polygon(
            np.array(path), closed=True,
            facecolor='none', edgecolor=dc,
            lw=box_lw, ls=box_ls
        )
        ax.add_patch(poly)

        if mark_mean:
            mu = s.mean()
            if isinstance(mark_mean, dict):
                mn_kws = mark_mean
            else:
                # lighten the mean line slightly
                lc = .67 * np.array(dc) + .33 * np.ones(len(dc))
                mn_kws = dict(color=lc, lw=1.25)
            if horiz:
                ax.plot([mu, mu], [x0 + .25 * box_w, x1 - .25 * box_w], **mn_kws)
            else:
                ax.plot([x0 + .25 * box_w, x1 - .25 * box_w], [mu, mu], **mn_kws)

    # ax.set_xticks(np.arange(1, len(samps)+1))
    if horiz:
        ax.set_yticks(x_ax)
        if len(names):
            ax.set_yticklabels(names)
        ax.grid('off', axis='y')
        ax.set_ylim(-x_sep / 2, (len(samps) - 0.5) * x_sep)
    else:
        ax.set_xticks(x_ax)
        if len(names):
            ax.set_xticklabels(names)
        ax.grid('off', axis='x')
        ax.set_xlim(-x_sep / 2, (len(samps) - 0.5) * x_sep)

    if sampcount:
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )
        samp_height = sampcount if isinstance(sampcount, float) else 0.95
        for n in range(len(samps)):
            x_, y_ = (samp_height, x_ax[n]) if horiz else (x_ax[n], samp_height)
            ax.text(
                x_, y_, 'n={0}'.format(samp_sizes[n]),
                va='baseline', ha='center', transform=trans,
                fontsize=float(mpl.rcParams['font.size'])
            )
    return f


def line_dist(
        samps, pts_or_names=(),
        mark_median=True, mark_mean=False, lcolor=None, mwid=10, lw=2.0,
        plot_outliers=True, iqr_fac=3, ax=None, **marker_kws
):
    """Minimized box-and-whisker plots of one or more samples.

    This method functions much like "light_boxplot()", but reduces the
    interquartile box to a thick line drawn over a thinner line representing
    whiskers. Whiskers represent the extent of "inlier"
    samples. Median and/or mean levels are notched with a horizontal
    stripe

    Parameters
    ----------
    samps : sequence
        Sequence of sample vectors (do not need to be the same length)
    pts_or_names : sequence (optional)
        Sequence of x-axis ticks (if numerical) or sample names (if
	strings)
    mark_median : {True/False | string}
        Add median level mark, a given string indicates line
        color (default True)
    mark_mean : {True/False | string}
        Add mean level mark, a given string indicates line
        color (default True)
    lcolor : matplotlib color spec
        Color for box & whisker lines (if None, default color cycle is used)
    lw : float
        Width of interquartile line (default 2.0)
    plot_outliers : {True/False}
        Make individual marks of outliers (default True)
    iqr_fac : float
        Outlier threshold factor
    ax : matplotlib Axes
        If given, plot into this axes

    Returns
    -------
    fig : matplotlib figure

    """

    if ax is None:
        import matplotlib.pyplot as pp
        f, ax = pp.subplots()
    else:
        f = ax.figure

    # get plot elements
    if not len(pts_or_names) or isinstance(pts_or_names[0], str):
        x_labels = pts_or_names
        x_axis = np.arange(len(samps))
    else:
        x_labels = None
        x_axis = pts_or_names
    color_cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    if lcolor is None:
        lcolor = color_cycle[0]
    if mark_median:
        mdcolor = mark_median if isinstance(mark_median, str) else color_cycle[1]
    if mark_mean:
        mncolor = mark_mean if isinstance(mark_mean, str) else color_cycle[2]

    outliers = []
    box_lines = []
    whisk_lines = []
    meds = []
    means = []
    for x, samp in zip(x_axis, samps):
        samp = np.ma.masked_invalid(samp).compressed()
        lo, md, hi = np.percentile(samp, [25, 50, 75])
        iqr = hi - lo

        box_lines.append([(x, lo), (x, hi)])

        hi_out = hi + iqr_fac / 2.0 * iqr
        lo_out = lo - iqr_fac / 2.0 * iqr
        mn = samp.min()
        mx = samp.max()
        out = samp[(samp < lo_out) | (samp > hi_out)]
        if plot_outliers:
            outliers.append(out)
            lo_out = max(lo_out, mn)
            hi_out = min(hi_out, mx)
            whisk_lines.append([(x, lo_out), (x, hi_out)])
        else:
            whisk_lines.append([(x, mn), (x, mx)])
        if mark_median:
            meds.append(md)
        if mark_mean:
            means.append(np.nanmean(samp))
    lcolor = mpl.colors.to_rgb(lcolor)
    whisk_lines = LineCollection(whisk_lines, colors=0.5 + 0.5 * np.array(lcolor), linewidths=lw / 2.0)
    box_lines = LineCollection(box_lines, colors=lcolor, linewidths=lw, zorder=10)
    ax.add_collection(box_lines)
    ax.add_collection(whisk_lines)
    if mark_median:
        ax.plot(x_axis, meds, ls='none', marker='_', ms=mwid, mew=1, mec=mdcolor, color=mdcolor, zorder=11)
    if mark_mean:
        ax.plot(x_axis, means, ls='none', marker='_', ms=mwid, mew=1, mec=mncolor, color=mncolor, zorder=11)
    if plot_outliers and len(outliers):
        x_ = [[x] * len(outl) for x, outl in zip(x_axis, outliers)]
        x_ = np.concatenate(x_)
        outliers = np.concatenate(outliers)
        marker_kws.setdefault('marker', '+')
        marker_kws.setdefault('mec', 'k')
        marker_kws.setdefault('mew', 0.5)
        marker_kws.setdefault('ms', 4)
        ax.plot(x_, outliers, ls='none', **marker_kws)

    if x_labels is not None:
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(5))
    return f


def mark_axes_edge(
        ax, pts, widths, color, axis='x', where='bottom', size=10
):
    """
    Draws marks outside of the axes box. Useful for annotating events
    or levels.

    Parameters
    ----------
    ax : Axes instance
    pts : sequence
        x- or y-data locations to make marks
    widths : {sequence | constant}
        length of mark along axis
    color : {sequence | single color-code}
        color(s) of marks
    axis : string
        Mark the 'x' or 'y' axis
    where : {'top' | 'bottom'}
        Which axis to mark. Corresponds to 'right' and 'left' if axis=='y'
    size : {sequence | constant}
        Linewidth of the drawn mark

    Returns
    -------
    lines : list
        The marks

    """

    # normalize input parameters
    if not np.iterable(pts):
        pts = [pts]
    if isinstance(color, str) or not isinstance(color, (list, tuple, np.ndarray)):
        color = [color] * len(pts)
    if not np.iterable(widths):
        widths = [widths] * len(pts)
    if not np.iterable(size):
        size = [size] * len(pts)

    odd_ax = where in ('top', 'right')
    lim = 1 if odd_ax else 0
    edge = ax.get_ylim()[lim] if axis == 'x' else ax.get_xlim()[lim]
    coord = 1 if axis == 'x' else 0
    # get the fig coordinates of the bottom edge
    edge_coord = (0, edge) if axis == 'x' else (edge, 0)
    ax_edge_dots = ax.transData.transform(edge_coord)
    if odd_ax:
        ax_edge_dots[coord] += max(size) / 2.0 + 5
    else:
        ax_edge_dots[coord] -= (max(size) / 2.0 + 5)
    ln_edge_ax = ax.transAxes.inverted().transform(ax_edge_dots)[coord]

    def _make_coord(x):
        # if axis is y, we're interested in forming points like (<any>, x)
        # otherwise, points like (x, <any>)
        if axis == 'x':
            return np.array([x, ln_edge_ax])
        else:
            return np.array([ln_edge_ax, x])

    coord = 1 - coord
    # get the stepsize of one data unit in axes units
    fig_u = ax.transData.get_matrix()[coord, coord]
    ax_u = ax.transAxes.inverted().get_matrix()[coord, coord] * fig_u
    lines = list()
    fig_to_data = ax.transData.inverted()
    ax_to_fig = ax.transAxes
    for p, w, c, s in zip(pts, widths, color, size):
        # look up start point for the line
        ax_p = ax.transAxes.inverted().transform(
            ax.transData.transform(_make_coord(p))
        )[coord]
        c1 = fig_to_data.transform(ax_to_fig.transform(_make_coord(ax_p)))
        c2 = fig_to_data.transform(ax_to_fig.transform(_make_coord(ax_p + w * ax_u)))
        ln = Line2D(*list(zip(c1, c2)), color=c, linewidth=s, solid_capstyle='butt')
        ax.add_line(ln)
        ln.set_clip_on(False)
        lines.append(ln)
    return lines


def stacked_epochs_image(
        tx, chan_samps, cond_a, cond_b, breaks, clim=(), title='', tm=None,
        stacked='cond_a', cmap='jet', clabel=r"$\mu V$", bcolor='white'
):
    """
    cond_a : list of labels for condition A
    cond_b : list of labels for condition B

    """
    # chan_samps should be (len(cond_a), len(cond_b), n_trials, n_pts),
    # if not, then fix here
    shape = list(chan_samps.shape)
    if len(shape) < 4:
        if stacked == 'cond_a':
            shape.insert(1, 1)
        else:
            shape.insert(0, 1)
        chan_samps = chan_samps.reshape(shape)

    cond_a, cond_b = map(np.atleast_1d, (cond_a, cond_b))
    nframe = len(cond_a) if stacked == 'cond_b' else len(cond_b)
    ## fig, axs = tile_images.quick_tiles(
    ##     nframe, nrow=1, tilesize=(4.0, 4.5), title=title,
    ##     calib='right'
    ##     )
    fig, axs = quick_tiles(
        nframe, nrow=1, tilesize=(2.0, 4.5), title=title,
        calib='right'
    )

    if not clim:
        mn = chan_samps.min();
        mx = chan_samps.max()
        clim = (mn, mx)

    if stacked == 'cond_a':
        chan_samps = np.rollaxis(chan_samps, 1)
        x_labels = cond_b
        y_labels = cond_a
    else:
        x_labels = cond_a
        y_labels = cond_b

    n_stacked = chan_samps.shape[1] * chan_samps.shape[2]
    if tx[-1] - tx[0] < 5:
        t0, tf = tx[0] * 1000, tx[-1] * 1000
    else:
        t0, tf = tx[0], tx[-1]
    extent = [t0, tf, 0, n_stacked - 1]

    for n, ax in enumerate(axs):
        img = chan_samps[n].copy().reshape(-1, shape[-1])
        ax.imshow(img, extent=extent, clim=clim, cmap=cmap, origin='lower')
        ax.axis('auto')
        for b in breaks:
            ax.axhline(b, color=bcolor, linestyle='--', linewidth=1.5)

        ax.set_ylim(0, n_stacked - 1)
        if n == 0:
            ticks = 0.5 * (np.r_[0, breaks] + np.r_[breaks, n_stacked])
            ax.set_yticks(ticks)
            ax.set_yticklabels(y_labels)
        else:
            ax.yaxis.set_visible(False)
        if len(axs) > 3:
            ax.tick_params(labelsize=8)
        else:
            ax.tick_params(labelsize=10)

        ax.set_xlabel(x_labels[n])
        # ax.xaxis.set_label_position('top')

        if tm is not None:
            ax.axvline(tm, color='k', linestyle='--', linewidth=1)
        if t0 < 0 and tf > 0:
            ax.axvline(0, color=(.5, .5, .5), linestyle='--')
        ax.set_xlim(t0, tf)

    figwid = fig.get_figwidth()
    fig.subplots_adjust(
        top=0.925,
        left=0.5 / figwid,
        bottom=0.1,
        wspace=0.05
    )

    bbox = axs[-1].get_position()
    # situate new axes in 2nd quarter of the region to the
    # right of the last plotted axes
    edge = 1 - 1 / figwid  # 0.8
    x1 = 11 * edge / 12 + 1.0 / 12
    xw = (1 - edge) / 6.
    fig.subplots_adjust(right=edge)
    cax = fig.add_axes([x1, bbox.y0, xw, bbox.y1 - bbox.y0])
    cbar = fig.colorbar(ax.images[0], cax=cax)
    cax.tick_params(labelsize=10)
    cbar.set_label(clabel)
    return fig


def stacked_epochs_traces(
        tx, chan_samps, labels, title='', tm=None, calib_unit='V'
):
    # plots n groups of stacks, each stack is m[n] traces tall
    # * chan_samps is iterable with length-n
    # * each item in chan_samps is converted to a LineCollection

    len_tx = tx[-1] - tx[0]
    tx_pad = 0.1 * len_tx

    n = len(chan_samps)
    m_mx = max([len(s) for s in chan_samps])
    import matplotlib.pyplot as pp
    fig = pp.figure(figsize=(1.5 * n + 0.5, 6))
    ax = fig.add_subplot(111)

    all_line_groups = list()
    all_offsets = np.zeros(n)
    for ni, samps in enumerate(chan_samps):
        line_group = list()
        all_offsets[ni] = np.median(samps.ptp(axis=1))
        for timeseries in samps:
            line = list(zip(tx + ni * (len_tx + tx_pad), timeseries))
            line_group.append(line)
        all_line_groups.append(line_group)

    offset = np.median(all_offsets)
    for ni, line_group in enumerate(all_line_groups):
        lc = LineCollection(
            line_group, offsets=(0, offset), colors='k', linewidths=0.5
        )
        ax.add_collection(lc)
        if tm is not None:
            ax.axvline(tm + ni * (len_tx + tx_pad), color='r', ls='--')

        # put label at center of this stack
        x0 = tx[0] + ni * (len_tx + tx_pad)
        x1 = x0 + len_tx
        xc = 0.5 * (x0 + x1)
        ax.text(
            xc, -2.5 * offset, labels[ni],
            transform=ax.transData, ha='center'
        )

    ax.set_xlim(tx[0] - tx_pad, n * (len_tx + tx_pad) + tx[0])
    ax.set_ylim(-offset, m_mx * offset)
    ax.axis('off')
    ax.set_title(title)
    # leave 1 inch for scale
    w = fig.get_figwidth()
    fig.subplots_adjust(left=0.01, right=max(0.85, 1 - 1 / float(w)))

    calib_ax = calibration_axes(
        ax, y_scale=2 * offset, calib_unit=calib_unit
    )

    return fig


def plot_samples(
        tests, samples, baseline=(), sample_label='score value', ttl='',
        autolim=True
):
    """
    tests : sequence of test labels
    samples : sequence of samples (of some score or another)
    baseline : sequence of baseline scores
    sample_label

    """
    import matplotlib.pyplot as pp
    f = pp.figure()
    ax = f.add_subplot(111)
    ax.boxplot(samples, widths=.25)

    for n in range(len(samples)):
        pts = samples[n]
        label = '_nolegend_'
        if n == 0:
            label = 'site score'
        pp.plot(
            np.ones_like(pts) * (n + 1.26), pts, 'k+', label=label
        )
        if len(baseline):
            if n == 0:
                label = 'median bkgrnd score'
            bln = baseline[n]
            pp.plot(
                [(n + 1) - 0.125 - 0.3, (n + 1) + 0.125 - 0.3], [bln, bln],
                'g-', lw=2, label=label
            )

    samples = np.concatenate([s.ravel() for s in samples])
    if autolim:
        mx = np.diff(np.percentile(samples, [25, 75])) * 5 + \
             np.median(samples)
        pp.ylim(0, mx)
    pp.legend()
    pp.xticks(np.arange(1, len(tests) + 1), tests, rotation=90, fontsize=8)
    f.subplots_adjust(bottom=.35, top=.9)
    ax.set_title(ttl)
    ax.set_ylabel(sample_label)
    return f


def blended_image(
        fields, colors, afields=None, mode='darken',
        clip_min=2, clip_max=98
):
    colors = sns.color_palette(colors, n_colors=len(fields))
    mode = mode.lower()
    blend_to = 'white' if mode == 'darken' else 'black'
    N = mpl.cm.jet.N
    cmaps = [sns.blend_palette([blend_to, c], n_colors=N, as_cmap=True)
             for c in colors]

    values = np.asarray(fields).ravel()
    clim = np.percentile(values[np.isfinite(values)], [clip_min, clip_max])
    if afields is None:
        afields = [None] * len(fields)
    rgba_fields = np.array([rgba_field(cm, f, afield=a, clim=clim)[0]
                            for cm, f, a in zip(cmaps, fields, afields)])

    if mode == 'darken':
        return rgba_fields.min(axis=0), rgba_fields
    else:
        return rgba_fields.max(axis=0), rgba_fields


def normalize_figure_edges(figs, edges=('left', 'top')):
    try:
        for f in figs:
            f.canvas.draw()
    except:
        # try the following anyway
        pass

    for edge in edges:
        if edge in ('left', 'bottom'):
            norm_fn = max
        else:
            norm_fn = min
        this_edge = norm_fn([getattr(f.subplotpars, edge) for f in figs])
        for f in figs:
            f.subplots_adjust(**{edge: this_edge})
    return


def light_spines(ax, lw):
    """Set axes spines to given linewidth"""

    for spine in ax.spines.values():
        spine.set_lw(lw)
    ax.tick_params(axis='both', width=lw)


def grid_lims(ax, axis='both'):
    # assumes that the axes limits are currently aligned with major or
    # minor ticks where there should be grid lines. In some cases, the
    # axes bounding box cuts off the last grid line because it has a
    # non-zero width. Find out the width of the grid lines in terms of
    # data coordinates, and then boost the limits by this increment.
    m = ax.transData.get_matrix()
    dx = m[0, 0];
    dy = m[1, 1]
    lw = mpl.rcParams['grid.linewidth']
    # don't know how to do log-scale yet
    if axis in ('both', 'x') and ax.xaxis.get_scale() == 'linear':
        lx = lw / dx
        xl = ax.get_xlim()
        ax.set_xlim(xl[0] - lx, xl[1] + lx)
    if axis in ('both', 'y') and ax.yaxis.get_scale() == 'linear':
        ly = lw / dy
        yl = ax.get_ylim()
        ax.set_ylim(yl[0] - ly, yl[1] + ly)
    return


def waterfall(x, y, z, color='winter', rev_y=False, ax=None):
    """ Plot rows z(x) in z, staggered by levels in y """

    if ax is None:
        import matplotlib.pyplot as pp
        from mpl_toolkits.mplot3d import Axes3D
        fig = pp.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig = ax.figure

    lines = []
    for zx in z:
        lines.append(np.c_[x, zx])

    colors = mpl.cm.cmap_d[color](np.linspace(0, 1, len(y)))
    lines = LineCollection(
        lines, linewidth=2, colors=colors, zorder=10
    )

    ax.add_collection3d(lines, zs=y, zdir='y')
    ax.set_xlim3d(x[0], x[-1])
    ax.set_ylim3d(np.nanmin(y), np.nanmax(y))
    ax.set_zlim3d(np.nanmin(z), np.nanmax(z))
    xl = ax.get_xlim();
    yl = ax.get_ylim();
    zl = ax.get_zlim()
    ax.plot([xl[0], xl[0], xl[1], xl[1], xl[0]],
            [yl[0], yl[1], yl[1], yl[0], yl[0]],
            [zl[0], zl[0], zl[0], zl[0], zl[0]],
            zdir='z', color='k')
    # these lines draw the back plane
    # (which is at yl[0] or yl[1], depending on rev_y)
    y_pln = yl[0] if rev_y else yl[1]
    ax.plot([xl[0], xl[1]],
            [y_pln, y_pln],
            [zl[1], zl[1]], color='k')
    ax.plot([xl[0], xl[0]],
            [y_pln, y_pln],
            [zl[0], zl[1]], color='k')
    # ax.plot( [xl[0], xl[0]], [yl[1], yl[1]], [zl[0], zl[1]], color='k' )
    ax.plot([xl[1], xl[1]],
            [y_pln, y_pln],
            [zl[0], zl[1]], color='k')
    # ax.plot( [xl[1], xl[1]], [yl[1], yl[1]], [zl[0], zl[1]], color='k' )
    if rev_y:
        ax.set_ylim(ax.get_ylim()[::-1])
    return fig


def subplot2grid(fig, shape, loc, rowspan=1, colspan=1, **kwargs):
    """
    Create a subplot in a grid.  The grid is specified by *shape*, at
    location of *loc*, spanning *rowspan*, *colspan* cells in each
    direction.  The index for loc is 0-based. ::

      subplot2grid(shape, loc, rowspan=1, colspan=1)

    is identical to ::

      gridspec=GridSpec(shape[0], shape[1])
      subplotspec=gridspec.new_subplotspec(loc, rowspan, colspan)
      subplot(subplotspec)
    """
    from matplotlib.gridspec import GridSpec
    s1, s2 = shape
    subplotspec = GridSpec(s1, s2).new_subplotspec(loc,
                                                   rowspan=rowspan,
                                                   colspan=colspan)
    a = fig.add_subplot(subplotspec, **kwargs)
    bbox = a.bbox
    byebye = []
    for other in fig.axes:
        if other == a:
            continue
        if bbox.fully_overlaps(other.bbox):
            byebye.append(other)
    for ax in byebye:
        fig.delaxes(ax)

    return a


def desaturated_map(values, desat, colormap, drange=(), labels=(), title='', fig=None):
    if not fig:
        import matplotlib.pyplot as pp
        # set up figure then
        fig = pp.figure(figsize=(5, 3.5))

    if not fig.axes:
        arr_ax = subplot2grid((1, 100), (0, 0), colspan=80)
        cbar_ax = subplot2grid((1, 100), (0, 80), colspan=20)
        ## arr_ax = fig.add_axes( [0.05, 0.05, .75, .75] )
        ## cbar_ax = fig.add_axes( [0.8, 0.05, .125, .75] )
    else:
        arr_ax = fig.axes[0]
        ## arr_ax.images = []
        cbar_ax = fig.axes[1]
        ## cbar_ax.images = []
        ## fig.texts = []

    colors = colormap(mpl.colors.Normalize(*drange)(values), bytes=True)
    alpha = np.round(desat * 255).astype(colors.dtype)
    colors[..., -1] = alpha

    arr_ax.imshow(colors, origin='upper')
    arr_ax.axis('image')

    cbar_img = colormap(
        np.tile(np.linspace(0, 1, 100), (20, 1)).T, bytes=True
    )
    if np.iterable(desat):
        saturation = 255 * np.linspace(1, 0, 20, endpoint=False) ** 2
        cbar_img[..., -1] = saturation.astype('B')
    else:
        # otherwise make it single valued
        cbar_img[..., -1] = int(desat * 255)
    # this tested at a ratio of 1:12
    cbar_ax.imshow(
        cbar_img, extent=[0, 1 / 12., 0, 1], origin='lower'
    )
    cbar_ax.xaxis.set_visible(False)
    cbar_ax.yaxis.tick_right()
    if len(labels):
        cbar_ax.set_yticks(np.linspace(0, 1, len(labels)))
        cbar_ax.set_yticklabels(labels, fontsize=10)
    if title:
        fig.text(0.5, 0.9, title, ha='center', va='center')

    return fig


def colorbar_in_axes(ax, cmap, clim, **kwargs):
    # TODO fill in this method
    if ax is None:
        import matplotlib.pyplot as pp
        f, ax = pp.subplots(figsize=(5, 1))
    else:
        f = ax.figure
    cb = mpl.colorbar.ColorbarBase(ax, cmap='gray', orientation='horizontal')
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['neg', 'pos'])
    cb.set_label('Normalized Amplitude')
    f.tight_layout(pad=0.1)
    return f


def label_multi_axes(
        fig, text_size, bottom='', left='', top='', right='',
        lpad=1.5, bpad=2, text_margin=0.25
):
    """Scoot figure axes over to accomodate a single y- and/or x-label

    Should be called after tight_layout.
    Can also do top- and right-labels.

    """

    if not (bottom or left or top or right):
        return

    def tex_filter(s):
        import re
        if s.startswith('$'):
            m = re.search(r'[0-9]+\^\{[-+]?[0-9]+\}', s)
            m = m.group()
            if not m:
                return s
            return m.replace('{', '').replace('}', '').replace('^', '')
        return s

    if bottom:
        b_ax = [x for x in fig.axes if x.is_last_row()]
        if not len(b_ax):
            b_text = 0
        ax = b_ax[0]
        b_text = [len(ax.get_xticklabels()) > 0 for ax in b_ax]
        b_text = np.any(b_text)
    if left:
        l_ax = [x for x in fig.axes if x.is_first_col()]
        if not len(l_ax):
            l_text = 0
        l_text = []
        for ax in l_ax:
            l_text.extend([len(tex_filter(t.get_text()))
                           for t in ax.get_yticklabels()])
        # l_text = np.any(l_text)
        t = ax.get_yticklabels()[0]
        l_text = max(l_text) * t.get_fontsize()
    if top:
        t_ax = [x for x in fig.axes if x.is_first_row()]
        if not len(t_ax):
            t_text = 0
        t_text = [len(ax.get_title()) > 0 for ax in t_ax]
        t_text = np.any(t_text)
    if right:
        r_ax = [x for x in fig.axes if x.is_last_col()]
        if not len(r_ax):
            r_text = 0
        r_text = []
        for ax in r_ax:
            if ax.yaxis.get_ticks_position() == 'left':
                continue
            r_text.extend([len(tex_filter(t.get_text()))
                           for t in ax.get_yticklabels()])
        if not len(r_text):
            r_text = 0
        else:
            r_text = max(r_text) * t.get_fontsize()

    fw, fh = fig.transFigure.get_matrix().diagonal()[:2]
    # account for 2 fontsizes for x label, 2 fontsize padding
    lm = (lpad * text_size + l_text * 0.75) / fw if left else 0
    rm = (lpad * text_size + r_text * 0.75) / fw if right else 0
    bm = (text_size + bpad * int(b_text) * text_size) / fh if bottom else 0
    tm = (text_size + bpad * int(t_text) * text_size) / fh if top else 0
    if left:
        fig.subplots_adjust(left=lm)
    if right:
        fig.subplots_adjust(right=1 - rm)
    if bottom:
        fig.subplots_adjust(bottom=bm)
    if top:
        fig.subplots_adjust(top=1 - tm)
    md_y = 0.5 * (fig.subplotpars.bottom + fig.subplotpars.top)
    md_x = 0.5 * (fig.subplotpars.left + fig.subplotpars.right)
    # put a partial-text-size margin
    txm_x = text_margin * text_size / fw
    txm_y = text_margin * text_size / fh
    if left:
        fig.text(txm_x, md_y, left, va='center',
                 ha='left', rotation='vertical',
                 fontsize=text_size)
    if right:
        # fig.subplots_adjust(right=1-lm)
        fig.text(1 - txm_x, md_y, right, va='center',
                 ha='right', rotation=-90,
                 fontsize=text_size)
    if bottom:
        # fig.subplots_adjust(bottom=bm)
        fig.text(md_x, txm_y, bottom, va='baseline',
                 ha='center', fontsize=text_size)
    if top:
        # fig.subplots_adjust(top=1-bm)
        fig.text(md_x, 1 - txm_y, top, va='top',
                 ha='center', fontsize=text_size)

