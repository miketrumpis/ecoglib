import numpy as np

from ecogdata.util import get_default_args
from ecogdata.util import fenced_out
from ecogdata.devices.units import nice_unit_text

from ecoglib.vis.plot_util import filled_interval, light_boxplot
from ecoglib.vis.colormaps import nancmap
from ecoglib.estimation.spatial_variance import covar_to_iqr_lines, matern_semivariogram, make_matern_label, \
    plot_electrode_graph
from .signal_tools import bad_channel_mask, band_power, block_psds, logged_estimators, safe_avg_power, safe_corrcoef,\
    spatial_autocovariance

import seaborn as sns
sns.reset_orig()


__all__ = ['plot_psds', 'plot_electrode_graph', 'plot_avg_psds', 'plot_centered_rxx', 'plot_channel_mask',
           'plot_mean_psd', 'plot_mux_columns', 'plot_rms_array', 'plot_site_corr', 'plot_site_corr_new',
           'spatial_variance']


psd_colors = ["#348ABD", "#A60628"]


def plot_psds(f, gf, df, fc, title, ylims=(), root_hz=True, units='V', iqr_thresh=None):
    """Plot spectral power density estimate for each array channel
    (and possibly ground channels). Compute RMS power for the bandpass
    determined by "fc".

    Parameters
    ----------
    f : ndarray
        frequency vector
    gf : ndarray
        psd matrix for grounded input channels
    df : ndarray
        psd matrix for array signal channels
    fc : float
        cutoff frequency for RMS power calculation
    title : str
        plot title
    ylims : pair (optional)
        plot y-limits
    root_hz : (boolean)
        units normalized by 1/sqrt(Hz) (true) or 1/Hz (false)

    Returns
    -------
    figure

    """

    # compute outliers based on sum power
    if not iqr_thresh:
        iqr_thresh = get_default_args(fenced_out)['thresh']

    import matplotlib.pyplot as pp
    fig = pp.figure()
    fx = (f > 1) & (f < fc)
    # apply a wide-tolerance mask -- want to avoid plotting any
    # channels with zero (or negligable) power
    s_pwr = band_power(f, df, fc=fc, root_hz=root_hz)
    m = bad_channel_mask(np.log(s_pwr), iqr=iqr_thresh)
    df = df[m]
    pp.semilogy(
        f[fx], df[0, fx], color=psd_colors[0], label='sig channels'
    )
    pp.semilogy(
        f[fx], df[1:, fx].T, color=psd_colors[0], label='_nolegend_'
    )
    df_band_pwr = (df[:, fx] ** 2).mean()
    avg_d = np.sqrt(df_band_pwr * f[-1])
    pp.axhline(
        y=np.sqrt(df_band_pwr), color='chartreuse', linestyle='--',
        linewidth=4, label='sig avg RMS/$\sqrt{Hz}$'
    )

    if gf is not None and len(gf):
        pp.semilogy(f[fx], gf[0, fx], color=psd_colors[1], label='ground channels')
        if len(gf):
            pp.semilogy(f[fx], gf[1:, fx].T, color=psd_colors[1], label='_nolegend_')
        gf_band_pwr = (gf[:, fx] ** 2).mean()
        avg_g = np.sqrt(gf_band_pwr * f[-1])
        pp.axhline(
            y=np.sqrt(gf_band_pwr), color='k', linestyle='--', linewidth=4,
            label='gnd avg RMS/$\sqrt{Hz}$'
        )

    pp.legend(loc='upper right')
    units = nice_unit_text(units).strip('$')
    if root_hz:
        units_label = '$' + units + '/\sqrt{Hz}$'
    else:
        units_label = '$%s^{2}/Hz$' % units
    pp.ylabel(units_label);
    pp.xlabel('Hz (half-BW %d Hz)' % int(f[-1]))
    title = title + '\nSig RMS %1.2e' % avg_d
    if gf is not None:
        title = title + '; Gnd RMS %1.2e' % avg_g
    pp.title(title)
    pp.grid(which='both')
    if ylims:
        pp.ylim(ylims)
        offscreen = df[:, fx].mean(axis=1) < ylims[0]
        if np.any(offscreen):
            pp.gca().annotate(
                '%d chans off-screen' % offscreen.sum(),
                (200, ylims[0]), xycoords='data',
                xytext=(50, 3 * ylims[0]), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05)
            )
    return fig


def plot_mean_psd(f, gf, df, fc, title, ylims=(), root_hz=True, units='V', iqr_thresh=None):
    """Plot the mean spectral power density estimate for array
    channels (and possibly ground channels). Compute RMS power for the
    bandpass determined by "fc". Plot outlier PSDs individually.

    Parameters
    ----------
    f : sequence
        frequency vector
    gf : ndarray
        psd matrix for grounded input channels
    df : ndarray
        psd matrix for array signal channels
    fc : float
        cutoff frequency for RMS power calculation
    title : str
        plot title
    ylims : pair (optional)
        plot y-limits
    root_hz : (boolean)
        units normalized by 1/sqrt(Hz) (true) or 1/Hz (false)
    iqr_thresh : float (optional)
        set the outlier threshold (as a multiple of the interquartile
        range)

    Returns
    -------
    figure

    """

    import matplotlib.pyplot as pp
    # compute outliers based on sum power
    if not iqr_thresh:
        iqr_thresh = get_default_args(fenced_out)['thresh']

    s_pwr = band_power(f, df, fc=fc, root_hz=root_hz)
    s_pwr_mask = bad_channel_mask(np.log(s_pwr), iqr=iqr_thresh)
    ## s_pwr_mask = nut.fenced_out(np.log(s_pwr), thresh=iqr_thresh)
    ## s_pwr_mask = s_pwr_mask & (s_pwr > 0)
    s_pwr_mean = np.mean(s_pwr[s_pwr_mask])

    df = np.log(df)
    s_psd_mn = np.mean(df[s_pwr_mask], axis=0)
    s_psd_stdev = np.std(df[s_pwr_mask], axis=0)
    s_psd_lo = s_psd_mn - s_psd_stdev
    s_psd_hi = s_psd_mn + s_psd_stdev
    s_psd_mn, s_psd_lo, s_psd_hi = map(np.exp, (s_psd_mn, s_psd_lo, s_psd_hi))
    avg_d = np.sqrt(s_pwr[s_pwr_mask].mean())

    fig, ln = filled_interval(
        pp.semilogy, f, s_psd_mn, (s_psd_lo, s_psd_hi), psd_colors[0]
    )

    sig_baseline = s_psd_mn[f > f.max() / 2].mean()
    legends = [r'mean signal PSD $\pm \sigma$']
    df_o = None
    if np.any(~s_pwr_mask):
        df_o = np.exp(df[~s_pwr_mask])
        o_lines = pp.semilogy(f, df_o.T, '#BD6734', lw=0.5)
        ln.append(o_lines[0])
        legends.append('outlier signal PSDs')
        # let's label these lines
        chan_txt = 'outlier sig chans: ' + \
                   ', '.join([str(c) for c in (~s_pwr_mask).nonzero()[0]])
        y = 0.5 * (np.ceil(np.log(s_psd_mn.max())) + np.log(sig_baseline))
        pp.text(200, np.exp(y), chan_txt, fontsize=10, va='baseline')

    if gf is not None and len(gf):
        g_pwr = band_power(f, gf, fc=fc, root_hz=root_hz)
        if len(g_pwr) > 1:
            g_pwr_mask = fenced_out(np.log(g_pwr), thresh=iqr_thresh)
        else:
            g_pwr_mask = np.array([True])
        g_pwr_mean = np.mean(g_pwr[g_pwr_mask])

        gf = np.log(gf)
        g_psd_mn = np.mean(gf[g_pwr_mask], axis=0)
        g_psd_stdev = np.std(gf[g_pwr_mask], axis=0)
        g_psd_lo = g_psd_mn - g_psd_stdev
        g_psd_hi = g_psd_mn + g_psd_stdev
        g_psd_mn, g_psd_lo, g_psd_hi = map(np.exp, (g_psd_mn, g_psd_lo, g_psd_hi))
        avg_g = np.sqrt(g_pwr[g_pwr_mask].mean())

        fig, g_ln = filled_interval(
            pp.semilogy, f, g_psd_mn, (g_psd_lo, g_psd_hi), psd_colors[1],
            ax=fig.axes[0]
        )
        ln.extend(g_ln)
        legends.append(r'mean grounded input $\pm \sigma$')
        if np.any(~g_pwr_mask):
            o_lines = pp.semilogy(
                f, np.exp(gf[~g_pwr_mask]).T, '#06A684', lw=0.5
            )
            ln.append(o_lines[0])
            legends.append('outlier grounded PSDs')
            chan_txt = 'outlier gnd chans: ' + \
                       ', '.join([str(c) for c in (~g_pwr_mask).nonzero()[0]])
            y = sig_baseline ** 0.33 * g_psd_mn.mean() ** 0.67

            pp.text(200, y, chan_txt, fontsize=10, va='baseline')

    pp.legend(ln, legends, loc='upper right', fontsize=11)
    units = nice_unit_text(units).strip('$')
    if root_hz:
        units_label = '$' + units + '/\sqrt{Hz}$'
    else:
        units_label = '$%s^{2}/Hz$' % units
    pp.ylabel(units_label);
    pp.xlabel('Hz (half-BW %d Hz)' % int(f[-1]))
    if gf is not None and len(gf):
        title = title + \
                '\nGnd RMS %1.2e; Sig RMS %1.2e (to %d Hz)' % (avg_g, avg_d, fc)
    else:
        title = title + \
                '\nSig RMS %1.2e (to %d Hz)' % (avg_d, fc)
    pp.title(title)
    pp.grid(which='both')
    if ylims:
        pp.ylim(ylims)
        if df_o is not None:
            offscreen = df_o.mean(axis=1) < ylims[0]
            if np.any(offscreen):
                pp.gca().annotate(
                    '%d chans off-screen' % offscreen.sum(),
                    (200, ylims[0]), xycoords='data',
                    xytext=(50, 3 * ylims[0]), textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05)
                )

    return fig


def plot_avg_psds(data, d_chans, g_chans, title, bsize_sec=2, Fs=1, iqr_thresh=None, units='V', **mtm_kw):
    # make two plots with
    # 1) all spectra
    # 2) average +/- sigma, and outliers

    freqs, psds = block_psds(data + 1e-10, bsize_sec, Fs, **mtm_kw)
    psds, p_mn, p_err_lo, p_err_hi = logged_estimators(psds, sem=False)
    g_psds = np.sqrt(psds[g_chans]) if len(g_chans) else None
    d_psds = np.sqrt(psds[d_chans])
    ## g_psds = psds[g_chans] if len(g_chans) else None
    ## d_psds = psds[d_chans]

    ttl_str = '%s Fs=%d' % (title, round(Fs))
    ymax = 10 ** np.ceil(np.log10(d_psds.max()) + 1)
    ymin = ymax * 1e-6
    fig = plot_psds(
        freqs, g_psds, d_psds, Fs / 2, ttl_str,
        ylims=(ymin, ymax),
        iqr_thresh=iqr_thresh, units=units, root_hz=True
    )
    fig_avg = plot_mean_psd(
        freqs, g_psds, d_psds, Fs / 2, ttl_str,
        ylims=(ymin, ymax),
        iqr_thresh=iqr_thresh, units=units, root_hz=True
    )

    return fig, fig_avg


def plot_centered_rxx(data, d_chans, chan_map, label, pitch=1.0, cmap='bwr', normed=True, clim=None):
    import matplotlib.pyplot as pp
    from seaborn import JointGrid

    cxx = safe_corrcoef(data[d_chans], 2000, normed=normed)
    n = cxx.shape[0]

    pitch = chan_map.pitch
    if np.iterable(pitch):
        pitch_x, pitch_y = pitch
    else:
        pitch_x = pitch_y = pitch

    centered_rxx = spatial_autocovariance(cxx, chan_map, mean=False)
    y, x = centered_rxx.shape[-2:]
    midx = int(x / 2)
    xx = (np.arange(x) - midx) * pitch_x
    midy = int(y / 2)
    yy = (np.arange(y) - midy) * pitch_y
    # centered_rxx[:,midy,midx] = 1
    with sns.axes_style('ticks'):

        jgrid = JointGrid(
            np.random.rand(50), np.random.rand(50), ratio=4,
            xlim=(xx[0], xx[-1]), ylim=(yy[0], yy[-1]), size=8
        )

        cm = nancmap(cmap, nanc=(.5, .5, .5, .5))

        ## Joint plot
        ax = jgrid.ax_joint
        if clim is None:
            clim = (-1, 1) if normed else np.percentile(centered_rxx, [2, 98])
        ax.imshow(
            np.nanmean(centered_rxx, axis=0), clim=clim, cmap=cm,
            extent=[xx[0], xx[-1], yy[0], yy[-1]]
        )
        ax.set_xlabel('Site-site distance (mm)')
        ax.set_ylabel('Site-site distance (mm)')

        ## Marginal-X
        ax = jgrid.ax_marg_x
        ax.spines['left'].set_visible(True)
        ax.yaxis.tick_left()
        pp.setp(ax.yaxis.get_majorticklines(), visible=True)
        pp.setp(ax.get_yticklabels(), visible=True)
        # arrange as samples over all x-distances
        rxx_mx = np.reshape(centered_rxx, (-1, x))

        vals = list()
        for c in rxx_mx.T:
            valid = ~np.isnan(c)
            if valid.any():
                vals.append(np.percentile(c[valid], [25, 50, 75]))
            else:
                vals.append([np.nan] * 3)

        mx_lo, mx_md, mx_hi = map(np.array, zip(*vals))
        filled_interval(
            ax.plot, xx, mx_md, (mx_lo, mx_hi), cm(0.6), ax=ax, lw=2, alpha=.6
        )
        ax.set_yticks(np.linspace(-1, 1, 6))
        ax.set_ylim(clim)

        ## Marginal-Y
        ax = jgrid.ax_marg_y
        ax.spines['top'].set_visible(True)
        ax.xaxis.tick_top()
        pp.setp(ax.xaxis.get_majorticklines(), visible=True)
        pp.setp(ax.get_xticklabels(), visible=True)
        rxx_my = np.reshape(np.rollaxis(centered_rxx, 2).copy(), (-1, y))
        vals = list()
        for c in rxx_my.T:
            valid = ~np.isnan(c)
            if valid.any():
                vals.append(np.percentile(c[valid], [25, 50, 75]))
            else:
                vals.append([np.nan] * 3)

        my_lo, my_md, my_hi = map(np.array, zip(*vals))
        filled_interval(
            ax.plot, yy, my_md, (my_lo, my_hi), cm(0.6),
            ax=ax, lw=2, alpha=.6, fillx=True
        )
        ax.set_xticks(np.linspace(-1, 1, 6))
        pp.setp(ax.xaxis.get_ticklabels(), rotation=-90)
        ax.set_xlim(clim)

        jgrid.fig.subplots_adjust(left=0.1, bottom=.1)

    jgrid.ax_marg_x.set_title(
        'Average centered correlation map: ' + label, fontsize=12
    )
    return jgrid.fig


def spatial_variance(data, chan_map, label, normed=False):
    import matplotlib.pyplot as pp
    from seaborn import despine, xkcd_rgb

    cxx = safe_corrcoef(data, 2000, normed=normed, semivar=True)
    n = cxx.shape[0]
    cxx_pairs = cxx[np.triu_indices(n, k=1)]
    rms = safe_avg_power(data)
    var_mu = (rms ** 2).mean()
    var_se = (rms ** 2).std() / np.sqrt(len(rms))

    chan_combs = chan_map.site_combinations
    dist = chan_combs.dist
    if np.iterable(chan_map.pitch):
        pitch_x, pitch_y = chan_map.pitch
    else:
        pitch_x = pitch_y = chan_map.pitch
    binsize = np.ceil(10 * (pitch_x ** 2 + pitch_y ** 2) ** 0.5) / 10.0
    clrs = pp.rcParams['axes.prop_cycle'].by_key()['color']
    pts, lines = covar_to_iqr_lines(dist, cxx_pairs, binsize=binsize, linewidths=1, colors=clrs[0])
    xb, yb = pts
    # set a fairly wide range for nugget and sill
    bounds = {'nugget': (0, yb[0]), 'sill': (np.mean(yb), var_mu + 5 * var_se),
              'nu': (0.4, 10), 'theta': (0.5, None)}
    p = matern_semivariogram(
        dist, y=cxx_pairs, theta=1, nu=1, sill=var_mu, nugget=yb[0] / 5.0,
        free=('theta', 'nu', 'nugget', 'sill'), dist_limit=0.67,
        wls_mode='irls', fit_mean=True, fraction_nugget=False, bounds=bounds)

    f, ax = pp.subplots(figsize=(8, 5))
    ax.scatter(dist, cxx_pairs, s=5, color='gray', alpha=0.2, rasterized=True, label='Pairwise semivariance')
    ax.plot(*pts, color=clrs[0], ls='--', marker='o', ms=8, label='Binned semivariance')
    ax.add_collection(lines)
    ax.axhline(var_mu, lw=1, color=xkcd_rgb['reddish orange'], label='Avg signal variance', alpha=0.5)
    ax.axhline(var_mu + var_se, lw=0.5, color=xkcd_rgb['reddish orange'], linestyle='--', alpha=0.5)
    ax.axhline(var_mu - var_se, lw=0.5, color=xkcd_rgb['reddish orange'], linestyle='--', alpha=0.5)
    ax.axhline(p['nugget'], lw=2, color=xkcd_rgb['dark lavender'], alpha=0.5, label='Noise "nugget" (uV^2)')
    ax.axhline(p['sill'], lw=2, color=xkcd_rgb['teal green'], alpha=0.5, label='Spatial var. "sill" (uV^2)')
    xm = np.linspace(dist.min(), dist.max(), 100)
    model_label = 'Model: ' + make_matern_label(theta=p['theta'], nu=p['nu'])
    ax.plot(xm, matern_semivariogram(xm, **p), color=clrs[1], label=model_label)
    ax.set_xlabel('Site-site distance (mm)')
    if normed:
        units = '(normalized)'
    else:
        units = '(uV^2)'
    ax.set_ylabel('Semivariance ' + units)
    despine(fig=f)
    leg = ax.legend(loc='upper left', ncol=3, frameon=True)
    for h in leg.legendHandles:
        h.set_alpha(1)
        try:
            h.set_sizes([15] * len(h.get_sizes()))
        except:
            pass
    ax.set_title(label + ' spatial variogram')
    f.tight_layout(pad=0.2)
    return f


def scatter_correlations(data, d_chans, chan_map, mask, title, highlight='rows', pitch=1.0):

    # plot the pairwise correlation values against distance of the pair
    # Highlight channels that
    # 1) share a row (highlight='rows')
    # 2) share a column (highlight='cols')
    # 3) either of the above (highlight='rows+cols')
    # 4) are neighbors on a row (highlight='rownabes')
    # 5) are neighbors on a column (highlight='colnabes')
    # 6) any neighbor (4-5) (highlight='allnabes')
    import matplotlib.pyplot as pp

    # data[g_chans] = np.nan
    cxx = safe_corrcoef(data[d_chans[mask]], 2000)
    n = cxx.shape[0]

    cxx_pairs = cxx[np.triu_indices(n, k=1)]
    if np.iterable(pitch):
        pitch_x, pitch_y = pitch
    else:
        pitch_x = pitch_y = pitch
    chan_combs = chan_map.subset(mask).site_combinations
    dists = chan_combs.dist

    fig = pp.figure()

    panels = highlight.split(',')
    if panels[0] == highlight:
        pp.subplot(111)
        pp.scatter(
            dists, cxx_pairs, 9, label='_nolegend_', edgecolors='none', alpha=0.25, rasterized=True
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.text(0.5, .96, title, fontsize=16, va='baseline', ha='center')
        return fig

    # xxx: hardwired for 16 channel muxing with grounded input on 1st chan
    mux_index = np.arange(len(d_chans)).reshape(-1, 15).transpose()[1:]
    nrow, ncol = mux_index.shape
    mux_index -= np.arange(1, ncol + 1)

    colors = dict(rows='#E480DA', cols='#80E48A')
    cxx = safe_corrcoef(data[d_chans], 2000)
    cxx_pairs = cxx[np.triu_indices(len(cxx), k=1)]
    chan_combs = chan_map.site_combinations

    dists = chan_combs.dist

    for n, highlight in enumerate(panels):
        pp.subplot(len(panels), 1, n + 1)
        pp.scatter(
            dists, cxx_pairs, 9, edgecolor='none', label='_nolegend_', alpha=0.25, rasterized=True
        )
        if highlight in ('rows', 'rows+cols'):
            for row in mux_index:
                row = [r for r in row if mask[r]]
                if len(row) < 2:
                    continue
                subset = chan_map.subset(row)
                subcxx = cxx[row][:, row][np.triu_indices(len(row), k=1)]
                subdist = subset.site_combinations.dist
                c = pp.scatter(
                    subdist, subcxx, 20, colors['rows'],
                    edgecolor='white', label='_nolegend_'
                )
            # set label on last one
            c.set_label('row combo')
        if highlight in ('cols', 'rows+cols'):
            for col in mux_index.T:
                col = [c for c in col if mask[c]]
                if len(col) < 2:
                    continue
                subset = chan_map.subset(col)
                subcxx = cxx[col][:, col][np.triu_indices(len(col), k=1)]
                subdist = subset.site_combinations
                c = pp.scatter(
                    subdist, subcxx, 20, colors['cols'],
                    edgecolor='white', label='_nolegend_'
                )
            # set label on last one
            c.set_label('col combo')

        if highlight in ('rownabes', 'allnabes'):
            row_cxx = list()
            row_dist = list()
            for row in mux_index:
                row = [r for r in row if mask[r]]
                if len(row) < 2:
                    continue
                for i1, i2 in zip(row[:-1], row[1:]):
                    ii = np.where(
                        (chan_combs.p1 == min(i1, i2)) & \
                        (chan_combs.p2 == max(i1, i2))
                    )[0][0]
                    row_cxx.append(cxx_pairs[ii])
                    row_dist.append(dists[ii])
            c = pp.scatter(
                row_dist, row_cxx, 20, colors['rows'],
                edgecolor='white', label='row neighbors'
            )
        if highlight in ('colnabes', 'allnabes'):
            col_cxx = list()
            col_dist = list()
            for col in mux_index.T:
                col = [c for c in col if mask[c]]
                if len(col) < 2:
                    continue
                for i1, i2 in zip(col[:-1], col[1:]):
                    ii = np.where(
                        (chan_combs.p1 == min(i1, i2)) & \
                        (chan_combs.p2 == max(i1, i2))
                    )[0][0]
                    col_cxx.append(cxx_pairs[ii])
                    col_dist.append(dists[ii])
            c = pp.scatter(
                col_dist, col_cxx, 20, colors['cols'],
                edgecolor='white', label='col neighbors'
            )
        pp.legend(loc='best')

    ax = pp.gca()
    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Correlation coef.')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.text(0.5, .96, title, fontsize=16, va='baseline', ha='center')

    return fig


def plot_mux_columns(data, d_chans, g_chans, title, color_lims=True, units='uV'):
    import matplotlib.pyplot as pp

    # data[g_chans] = np.nan
    rms = safe_avg_power(data, 2000)
    rms[g_chans] = np.nan
    if color_lims:
        vals = rms[np.isfinite(rms)]
        # basically try to clip out anything small
        vals = vals[vals > 1e-2 * np.median(vals)]
        quantiles = np.percentile(vals, [5., 95.])
        clim = tuple(quantiles)
    else:
        clim = (np.nanmin(rms), np.nanmax(rms))

    d_rms = rms[d_chans].reshape(-1, 15)
    if len(g_chans):
        rms = np.column_stack((rms[g_chans], d_rms))
    else:
        rms = d_rms
    # rms.shape = (-1, 16)
    fig = pp.figure()
    cm = nancmap('hot', nanc='dodgerblue')
    pp.imshow(rms.T, origin='upper', cmap=cm, clim=clim)
    cbar = pp.colorbar()
    cbar.set_label(nice_unit_text(units) + ' RMS')
    pp.title(title)
    pp.xlabel('data column')
    ax = fig.axes[0]
    ax.set_aspect('auto')
    ax.set_xticks(range(rms.shape[0]))
    return fig


def plot_rms_array(data, d_chans, chan_map, title, color_lims=True, units='uV'):
    import matplotlib.pyplot as pp
    # data[g_chans] = np.nan
    rms = safe_avg_power(data, 2000)
    rms = rms[d_chans]
    if color_lims:
        vals = rms[np.isfinite(rms)]
        # basically try to clip out anything small
        vals = vals[vals > 1e-2 * np.median(vals)]
        quantiles = np.percentile(vals, [5., 95.])
        clim = tuple(quantiles)
    else:
        clim = (np.nanmin(rms), np.nanmax(rms))
    rms_arr = chan_map.embed(rms)
    # rms_arr = np.ones(chan_map.geometry)*np.nan
    # np.put(rms_arr, chan_map, rms)
    cm = nancmap('hot', nanc='dodgerblue')

    f = pp.figure()
    pp.imshow(rms_arr, origin='upper', cmap=cm, clim=clim)
    cbar = pp.colorbar()
    cbar.set_label(nice_unit_text(units) + ' RMS')

    pp.title(title)
    return f


def plot_site_corr(data, d_chans, title):
    import matplotlib.pyplot as pp
    # data[g_chans] = np.nan
    cxx = safe_corrcoef(data[d_chans], 2000)
    n = cxx.shape[0]
    cxx.flat[0:n * n:n + 1] = np.nan

    cm = pp.cm.jet

    f = pp.figure()
    pp.imshow(cxx, cmap=cm)
    cbar = pp.colorbar()
    cbar.set_label('avg corr coef')

    pp.title(title)
    return f


def plot_site_corr_new(data, chan_map, title, bsize=2000, cmap=None, normed=True, stagger_x=False, stagger_y=False):
    import matplotlib.pyplot as pp
    # data[g_chans] = np.nan
    cxx = safe_corrcoef(data, bsize, normed=normed)
    n = cxx.shape[0]
    cxx.flat[0:n * n:n + 1] = np.nan

    clim = (-1, 1) if normed else np.percentile(cxx, [2, 98])
    if cmap is None:
        import ecogana.anacode.colormaps as cmaps
        cmap = cmaps.diverging_cm(clim[0], clim[1], ((0, 0, 0), (1, 0, 0)))

    f, axs = pp.subplots(1, 2, figsize=(12, 5))

    corr_ax = axs[0]
    graph_ax = axs[1]

    im = corr_ax.imshow(cxx, cmap=cmap, norm=pp.Normalize(*clim))
    cbar = pp.colorbar(im, ax=corr_ax, use_gridspec=True)
    cbar.set_label('avg corr coef')
    corr_ax.axis('image')

    plot_electrode_graph(cxx, chan_map, ax=graph_ax, stagger_y=stagger_y, stagger_x=stagger_x)

    f.subplots_adjust(top=0.9, left=0.05, right=0.95, wspace=0.1)
    f.text(0.5, 0.92, title, ha='center', va='baseline', fontsize=20)
    return f


def plot_channel_mask(data, chan_map, title, units='V', bsize=2000, quantiles=(50, 80), iqr=3):
    import matplotlib.pyplot as pp
    from seaborn import violinplot, xkcd_rgb
    rms = safe_avg_power(data, bsize=bsize, iqr_thresh=7)
    rms = np.log(rms)
    mask = bad_channel_mask(rms, quantiles=quantiles, iqr=iqr)
    f = pp.figure(figsize=(7, 4))
    ax = f.add_subplot(121)
    with sns.axes_style('whitegrid'):
        violinplot(
            np.ma.masked_invalid(rms).compressed(),
            alpha=0.5, widths=0.5, names=[' '],
            color=xkcd_rgb['amber'], orient='v'
        )
        sns.despine(ax=ax, left=True)
        ax.plot(np.ones(mask.sum()) * 1.3, rms[mask], 'k+')
        if np.sum(~mask):
            ax.plot(np.ones(np.sum(~mask)) * 1.3, rms[~mask], 'r+')
        ax.set_yticklabels(['%.1f' % s for s in np.exp(ax.get_yticks())])
        ax.set_ylabel(nice_unit_text(units) + ' RMS')
    ax.set_title('Distribution of log-power')
    ax = f.add_subplot(122)
    site_mask = np.ones(chan_map.geometry) * np.nan
    site_mask.flat[chan_map.subset(mask.nonzero()[0])] = 1
    site_mask.flat[chan_map.subset((~mask).nonzero()[0])] = 0
    N = pp.cm.binary.N
    im = ax.imshow(
        site_mask,
        cmap=pp.cm.winter, norm=pp.cm.colors.BoundaryNorm([0, .5, 1], N),
        alpha=0.5, origin='upper'
    )
    cbar = pp.colorbar(im)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(('rejected', 'accepted'))
    ax.axis('image')
    ax.set_title('Inlier electrodes')
    f.text(0.5, 0.02, title, ha='center', va='baseline', fontsize=18)
    return f, mask


def sinusoid_gain(data, ref, chan_map, log=True, **im_kws):
    import matplotlib.pyplot as pp
    ## d_rms = data.std(1)
    ## r_rms = ref.std()
    ## gain = d_rms / r_rms
    data = data - data.mean(axis=-1, keepdims=1)
    ref = ref - ref.mean()
    gain = np.dot(data, ref) / np.dot(ref, ref)

    f = pp.figure(figsize=(7.5, 4))
    ax = pp.subplot2grid((1, 100), (0, 0), colspan=25)

    light_boxplot(
        np.log10(gain) if log else gain, names=[''],
        mark_mean=True, box_ls='solid', ax=ax
    )
    ax.set_ylabel('log10 gain' if log else 'gain')

    ax = pp.subplot2grid((1, 100), (0, 25), colspan=75)
    _, cbar = chan_map.image(gain, ax=ax, **im_kws)
    cbar.set_label('array gain')
    return f


