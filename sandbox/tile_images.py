from __future__ import division
import numpy as np
import matplotlib.pyplot as pp

from ecoglib.util import ChannelMap, flat_to_mat, mat_to_flat
from ecoglib.numutil import ndim_prctile

from ecogana.devices import units

def _build_map(p, geometry, col_major):
    if isinstance(p, ChannelMap):
        return p
    return ChannelMap(p, geometry, col_major=col_major)

def quick_tiles(n_frames, nrow=None, ncol=None, **kws):
    if not (nrow or ncol):
        ncol = 4

    if nrow and not ncol:
        ncol = int( np.ceil( float(n_frames) / nrow ) )
    elif ncol and not nrow:
        nrow = int( np.ceil( float(n_frames) / ncol ) )
    if ncol * nrow < n_frames:
        raise ValueError('Not enough tiles for frames')

    fig, axes, _ = tiled_axes( 
        (nrow, ncol), np.arange(n_frames),
        col_major=False, **kws
        )
    return fig, axes

def tiled_axes(
        geo, p, tilesize=(1,1), col_major=True, 
        title='', calib='none'
    ):
    """
    Creates a tiling of axes in a grid with given geometry.

    Parameters
    ----------

    geo : (nrow, ncol)
        Dimension of axes grid

    p : sequence, ChannelMap
        Flat index into the grid. Subplot axes are returned corresonding
        to this order

    Returns
    -------

    fig, plot_axes, missed_axes
        The i) Figure, ii) sequence of axes in the order specified by "p",
        and iii) any grid positions not specified by "p"

    """

    p = _build_map(p, geo, col_major).as_row_major()
    assert p.geometry == geo, 'Provided channel map has different geometry'
    n_plots = len(p)
    missed_tiles = set(range(geo[0]*geo[1]))
    missed_tiles.difference_update(p)
    figsize = [geo[1]*tilesize[0], geo[0]*tilesize[1]]

    subplots_bottom = subplots_left = 0.02
    subplots_top = subplots_right = 0.98
    if title:
        figsize[1] += 1
        subplots_top = 1 - 1.0/figsize[1]
    if calib.lower() != 'none':
        if calib in ('left', 'right', 'side'):
            figsize[0] += 1
            if calib == 'left':
                subplots_left = 1.0/figsize[0]
            if calib == 'right':
                subplots_right = 1 - 1.0/figsize[0]
        else:
            figsize[1] += 1
            subplots_bottom = 1.0/figsize[1]

    fig, axs = pp.subplots(
        *geo, figsize=figsize, sharex=True, sharey=True, squeeze=False
        )
    ## fig = pp.figure(figsize=figsize)
    fig.subplots_adjust(
        left=subplots_left, right=subplots_right,
        bottom=subplots_bottom, top=subplots_top
        )
    plot_axes = list()
    plot_axes = axs.ravel()[ p.as_row_major() ]
    missed_axes = axs.ravel()[ list(missed_tiles) ]
    for ax in missed_axes:
        ax.axis('off')
    ## ii, jj = p.to_mat()
    ## subplot_grid = pp.GridSpec(*geo)
    ## for i, j in zip(ii, jj):
    ##     ax_spec = subplot_grid.new_subplotspec( (i,j) )
    ##     ax = fig.add_subplot(ax_spec)
    ##     plot_axes.append(ax)
    ## missed_axes = list()
    ## for ax in missed_tiles:
    ##     (i, j) = flat_to_mat(geo, pn, col_major=p.col_major)
    ##     ax_spec = subplot_grid.new_subplotspec( (i,j) )
    ##     ax = fig.add_subplot(ax_spec)
    ##     ax.axis('off')
    ##     missed_axes.append(ax)

    if title:
        fig.text(
            0.5, .95, title, fontsize=18, 
            va='baseline', ha='center'
            )

    return fig, plot_axes, missed_axes

def calibration_axes(
        ref_ax, y_scale=None, t_scale=None, calib_ax=None, 
        calib_unit='V', time_units='ms', fontsize=11
        ):

    """
    Automated drawing of calibration bars in abscissa and ordinate.
    The strategy is to try to infer the appropriate scales from a
    referrence Axes object, and draw calibration bars either into
    the space already provided in the Figure, or in the Axes object
    provided in the parameters.

    Parameters
    ----------

    ref_ax : Axes
        An Axes object whose abscissa and ordinate are used to determine
        the appropriate scales

    y_scale : float, optional
        This parameter is used to fix the ordinate scale.

    t_scale : float, optional
        This parameter is used to fix the abscissa scale.

    calib_ax : Axes, optional
        Draw calibration bars into this object.

    calib_units : str
        The units of the ordinate, by default Volts.

    Returns
    -------

    calib_ax
    
    """
    
    
    #t_len = time[-1] - time[0]
    xlim = ref_ax.get_xlim()
    t_len = xlim[1] - xlim[0]
    if t_len < 20:
        t_len *= 1e3

    #time_units = 'ms'
    time_quantum = 50
    #print 'ref axis t-len:', t_len

    ylim = ref_ax.get_ylim()
    if not y_scale:
        y_scale = (ylim[1] - ylim[0]) / 2.0
    y_step, scaling, calib_unit = units.best_scaling_step(y_scale, calib_unit)
    y_scale *= scaling
    scale_back = 1/scaling
    calib_label = units.nice_unit_text(calib_unit)

    y_calib = np.floor( y_scale / y_step ) * y_step
    y_txt = r'%d %s'%(y_calib, calib_label)
    
    # find out how big our calib axes is before computing time scale
    pos = ref_ax.get_position()
    x0 = pos.x0; x1 = pos.x1; y0 = pos.y0; y1 = pos.y1

    if not calib_ax:
        xw = 1 - x1 - 0.02
        yw = y1 - y0
        calib_ax = ref_ax.figure.add_axes([x1+0.01, y0, xw, yw])
    else:
        c_pos = calib_ax.get_position()
        xw = c_pos.x1 - c_pos.x0
        
    # full plot width
    xw1 = x1 - x0
    # equivalently scaled sub-interval in new axes is this long
    sub_t_len = (float(xw)/float(xw1)) * t_len

    #print 'calib axis t-len:', sub_t_len
    if t_scale is None:
        t_calib = 0
        while t_calib == 0:
            t_calib = np.floor( sub_t_len / time_quantum ) * time_quantum
            if abs(t_calib - sub_t_len) < 1e-6 or t_calib > t_len:
                # if that's the full frame, 
                # of if that is longer than the reference axes,
                # then cut it in half
                t_calib = t_calib // 2
            # if that time length is too big, try half
            time_quantum = time_quantum // 2
    else:
        t_calib = t_scale
        if t_calib < 1:
            t_calib *= 1e3
        while t_calib > sub_t_len:
            t_calib = t_calib // 2
    
    t_txt = '%d %s'%(t_calib, time_units)

    ## if time_units == 'sec':
    ##     t_calib /= 1e3
    ##     sub_t_len /= 1e3

    calib_ax.set_ylim(ylim)
    calib_ax.set_xlim(-sub_t_len/2.0, sub_t_len/2.0)
    calib_ax.axis('off')
    try:
        pp.draw()
        # offset left by 5 pts
        dx = calib_ax.transData.inverted().get_matrix()[0,0]
        dy = calib_ax.transData.inverted().get_matrix()[1,1]
    except:
        # try to approximate by figure settings
        f = ref_ax.figure
        pos = calib_ax.get_position()
        fh = f.get_figheight() * f.dpi
        fw = f.get_figwidth() * f.dpi
        dy = np.diff(ylim)[0] / ( fh * (pos.y1-pos.y0) )
        dx = sub_t_len / ( fw * (pos.x1-pos.x0) )
    
    # repurpose y vars
    # start vertical bar 1/4 of the way from the bottom
    # y0 = (3/4.) * ylim[0] + (1/4.) * ylim[1]
    # y0 = 0.5 * (ylim[0] + ylim[1] - y_calib * scale_back)
    # start vertical bar a few fontscales from the bottom
    y0 = ylim[0] + 2 * fontsize * dy
    y1 = y0 + y_calib * scale_back
    ## if ylim[1] - ylim[0] < 1:
    ##     y1 = y0 + y_calib/1e6
    ## else:
    ##     y1 = y0 + y_calib

    # center the t-calibration bar
    t0 = -t_calib/2.0
    t1 = t_calib/2.0

    calib_kws = dict(color='k', linewidth=3, solid_capstyle='butt')

    calib_ax.plot([t0, t1], [y0, y0], **calib_kws)
    calib_ax.plot([t0, t0], [y0, y1], **calib_kws)

    calib_ax.text(
        t0 - 0.75*fontsize*dx, y0, y_txt, ha='center', va='bottom', 
        rotation='vertical', fontsize=fontsize
        )
    calib_ax.text(
        0, y0 - 0.75*fontsize*dy, t_txt, ha='center', va='top', 
        fontsize=fontsize
        )
    return calib_ax

         
def fill_null(ax_list, cmap=pp.cm.gray):
    for ax in ax_list:
        ax.imshow(
            np.array([ [.85, .75], [.75, .85] ]), cmap=cmap,
            clim=(0, 1), interpolation='nearest'
            )
        ax.axis('off')

def tile_images(
        maps, geo=(), p=(), col_major=True,
        border_axes=False, title='', 
        fill_empty=False, fill_cmap=pp.cm.gray, 
        x_labels=(), y_labels=(),
        clabel='none', **imkw
        ):

    # maps is (n_maps, mx, my), or (nx_tiles, ny_tiles, mx, my)
    # if ndim < 4, then geo must be provided

    # filter the cmap in case it's a literal table
    if imkw.has_key('cmap'):
        cm = imkw['cmap']
        if type(cm) == np.ndarray:
            imkw['cmap'] = pp.cm.colors.ListedColormap(cm)

    if not ( imkw.has_key('clim') or imkw.has_key('vmin') or \
             imkw.has_key('vmax') or imkw.has_key('norm') ):
        vmin = np.nanmin(maps); vmax = np.nanmax(maps)
        imkw['norm'] = pp.Normalize(vmin, vmax)

    p = _build_map(p, geo, col_major)        
    geo = p.geometry

    if not geo:
        raise RuntimeError(
            "Tiling geometry was not provided"
            )

    ## if maps.ndim == 4:
    ##     geo = maps.shape[:2]
    ## elif not geo:
    ##     raise RuntimeError(
    ##         "Can't infer geometry from array, nor was a geometry provided"
    ##         )
    ## oshape = maps.shape
    ## maps.shape = (-1,) + maps.shape[-2:]
    
    if not len(p):
        # no permutation
        p = _build_map(np.arange(maps.shape[0]), p.geometry, p.col_major)


    nx, ny = map(float, maps.shape[-2:])
    tilesize = (nx/min(nx,ny), ny/min(nx,ny)) # width, height
    calib = 'none' if clabel == 'none' else 'bottom'
    fig, plotted, missed = tiled_axes(
        geo, p, title=title, tilesize=tilesize, calib=calib
        )
    
    ii, jj = p.to_mat()
    for n in xrange(maps.shape[0]):
        # get the (i,j) of this map
        (i, j) = (ii[n], jj[n])
        ax = plotted[n]
        map_n = maps[n].T if maps.ndim == 3 else maps[n].transpose(1,0,2)
        ax.imshow(map_n, **imkw)
        ax.tick_params(labelsize=10)
        if border_axes:
            ax.tick_params(labelsize=10)
            ax.yaxis.set_visible(True)
            ax.yaxis.tick_left()
            ax.xaxis.set_visible(True)
            ax.xaxis.tick_bottom()
            if j == 0 and i+1 == geo[0]:
                continue
            if j > 0:
                ax.yaxis.set_visible(False)
            if i+1 < geo[0]:
                ax.xaxis.set_visible(False)
        else:
            ax.axis('off')

    nx, ny = map(int, (nx, ny))
    xt = np.arange(nx-1, -1, -4)[::-1]
    ax.set_xticks(xt)
    if len(x_labels):
        ax.set_xticklabels( [x_labels[t] for t in xt], size=10 )
    yt = np.arange(ny-1, -1, -2)[::-1]
    ax.set_yticks(yt)
    if len(y_labels):
        ax.set_yticklabels( [y_labels[t] for t in yt], size=10 )
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    fig.subplots_adjust(left = 0.05, right = 0.95)
    if not clabel == 'none':
        edge = 0.2
        fig.subplots_adjust(bottom=edge)
        cbar_ax = fig.add_axes([0.25, 2*edge/4, 0.5, edge/4])
        cbar = pp.colorbar(
            ax.images[0], cax=cbar_ax, orientation='horizontal'
            )
        cbar_ax.tick_params(labelsize=10)
        #cbar.set_label(r"%s ($\mu V$)"%score_txt)
        cbar.set_label(clabel)
    else:
        fig.subplots_adjust(bottom = 0.05)
    if title:
        fig.subplots_adjust(top=0.9)
        
    ## # xxx: work this later
    ## if border_axes:
    ##     fig.subplots_adjust(
    ##         left=0.04, bottom=0.04, right=1-0.02,
    ##         wspace=0.02, hspace=0.02, top=1-.1
    ##         )
    ## else:
    ##     fig.subplots_adjust(
    ##         left=0.02, bottom=0.02, right=1-0.02,
    ##         wspace=0.02, hspace=0.02, top=1-.1
    ##         )

    if fill_empty:
        fill_null(missed, cmap=fill_cmap)
        
    return fig

def tile_traces(
        traces, geo=(), p=(), yl=(), twin=(), plot_style='sample',
        border_axes=False, col_major=True, title='', clabel=None,
        tilesize=(1,1), calib_unit='V'
        ):
    p = _build_map(p, geo, col_major)
    if not len(p):
        # no permutation
        p = _build_map(np.arange(traces.shape[0]), p.geometry, p.col_major)
    # traces is either (n_chan, n_trial, n_pts) or (n_chan, n_pts)
    geo = p.geometry
    npt = traces.shape[-1]
    if not yl:
        #yl = tuple(ndim_prctile(traces.ravel(), (0.1, 99.9)))
        if plot_style=='all':
            yl = np.nanmin(traces), np.nanmax(traces)
        else:
            avg = np.nanmean(traces, axis=1)
            std = np.nanstd(traces, axis=1)
            mn, mx = np.nanmin(avg-std), np.nanmax(avg+std)
            yl = mn, mx
    if not twin:
        twin = (0, npt-1)
    tx = np.linspace(twin[0], twin[1], npt)

    #calib = 'right' if clabel else 'none'
    fig, plotted, missed = tiled_axes(geo, p, title=title, tilesize=tilesize)
    ii, jj = p.to_mat()
    for n in xrange(traces.shape[0]):
        chan_t = traces[n]
        
        # get the (i,j) of this map
        (i, j) = (ii[n], jj[n])
        ax = plotted[n]
        if plot_style=='sample' and chan_t.ndim==2:
            mn = np.nanmean(chan_t, axis=0)
            margin = ndim_prctile(chan_t, (15, 85), axis=0)
            ax.fill_between(
                tx, margin[0], y2=margin[1],
                facecolor=(0.6, 0.6, 0.6), edgecolor='none'
                )
            ax.plot(tx, mn, color='k', linewidth=2)
        elif plot_style=='sem' and chan_t.ndim==2:
            mn = np.nanmean(chan_t, axis=0)
            margin = np.nanstd(chan_t, axis=0) / np.sqrt(chan_t.shape[0])
            ax.fill_between(
                tx, mn-margin, y2=mn+margin,
                facecolor=(0.7, 0.7, 0.7), edgecolor='none'
                )
            ax.plot(tx, mn, color='r', linewidth=0.5)
        elif plot_style=='stdev' and chan_t.ndim==2:
            mn = np.nanmean(chan_t, axis=0)
            margin = np.nanstd(chan_t, axis=0)
            ax.fill_between(
                tx, mn-margin, y2=mn+margin,
                facecolor=(0.7, 0.7, 0.7), edgecolor='none'
                )
            ax.plot(tx, mn, color='k', linewidth=2)
            
        else: #elif plot_style=='all':
            ax.plot(tx, chan_t.T, 'b', linewidth=0.1)
        ## ax.set_ylim(yl)
        ## ax.set_xlim(twin)
        if twin[0] < 0:
            ax.axvline(x=0, color='k', linestyle='--')
        
        if border_axes:
            ax.tick_params(labelsize=10)
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()
            if j == 0 and i+1 == geo[0]:
                continue
            if j == 0:
                ax.xaxis.set_visible(False)
            if i+1 == geo[0]:
                ax.yaxis.set_visible(False)
        else:
            ax.axis('off')
            #pp.axis('off')

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=1.0)
    if title:
        fig.subplots_adjust(top=0.95)
    
    if not len(missed):
        return fig
    calibration_axes(plotted[0], calib_ax=missed[-1], calib_unit=calib_unit)
    return fig

from matplotlib.collections import LineCollection, PolyCollection, PatchCollection

from matplotlib.patches import Polygon

def tile_traces_1ax(
        traces, geo=(), p=(), yl=(), twin=(), plot_style='sample',
        col_major=True, title='', tilesize=(1,1), calib_unit='V',
        x_labels=(), y_labels=(), table_style='matrix', **line_kws
        ):

    """

    Parameters
    ----------

    traces : ndarray 2,3-D

    geo : (rows, cols)

    p : flat index or ChannelMap

    col_major : bool
        indicates how p indexes into the table (ignored if p is ChannelMap)

    yl : y-limits for each trace

    twin : t-limits for each trace

    plot_style : str
        "sample" plots mean and interquartile range margins

        "sem" plot means and its standard error margins

        "stdev" plots mean and +/- sigma margins

        "all" plots all samples per grid location

        "single" plots mean-only

    x_labels : sequence
        labels for x-span

    y_labels : sequence
        labels for y-span

    table_style : str
        "matrix" puts cell (0,0) in the top-left corner, the 
        corresponding y_labels sequence downwards (direction of
        increasing row index)

        "cartesian" puts cell (0,0) in the bottom-left corner, the 
        corresponding y_labels sequence upwards (direction of 
        increasing y-value)


    """

    if traces.ndim == 2:
        traces = traces[:,None,:]
    p = _build_map(p, geo, col_major)
    if not len(p):
        # no permutation
        p = _build_map(np.arange(traces.shape[0]), p.geometry, p.col_major)
    # traces is either (n_chan, n_trial, n_pts) or (n_chan, n_pts)
    geo = p.geometry
    npt = traces.shape[-1]
    if not yl:
        #yl = tuple(ndim_prctile(traces.ravel(), (0.1, 99.9)))
        if plot_style=='all':
            yl = np.nanmin(traces), np.nanmax(traces)
        else:
            avg = np.nanmean(traces, axis=1)
            std = np.nanstd(traces, axis=1)
            mn, mx = np.nanmin(avg-std), np.nanmax(avg+std)
            yl = mn, mx
    if not twin:
        twin = (0, npt-1)

    # make calculations for axes limits
    tpad = 0.05 * (twin[1] - twin[0])
    twid = twin[1] - twin[0] + tpad
    ywid = yl[1] - yl[0]

    # leave space of ~ 5% for text
    txtgap_x = geo[1] * (twid+tpad) * 0.02 if len(y_labels) else 0
    txtgap_y = geo[0] * ywid * 0.02 if len(x_labels) else 0

    def vert_xform(v):
        if table_style == 'matrix':
            return geo[0] - v - 1
        else:
            return v
    
    # calculate fig size, given geometry and presence of title
    figsize = ( geo[1] * tilesize[1], geo[0] * tilesize[0] + (len(title)>0) )
    fig = pp.figure(figsize=figsize)

    bottom = left = 0.02
    top = 0.98
    right = 0.85
    if title:
        top = 1 - 1.0/figsize[1]

    ax = fig.add_axes( [left, bottom, right-left, top-bottom] )
    ax.set_ylim( yl[0] - txtgap_y, yl[0] + geo[0] * ywid )
    ax.set_xlim( twin[0] - tpad - txtgap_x, twin[0] + geo[1] * twid )

    # make a line collection of all traces, use the channel map to 
    # apply the appropriate offsets
    tx = np.linspace(twin[0], twin[1], npt)

    chan_offsets = list()
    chan_means = np.nanmean(traces, axis=1)
    for i, j in zip(*p.to_mat()):
        chan_offsets.append( ( j * twid, vert_xform(i) * ywid ) )

    # first make a line collection for any margins to be drawn in
    if plot_style not in ('all', 'single') and traces.shape[1] > 1:
        if plot_style == 'sem':
            mwid = np.nanstd(traces, axis=1) / np.sqrt(traces.shape[1])
            margins = np.array( 
                [ [mn-mw, mn+mw] for mn, mw in zip(chan_means, mwid) ] 
                )
        elif plot_style == 'stdev':
            mwid = np.nanstd(traces, axis=1)
            margins = np.array( 
                [ [mn-mw, mn+mw] for mn, mw in zip(chan_means, mwid) ] 
                )
        else:
            margins = np.array(np.percentile(traces, [25, 75], axis=1))
            margins = margins.transpose(1,0,2)

        lines = [ np.c_[ np.r_[tx, tx[::-1]], np.r_[m[0], m[1][::-1]] ]
                  for m in margins ]
        fills = PatchCollection(
            [ Polygon(line, closed=True) for line in lines ], 
            match_original=False, edgecolors='none',
            facecolors=(0.6, 0.6, 0.6)
            )
        fills.set_offset_position('data')
        fills.set_offsets(chan_offsets)

        ax.add_collection(fills)

    # now plot the single timeseries traces
    if plot_style != 'all':
        # else, chan_means already defined
        lines = [ np.c_[tx, mn] for mn in chan_means ]
        line_kws.setdefault('linewidths', 2 if plot_style == 'single' else 1)
        line_kws.setdefault(
            'colors', 
            dict(sem='r', stdev='k', sample='k', single='b')[plot_style]
            )
        lines = LineCollection(
            lines, offsets=chan_offsets, #transOffset=ax.transData,
            **line_kws
            )
    else:
        lines = list()
        for chan_samps in traces:
            lines.extend( [ np.c_[tx, samp] for samp in chan_samps ] )
        # repeat offsets enough times for each group of traces
        chan_multi_offsets = [ [offset for i in xrange(traces.shape[1])]
                               for offset in chan_offsets ]
        line_kws.setdefault('linewidths', 0.2)
        line_kws.setdefault('colors', 'b')
        lines = LineCollection(
            lines, offsets=chan_multi_offsets, #transOffset=ax.transData,
            **line_kws
            )

    ax.add_collection(lines)

    if twin[0] < 0:
        vert_line = np.array( [ [0, yl[0]], [0, yl[1]] ] )
        trig_lines = LineCollection(
            [ vert_line ] * traces.shape[0],
            offsets=chan_offsets, colors='k', linestyles='--'
            )
        ax.add_collection(trig_lines)

    ax.axis('off')
    #ax.axis('auto')

    if title:
        fig.text(
            0.5, .95, title, fontsize=18, 
            va='baseline', ha='center'
            )
    calibration_axes(
        ax, y_scale=ywid, calib_unit=calib_unit, t_scale=twin[1]-twin[0]
        )

    if len(x_labels) == geo[1]:
        y0 = yl[0] - txtgap_y
        step = twid# + tpad
        for i, lab in enumerate(x_labels):
            ax.text( 
                (i + 0.5)*step + twin[0] - tpad/2.0, 
                y0, lab, va='top', ha='center',
                fontsize=8
                )
    if len(y_labels) == geo[0]:
        # y-labels drawn down in matrix style
        t0 = twin[0] - tpad - txtgap_x
        step = ywid
        for i, lab in enumerate(y_labels):
            ax.text( 
                t0, 
                (vert_xform(i) + 0.5)*step + yl[0], 
                lab, va='center', ha='center',
                fontsize=8
                )
    
    return fig

def tile_images_1ax(
    maps, geo=(), p=(), col_major=True,
    border_axes=False, title='', 
    fill_empty=False, fill_cmap=pp.cm.gray, 
    clabel='none', **imkw
    ):
    pass
