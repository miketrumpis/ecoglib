from __future__ import division
import numpy as np
import matplotlib.pyplot as pp

from ecoglib.util import ChannelMap, flat_to_mat, mat_to_flat
from ecoglib.numutil import ndim_prctile

from devices import units

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

    p = _build_map(p, geo, col_major)
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
    
    fig = pp.figure(figsize=figsize)
    fig.subplots_adjust(
        left=subplots_left, right=subplots_right,
        bottom=subplots_bottom, top=subplots_top
        )
    plot_axes = list()
    ii, jj = p.to_mat()
    subplot_grid = pp.GridSpec(*geo)
    for i, j in zip(ii, jj):
        ax_spec = subplot_grid.new_subplotspec( (i,j) )
        ax = fig.add_subplot(ax_spec)
        plot_axes.append(ax)
    missed_axes = list()
    for pn in missed_tiles:
        (i, j) = flat_to_mat(geo, pn, col_major=p.col_major)
        ax_spec = subplot_grid.new_subplotspec( (i,j) )
        ax = fig.add_subplot(ax_spec)
        ax.axis('off')
        missed_axes.append(ax)

    if title:
        fig.text(
            0.5, .95, title, fontsize=18, 
            va='baseline', ha='center'
            )

    return fig, plot_axes, missed_axes

def calibration_axes(
        ref_ax, y_scale=None, calib_ax=None, calib_unit='V'
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

    time_units = 'msec'
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
        xw = 0.96-x1
        yw = y1 - y0

        calib_ax = ref_ax.figure.add_axes([x1+0.02, y0, xw, yw])
    else:
        c_pos = calib_ax.get_position()
        xw = c_pos.x1 - c_pos.x0
        
    # full plot width
    xw1 = x1 - x0
    # equivalently scaled sub-interval in new axes is this long
    sub_t_len = (float(xw)/float(xw1)) * t_len

    #print 'calib axis t-len:', sub_t_len
    
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
    
    t_txt = '%d %s'%(t_calib, time_units)

    ## if time_units == 'sec':
    ##     t_calib /= 1e3
    ##     sub_t_len /= 1e3
    
    # repurpose y vars
    # start vertical bar 1/4 of the way from the bottom
    # y0 = (3/4.) * ylim[0] + (1/4.) * ylim[1]
    y0 = 0.5 * (ylim[0] + ylim[1] - y_calib * scale_back)
    y1 = y0 + y_calib * scale_back
    ## if ylim[1] - ylim[0] < 1:
    ##     y1 = y0 + y_calib/1e6
    ## else:
    ##     y1 = y0 + y_calib

    # center the t-calibration bar
    t0 = -t_calib/2.0
    t1 = t_calib/2.0

    calib_ax.plot([t0, t1], [y0, y0], 'k', linewidth=3)
    calib_ax.plot([t0, t0], [y0, y1], 'k', linewidth=3)

    calib_ax.text(
        t0-0.2*sub_t_len, y0, y_txt, ha='center', va='bottom', 
        rotation='vertical', fontsize=11
        )
    calib_ax.text(
        0, y0-.25*(y1-y0), t_txt, ha='center', va='top', fontsize=11
        )
    calib_ax.set_ylim(ylim)
    calib_ax.set_xlim(-sub_t_len/2.0, sub_t_len/2.0)
    calib_ax.axis('off')
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
            ax.plot(tx, chan_t.T, 'b', linewidth=0.5)
        ax.set_ylim(yl)
        ax.set_xlim(twin)
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
    
    if not missed:
        return fig
    calibration_axes(plotted[0], calib_ax=missed[-1], calib_unit=calib_unit)
    return fig


