from __future__ import division
import numpy as np
import matplotlib.pyplot as pp

from ecoglib.util import ChannelMap, flat_to_mat, mat_to_flat
from ecoglib.numutil import ndim_prctile

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

    p = _build_map(p, geo, col_major)
    assert p.geometry == geo, 'Provided channel map has different geometry'
    n_plots = len(p)
    missed_tiles = set(range(geo[0]*geo[1]))
    missed_tiles.difference_update(p)
    figsize = [geo[1]*tilesize[0], geo[0]*tilesize[1]]
    if title:
        figsize[1] += 1
    if calib.lower() != 'none':
        if calib in ('left', 'right', 'side'):
            figsize[0] += 1
        else:
            figsize[1] += 1
    fig = pp.figure(figsize=figsize)
    plot_axes = list()
    ii, jj = p.to_mat()
    for i, j in zip(ii, jj):
        ax = pp.subplot2grid( geo, (i, j) )
        plot_axes.append(ax)
    missed_axes = list()
    for pn in missed_tiles:
        (i, j) = flat_to_mat(geo, pn, col_major=p.col_major)
        ax = pp.subplot2grid( geo, (i, j) )
        ax.axis('off')
        missed_axes.append(ax)

    if title:
        fig.text(
            0.5, .95, title, fontsize=18, 
            va='baseline', ha='center'
            )

    return fig, plot_axes, missed_axes
         
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
    tilesize = (nx/max(nx,ny), ny/max(nx,ny)) # width, height
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
        edge = 0.15
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
        tilesize=(1,1)
        ):
    p = _build_map(p, geo, col_major)
    if not len(p):
        # no permutation
        p = _build_map(np.arange(maps.shape[0]), p.geometry, p.col_major)
    # traces is either (n_chan, n_trial, n_pts) or (n_chan, n_pts)
    geo = p.geometry
    npt = traces.shape[-1]
    if not yl:
        #yl = tuple(ndim_prctile(traces.ravel(), (0.1, 99.9)))
        if plot_style=='all':
            yl = np.nanmin(traces), np.nanmax(traces)
        else:
            avg = np.nanmean(traces, axis=1)
            std = np.std(traces, axis=1)
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

    ma = missed[-1]
    ma.set_xlim(ax.get_xlim())
    ma.set_ylim(ax.get_ylim())
    y_calib_size = max((yl[1] // 1e-4 - 1), 1) * 1e2
    x_calib_size = 100
    # offset calib bars 50 ms after twin[0]?
    x0 = twin[0] + 50
    ma.add_line(
        pp.Line2D( [x0, x0], [0, y_calib_size*1e-4], color='k', linewidth=2 )
        )
    ma.add_line(
        pp.Line2D( 
            [x0, x0 + x_calib_size], 
            [0, 0], color='k', linewidth=2 
            )
        )
    ma.text(
        x0-10, y_calib_size*1e-6/2, r"%d $\mu V$"%y_calib_size, 
        ha='right', va='center', rotation='vertical'
        )
    ma.text(
        x0*1/3 + twin[-1]*2/3., -y_calib_size*1e-6/10, '%d ms'%x_calib_size, 
        ha='center', va='top'
        )
    
    return fig


