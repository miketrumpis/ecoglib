from __future__ import division
import numpy as np
import matplotlib.pyplot as pp

from ecoglib.util import flat_to_mat, mat_to_flat
from ecoglib.numutil import ndim_prctile

def quick_tiles(n_frames, nrow=None, ncol=None, figsize=()):
    if not (nrow or ncol):
        ncol = 4

    if nrow and not ncol:
        ncol = int( np.ceil( float(n_frames) / nrow ) )
    elif ncol and not nrow:
        nrow = int( np.ceil( float(n_frames) / ncol ) )
    if ncol * nrow < n_frames:
        raise ValueError('Not enough tiles for frames')

    if not figsize:
        figsize = (ncol, nrow)
    fig, axes, _ = tiled_axes( 
        (nrow, ncol), np.arange(n_frames), figsize=figsize,
        col_major=False, fill_empty=False
        )
    return fig, axes

def tiled_axes(
        geo, p, figsize=(10,10), col_major=True, cmap=pp.cm.gray,
        fill_empty=True
        ):
    n_plots = len(p)
    missed_tiles = set(range(geo[0]*geo[1]))
    missed_tiles.difference_update(p)
    fig = pp.figure(figsize=figsize)
    plot_axes = list()
    for pn in p:
        (i, j) = flat_to_mat(geo, pn, col_major=col_major)
        ax = pp.subplot2grid( geo, (i, j) )
        plot_axes.append(ax)
    missed_axes = list()
    for pn in missed_tiles:
        (i, j) = flat_to_mat(geo, pn, col_major=col_major)
        ax = pp.subplot2grid( geo, (i, j) )
        if fill_empty:
            ax.imshow(
                np.array([ [.85, .75], [.75, .85] ]), cmap=cmap,
                clim=(0, 1), interpolation='nearest'
                )
        ax.axis('off')
        missed_axes.append(ax)
    return fig, plot_axes, missed_axes
         

def tile_images(maps, geo=(), p=(), col_major=True,
                border_axes=False, title='', **imkw):

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

    if maps.ndim == 4:
        geo = maps.shape[:2]
    elif not len(geo):
        raise RuntimeError(
            "Can't infer geometry from array, nor was a geometry provided"
            )
    geo = map(int, geo)
    oshape = maps.shape
    maps.shape = (-1,) + maps.shape[-2:]
    
    if not len(p):
        # no permutation
        p = np.arange(maps.shape[0])

    fig, plotted, missed = tiled_axes(geo, p, col_major=col_major)

    for n in xrange(maps.shape[0]):
        # get the (i,j) of this map
        (i, j) = flat_to_mat(geo, p[n])
        print 'plotting ', i, j
        ## ax = pp.subplot2grid( geo, (i, j) )
        ax = plotted[n]
        ax.imshow(maps[n].T, **imkw)
        if border_axes:
            if j == 0 and i+1 == geo[0]:
                continue
            if j == 0:
                ax.xaxis.set_visible(False)
            if i+1 == geo[0]:
                ax.yaxis.set_visible(False)
        else:
            ax.axis('off')

    # xxx: work this later
    if border_axes:
        fig.subplots_adjust(
            left=0.04, bottom=0.04, right=1-0.02,
            wspace=0.02, hspace=0.02, top=1-.1
            )
    else:
        fig.subplots_adjust(
            left=0.02, bottom=0.02, right=1-0.02,
            wspace=0.02, hspace=0.02, top=1-.1
            )
    if title:
        fig.text(
            0.5, .925, title, fontsize=18, 
            va='baseline', ha='center'
            )

    maps.shape = oshape
    return fig

def tile_traces(
        traces, geo, p=(), yl=(), twin=(), plot_style='sample',
        border_axes=False, col_major=True, title=''
        ):

    # traces is either (n_chan, n_trial, n_pts) or (n_chan, n_pts)
    geo = map(int, geo)
    npt = traces.shape[-1]
    if not yl:
        yl = tuple(ndim_prctile(traces.ravel(), (0.1, 99.9)))
    if not twin:
        twin = (0, npt-1)
    tx = np.linspace(twin[0], twin[1], npt)
        
    if not len(p):
        # no permutation
        p = np.arange(maps.shape[0])
        
    fig, plotted, missed = tiled_axes(geo, p, col_major=col_major)

    for n in xrange(traces.shape[0]):
        chan_t = traces[n]
        
        # get the (i,j) of this map
        (i, j) = flat_to_mat(geo, p[n])
        print 'plotting ', i, j
        ## ax = pp.subplot2grid( geo, (i, j) )
        ax = plotted[n]
        if plot_style=='sample' and chan_t.ndim==2:
            mn = np.nanmean(chan_t, axis=0)
            margin = ndim_prctile(chan_t, (15, 85), axis=0)
            ax.fill_between(
                tx, margin[0], y2=margin[1],
                facecolor=(0.6, 0.6, 0.6), edgecolor='none'
                )
            ax.plot(tx, mn, color='k', linewidth=2)
        else: #elif plot_style=='all':
            ax.plot(tx, chan_t.T)
        ax.set_ylim(yl)
        if twin[0] < 0:
            ax.axvline(x=0, color='k', linestyle='--')
        
        if border_axes:
            if j == 0 and i+1 == geo[0]:
                continue
            if j == 0:
                ax.xaxis.set_visible(False)
            if i+1 == geo[0]:
                ax.yaxis.set_visible(False)
        else:
            ax.axis('off')
            #pp.axis('off')

    # xxx: work this later
    if border_axes:
        fig.subplots_adjust(
            left=0.04, bottom=0.04, right=1-0.02,
            wspace=0.02, hspace=0.02, top=1-.1
            )
    else:
        fig.subplots_adjust(
            left=0.02, bottom=0.02, right=1-0.02,
            wspace=0.02, hspace=0.02, top=1-.1
            )
    if title:
        fig.text(
            0.5, .925, title, fontsize=18, 
            va='baseline', ha='center'
            )
    return fig


