from __future__ import division
import numpy as np
import matplotlib.pyplot as pp

from ecoglib.util import flat_to_mat, mat_to_flat

def flat_to_mat(mn, idx, col_major=True):
    # convert a flat matrix index into (i,j) style
    (m, n) = mn if col_major else mn[::-1]

    j = idx // m
    i = (idx - j*m)
    return (i, j) if col_major else (j, i)


def tile_images(maps, geo=(), p=(), border_axes=False, **imkw):

    # maps is (n_maps, mx, my), or (nx_tiles, ny_tiles, mx, my)
    # if ndim < 4, then geo must be provided

    # filter the cmap in case it's a literal table
    if imkw.has_key('cmap'):
        cm = imkw['cmap']
        if type(cm) == np.ndarray:
            imkw['cmap'] = pp.cm.colors.ListedColormap(cm)

    if not ( imkw.has_key('clim') or imkw.has_key('vmin') or \
             imkw.has_key('vmax') or imkw.has_key('norm') ):
        vmin = maps.min(); vmax = maps.max()
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

    f = pp.figure(figsize=(10,10))

    for n in xrange(maps.shape[0]):
        # get the (i,j) of this map
        (i, j) = flat_to_mat(geo, p[n])
        print 'plotting ', i, j
        ax = pp.subplot2grid( geo, (i, j) )
        ax.imshow(maps[n].T, **imkw)
        if border_axes:
            if j == 0 and i+1 == geo[0]:
                continue
            if j == 0:
                ax.xaxis.set_visible(False)
            if i+1 == geo[0]:
                ax.yaxis.set_visible(False)
        else:
            pp.axis('off')

    missed_tiles = set(range(geo[0]*geo[1]))
    missed_tiles.difference_update(p)
    imkw['norm'] = pp.Normalize(0,1)
    imkw['clim'] = (0, 1)
    imkw['interpolation'] = 'nearest'
    for t in missed_tiles:
        (i, j) = flat_to_mat(geo, t)
        ax = pp.subplot2grid( geo, (i, j) )
        ax.imshow( np.array([ [.15, .25], [.25, .15] ]), **imkw )
        pp.axis('off')

    # xxx: work this later
    if border_axes:
        f.subplots_adjust(
            left=0.04, bottom=0.04, right=1-0.02,
            wspace=0.02, hspace=0.02, top=1-.1
            )
    else:
        f.subplots_adjust(
            left=0.02, bottom=0.02, right=1-0.02,
            wspace=0.02, hspace=0.02, top=1-.1
            )
    maps.shape = oshape
    return f


