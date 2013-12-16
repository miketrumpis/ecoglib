# brief numerical utility functions
from __future__ import division
import numpy as np

from ecoglib.util import *

def nextpow2(n):
    pow = int( np.floor( np.log2(n) ) + 1 )
    return 2**pow

def ndim_prctile(x, p, axis=0):
    xs = np.sort(x, axis=axis)
    dim = xs.shape[axis]
    idx = np.round( float(dim) * np.asarray(p) / 100 ).astype('i')
    slicer = [slice(None)] * x.ndim
    slicer[axis] = idx
    return xs[slicer]

def unity_normalize(x, axis=None):
    if axis is None:
        mn = x.min(axis=axis)
        mx = x.max(axis=axis)
        return (x-mn)/(mx-mn)

    x = np.rollaxis(x, axis)
    mn = x.min(axis=-1)
    mx = x.max(axis=-1)
    while mn.ndim > 1:
        mn = mn.min(axis=-1)
        mx = mx.max(axis=-1)
    
    slicer = [slice(None)] + [np.newaxis]*(x.ndim-1)
    x = x - mn[slicer]
    x = x / (mx-mn)[slicer]
    return np.rollaxis(x, 0, axis+1)

def density_normalize(x, axis=None):
    mn = x.min(axis=axis)
    if axis is None:
        x = x - mn
        return x / x.sum()

    x = np.rollaxis(x, axis)
    mn = x.min(axis=-1)
    accum = x.sum(axis=-1)
    while mn.ndim > 1:
        mn = mn.min(axis=-1)
        accum = accum.sum(axis=-1)

    n_pt = np.prod(x.shape[1:])
    slicer = [slice(None)] + [np.newaxis]*(x.ndim-1)
    x = x - mn[slicer]
    x = x / (accum-n_pt*mn)[slicer]
    return np.rollaxis(x, 0, axis+1)

def mirror_extend(x, border):
    """Extend matrix x by amount in border.
    """

    if np.iterable(border):
        bx, by = border
    else:
        bx = border
        by = border

    (m, n) = x.shape
    w = np.zeros( (m + 2*by, n + 2*bx), x.dtype )
    w[by:by+m, bx:bx+n] = x
    # top mirror
    w[:by, bx:bx+n] = x[1:by+1, :][::-1, :]
    # bottom mirror
    w[by+m:, bx:bx+n] = x[m-by-1:m-1, :][::-1, :]
    # left mirror
    w[by:by+m, :bx] = x[:, 1:bx+1][:, ::-1]
    # right mirror
    w[by:by+m, bx+n:] = x[:, n-bx-1:n-1][:, ::-1]

    return w    
    
def mean_shift(f, bw):
    """Use ramp-kernel mean shift to find the mode of frame f
    """
    
    fy, fx = f.shape
    if np.iterable(bw):
        bwx, bwy = bw
    else:
        bwx = bw
        bwy = bw
    
    ncol = 2 * (bwx//2) + 1
    nrow = 2 * (bwy//2) + 1
    
    midx = (ncol+1) // 2
    midy = (nrow+1) // 2

    # normalize the frame
    f -= f.min()
    f /= f.max()

    mode = np.argmax(f.ravel())
    (my, mx) = flat_to_mat(f.shape, mode, col_major=False)

    offx = np.arange(-midx, midx+1)
    offy = np.arange(-midy, midy+1)

    def _wrapped_window(x, y, wx, wy):
        # Return a window about (x,y) of size +/- (wx, wy).
        # The window uses wrap-around extension.
        xx, yy = np.mgrid[x-wx:x+wx+1, y-wy:y+wy+1]
        xx = np.mod(xx, fx).astype('i')
        yy = np.mod(yy, fy).astype('i')
        f_ind = mat_to_flat( 
            (fy, fx), yy.ravel(), xx.ravel(), col_major=False 
            )
        return f.flat[f_ind].reshape(2*wx+1, 2*wy+1)
    
    iter = 1
    extended = 0
    while True:

        rmx = round(mx)
        rmy = round(my)

        awin = _wrapped_window(rmx, rmy, midx, midy)
        
        mx = mx + np.sum( offx * np.sum(awin, axis=0) ) / awin.sum()
        my = my + np.sum( offy * np.sum(awin, axis=1) ) / awin.sum()

        if round(mx) == rmx and round(my) == rmy:
            break

        iter += 1
        if iter > 20 and not extended:
            f = mirror_extend(f, (midy, midx))
            mx = mx + midx
            my = my + midy
            extended = 1
        if iter > 40:
            print 'did not converge'
            mx = mx - midx
            my = my - midy
            return

    if extended:
        mx = mx - midx
        my = my - midy

    return mx, my
        

