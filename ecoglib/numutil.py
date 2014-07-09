# brief numerical utility functions
from __future__ import division
import numpy as np
from scipy.optimize import leastsq, fmin_tnc
import scipy.stats.distributions as dists
import scipy.stats as stats
from scipy.integrate import simps

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

def center_samples(x, axis=-1):
    # normalize samples with a "Normal" transformation
    mu = x.mean(axis=axis)
    sig = x.std(axis=axis)
    if x.ndim > 1:
        slices = [slice(None)] * x.ndim
        slices[axis] = None
        mu = mu[slices]
        sig = sig[slices]
    y = x - mu
    y /= sig
    return y

def sphere_samples(x, axis=-1):
    # normalize samples by projecting to a hypersphere
    norm = np.linalg.norm(axis=axis)
    slices = [slice(None)] * x.ndim
    slices[axis] = None
    y = x / norm
    return y

def fenced_out(samps, quantiles=(25,75), thresh=3.0, axis=None, low=True):

    oshape = samps.shape

    if axis is None:
        # do pooled distribution
        samps = samps.ravel()
    else:
        # roll axis of interest to the end
        samps = np.rollaxis(samps, axis, samps.ndim)

    quantiles = map(float, quantiles)
    qr = np.percentile(samps, quantiles, axis=-1)
    extended_range = thresh * (qr[1] - qr[0])
    high_cutoff = qr[1] + extended_range/2
    low_cutoff = qr[0] - extended_range/2
    if not low:
        out_mask = samps < high_cutoff[...,None]
    else:
        out_mask = (samps < high_cutoff[...,None]) & \
          (samps > low_cutoff[...,None])

    if axis is None:
        out_mask.shape = oshape
    else:
        out_mask = np.rollaxis(out_mask, samps.ndim-1, axis)
    return out_mask

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
    fn = f - f.min()
    fn /= fn.max()

    mode = np.argmax(fn.ravel())
    (my, mx) = flat_to_mat(fn.shape, mode, col_major=False)

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
        return fn.flat[f_ind].reshape(2*wx+1, 2*wy+1)
    
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
            f = mirror_extend(fn, (midy, midx))
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

def gauss1d(x, p, jacobian=False):
    x = x.ravel()
    # process as if all parameters are vectors and we're returning
    # a matrix of values
    x0, xw, alpha, bias = map(lambda x: np.atleast_2d(x).T, p)
    if jacobian:
        raise NotImplementedError

    xt = x - x0
    x_arg = -(xt/xw)**2
    fx = alpha * np.exp(x_arg/2) + bias
    return fx.squeeze()

def gauss_fit(y, x=None):

    y = np.atleast_2d(y)
    pf = np.zeros( (y.shape[0], 4) )
    res = list()
    if x is None:
        x = np.arange(y.shape[-1])
        x = np.tile(x, (y.shape[0], 1))

    def _costfn(p, xx, yy, resids=True, weighted=True):
        errs = yy - gauss1d(xx, p)
        if weighted:
            errs *= (dists.norm.pdf(xx, p[0], np.sqrt(p[1])) + .2)
        if resids:
            return errs
        else:
            return np.sum( errs**2 )
        
    for xi, yi in zip(x, y):
        mx = np.max(yi)
        mn = np.min(yi)
        p0 = np.array([np.argmax(yi), 1, mx, mn])
        
        pbnd = [ (0, len(yi)-1), (0.1, len(yi)/3.0),
                 (0, mx), (mn, mx) ]
        #print pbnd
        #r = leastsq(_costfn, p0, args=(xi, yi))
        r = fmin_tnc(_costfn, p0, args=(xi, yi, False, False), 
                     approx_grad=1, bounds=pbnd, messages=0)
    
        res.append(r[0])
    return np.asarray(res)

def spline_fit(y, x=None):
    pass


# XXX: Results seem funky in this method, **revisit!!**
def chisquare_normal(samps, k=30):
    cnts, edges = np.histogram(samps, bins=k)
    if np.any(cnts < 5):
        #print 'warning, some categories have a count less than 5'
        while np.any(cnts < 5):
            min_bin = np.argmin(cnts)
            bin_cnt = cnts[min_bin]
            #mode_bin = np.argmax(cnts)
            left_bin = cnts[min_bin-1] if min_bin>0 else 0
            right_bin = cnts[min_bin+1] if min_bin<len(cnts)-1 else 0
            # split it proportionally between left and right bins
            prop_left = left_bin / float(left_bin+right_bin)
            pl = int(round(bin_cnt * prop_left))
            pr = bin_cnt - pl
            #pr = right_bin / float(left_bin+right_bin)
            cnts = np.r_[cnts[:min_bin], cnts[min_bin+1:]]
            if min_bin > 0:
                cnts[min_bin-1] += pl
            if min_bin < len(cnts):
                cnts[min_bin] += pr
            dx = edges[min_bin+1] - edges[min_bin]
            edges = np.r_[edges[:min_bin+1], edges[min_bin+2:]]
            # also adjust top bin edge
            edges[min_bin] += prop_left * dx            
            
    e_freq = np.diff(dists.norm.cdf(edges))
    o_freq = cnts.astype('d') / len(samps)

    chi2, p = stats.chisquare(o_freq, e_freq, 2)
    return chi2, p
    
def roc(null_samp, sig_samp):
    # Create an empirical ROC from samples of a target distribution
    # and samples of a no-target distribution. For threshold values,
    # use only the union of sample values.
    
    mn = min( null_samp.min(), sig_samp.min() )
    mx = max( null_samp.max(), sig_samp.max() )
    
    #thresh = np.linspace(mn, mx, n)
    thresh = np.union1d(null_samp, sig_samp)

    # false pos is proportion of samps in null samp that are > thresh
    false_hits = (null_samp >= thresh[:,None]).astype('d')
    false_pos = np.sum(false_hits, axis=1) / len(null_samp)

    # true pos is proportion of samps in sig samp that are > thresh
    hits = (sig_samp >= thresh[:,None]).astype('d')
    true_pos = np.sum(hits, axis=1) / len(sig_samp)

    return np.row_stack( (false_pos[::-1], true_pos[::-1]) )

def integrate_roc(roc_pts):
    x, y = roc_pts
    same_x = np.r_[False, np.diff(x) == 0]
    # only keep pts on the curve where x increases
    x = x[~same_x]
    y = y[~same_x]
    cp = simps(y, x=x, even='avg')
    return cp


import mpmath

import math
 
def flnf(f):
    return f*math.log(f) if f>0.5 else 0
 
def gtest2x2(tab):
    a, b = tab[0]
    c, d = tab[1]
    row1 = a+b
    row2 = c+d
    col1 = a+c
    col2 = b+d
     
    total = flnf(a+b+c+d)
    celltotals = flnf(a)+flnf(b)+flnf(c)+flnf(d)
    rowtotals = flnf(row1)+flnf(row2)
    coltotals = flnf(col1)+flnf(col2)
     
    gstat = 2*(celltotals + total-(rowtotals+coltotals))
    p = GtoP(gstat, 1)
    return gstat, p
 
def mpmathcdf(g,df,dps=10):
    mpmath.mp.dps = dps
     
    x,k = mpmath.mpf(g), mpmath.mpf(df)
    cdf = mpmath.gammainc(k/2, 0, x/2, regularized=True)
     
    # floating point precision insufficient, use more precision
    if cdf == 1.0:
        if dps > 4000:
            return cdf # give up after a certain point
        else:
            cdf = mpmathcdf(g,df,dps*2)
    return cdf
 
def GtoP(g,df):
    assert g >= 0, g
    return float(1-mpmathcdf(g,df))

import scipy.ndimage as ndimage

def savitzky_golay(y, window_size, order, deriv=0, rate=1, axis=-1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------

    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute
        (default = 0 means only smoothing)

    Returns
    -------

    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----

    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------

    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------

    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    ## b = np.mat(
    ##     [[k**i for i in order_range]
    ##      for k in range(-half_window, half_window+1)]
    ##     )
    ## m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    ix = np.arange(-half_window, half_window+1, dtype='d')
    bt = np.array( [np.power(ix, k) for k in order_range] )
    if np.iterable(deriv):
        scl = [ rate**d * factorial(d) for d in deriv ]
        scl = np.array(scl).reshape( len(deriv), 1 )
    else:
        scl = rate**deriv * factorial(deriv)
    m = np.linalg.pinv(bt.T)[deriv] * scl

    if m.ndim == 2:
        ys = [ndimage.convolve1d(y, mr[::-1], mode='constant', axis=axis)
              for mr in m]
        return np.array(ys)
    
    return ndimage.convolve1d(y, m[::-1], mode='constant', axis=axis)
    
    # pad the signal at the extremes with
    # values taken from the signal itself
    #firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    #lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    #y = np.concatenate((firstvals, y, lastvals))
    #return np.convolve( m[::-1], y, mode='valid')

def mini_mean_shift(f, ix, m):
    #fdens = density_normalize(f[ix-m:ix+m+1])
    fdens = f[ix-m:ix+m+1]
    fdens = fdens - fdens.min()
    return (fdens * np.arange(ix-m, ix+m+1)).sum() / fdens.sum()

def peak_to_peak(x, m, p=4, xwin=(), msiter=4, points=False):
    # m is approimately the characteristic width of a peak
    # x is 2d, n_array x n_pts
    oshape = x.shape
    x = x.reshape(-1, oshape[-1])
    
    # force m to be odd and get 1st derivatives
    m = 2 * (m//2) + 1
    dx = savitzky_golay(x, m, p, deriv=1, axis=-1)

    if not xwin:
        xwin = (0, dx.shape[axis])
    xwin = map(int, xwin)

    pk_dx = np.argmax(dx[:, xwin[0]:xwin[1]], axis=-1) + xwin[0]
    # these are our starting points for peak finding

    pos_pks = np.zeros(x.shape[0], 'i')
    neg_pks = np.zeros(x.shape[0], 'i')
    bw = m//2
    for n in xrange(x.shape[0]):
        trace = x[n].copy()
        # must be all positive
        #trace = trace - trace.min()
        ix0 = pk_dx[n]
        k = 0
        while k < msiter:
            ix = round(mini_mean_shift(trace, ix0, bw))
            if not ix-ix0:
                break
            ix0 = ix
            k += 1
        pos_pks[n] = (ix-bw) + np.argmax(trace[ix-bw:ix+bw+1])

        # grab a slice rewinding a little bit from here (in order
        # to keep window slicing in-bounds)
        trace = -x[n, pos_pks[n]-m:] + x[n, pos_pks[n]]
        ix0 = m + bw
        skip = 1
        k = 0
        # allow more mean-shift iterations, since the initial point
        # is a very rough guess
        while k < 2*msiter:
            if ix0 > len(trace) - bw:
                # we have no idea where we are!
                # return global minimum
                ix = np.argmin(trace)
                break
            ix = round(mini_mean_shift(trace, ix0, bw))
            if not ix-ix0:
                break
            if ix < ix0:
                # going backwards.. reset with bigger skip ahead
                ix0 = m + skip*bw
                skip += 1
                k = 0
                continue
            ix0 = ix
            k += 1

        #neg_pks[n] = round(ix) + pos_pks[n] - m
        neg_pks[n] = (ix-bw) + np.argmax(trace[ix-bw:ix+bw+1]) + \
          (pos_pks[n]-m)

    cx = np.arange(x.shape[0])
    p2p = x[ (cx, pos_pks) ] - x[ (cx, neg_pks) ]

    if points:
        return map(lambda x: x.reshape(oshape[:-1]), (p2p, neg_pks, pos_pks))
    else:
        return p2p.reshape(oshape[:-1])

    
def peak_to_peak2(x, m, p=4, xwin=(), msiter=4, points=False):
    # m is approimately the characteristic width of a peak
    # x is 2d, n_array x n_pts
    oshape = x.shape
    x = x.reshape(-1, oshape[-1])
    
    # force m to be odd and get 1st derivatives
    m = 2 * (m//2) + 1
    if m < 2*p:
        # m *must* satisify m > p+1
        # but pad to 2*p - 1 for stability
        m = 2*p - 1
    
    dx = savitzky_golay(x, m, p, deriv=1, axis=-1)

    if not xwin:
        xwin = (0, dx.shape[axis])
    xwin = map(int, xwin)

    pk_dx = np.argmax(dx[:, xwin[0]:xwin[1]], axis=-1) + xwin[0]
    # these are our starting points for peak finding

    pos_pks = np.zeros(x.shape[0], 'i')
    neg_pks = np.zeros(x.shape[0], 'i')
    bw = m//2
    for n in xrange(x.shape[0]):
        trace = x[n].copy()
        # must be all positive
        #trace = trace - trace.min()
        ix = pk_dx[n]
        pos_pks[n] = (ix-m) + np.argmax(trace[ix-m:ix+m+1])

        # grab a slice rewinding a little bit from here (in order
        # to keep window slicing in-bounds)
        trace = -x[n, pos_pks[n]-m:] + x[n, pos_pks[n]]
        ix0 = m + bw
        skip = 1
        k = 0
        # allow more mean-shift iterations, since the initial point
        # is a very rough guess
        while k < msiter:
            if ix0 >= len(trace) - bw:
                # we have no idea where we are!
                # return global minimum
                ix = np.argmin(trace)
                break
            ix = round(mini_mean_shift(trace, ix0, bw))
            if not ix-ix0:
                break
            if ix < ix0:
                # going backwards.. reset with bigger skip ahead
                ix0 = m + skip*bw
                skip += 1
                k = 0
                continue
            ix0 = ix
            k += 1

        #neg_pks[n] = round(ix) + pos_pks[n] - m
        neg_pks[n] = (ix-bw) + np.argmax(trace[ix-bw:ix+bw+1]) + \
          (pos_pks[n]-m)

    cx = np.arange(x.shape[0])
    p2p = x[ (cx, pos_pks) ] - x[ (cx, neg_pks) ]

    if points:
        return map(lambda x: x.reshape(oshape[:-1]), (p2p, neg_pks, pos_pks))
    else:
        return p2p.reshape(oshape[:-1])

def peak_to_peak3(x, m, p=4, xwin=(), msiter=4, points=False):
    # m is approimately the characteristic width of a peak
    # x is 2d, n_array x n_pts
    oshape = x.shape
    x = x.reshape(-1, oshape[-1])
    
    # force m to be odd and get 1st derivatives
    m = 2 * (m//2) + 1
    if m < 2*p:
        # m *must* satisify m > p+1
        # but pad to 2*p - 1 for stability
        m = 2*p - 1
    
    sx, dx = savitzky_golay(x, m, p, deriv=(0,1), axis=-1)

    if not xwin:
        xwin = (0, dx.shape[axis])
    xwin = map(int, xwin)

    pk_dx = np.argmax(dx[:, xwin[0]:xwin[1]], axis=-1) + xwin[0]
    # these are our starting points for peak finding

    pos_pks = np.zeros(x.shape[0], 'i')
    neg_pks = np.zeros(x.shape[0], 'i')
    bw = m//2
    for n in xrange(x.shape[0]):
        trace = x[n].copy()
        # must be all positive
        #trace = trace - trace.min()
        ix = pk_dx[n]
        rewind = max(0, int(ix-0.5*m))
        pos_pks[n] = rewind + np.argmax(trace[rewind:int(ix+0.5*m)])

        #trace = sx[n, pos_pks[n]:pos_pks[n]+5*m]
        trace = x[n, pos_pks[n]:pos_pks[n]+3*m]
        neg_pks[n] = np.argmin(trace) + pos_pks[n]

    cx = np.arange(x.shape[0])
    p2p = x[ (cx, pos_pks) ] - x[ (cx, neg_pks) ]

    if points:
        return map(lambda x: x.reshape(oshape[:-1]), (p2p, neg_pks, pos_pks))
    else:
        return p2p.reshape(oshape[:-1])

    
