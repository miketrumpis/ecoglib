# brief numerical utility functions
from __future__ import division
import numpy as np
from scipy.optimize import leastsq, fmin_tnc
import scipy.stats.distributions as dists
import scipy.stats as stats
from scipy.integrate import simps
import scipy.linalg as la

from ecoglib.util import *
from sandbox.array_split import split_at

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

def density_normalize(x, raxis=None):
    if raxis is None:
        mn = np.nanmin(x)
        x = x - mn
        return x / np.nansum(x)

    if x.ndim > 2:
        shape = x.shape
        if raxis not in (0, -1, x.ndim-1):
            raise ValueError('can only normalized in contiguous dimensions')

        if raxis == 0:
            x = x.reshape(x.shape[0], -1)
        else:
            raxis = -1
            x = x.reshape(-1, x.shape[-1])
        xn = density_normalize(x, raxis=raxis)
        return xn.reshape(shape)

    # roll repeat axis to last axis
    x = np.rollaxis(x, raxis, start=2)
    mn = np.nanmin(x, 0)
    x = x - mn
    x = x / np.nansum(x, 0)
    return np.rollaxis(x, 1, start=raxis)

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

def bump1d(x, p):
    # bump fn is a + b * exp{ - ( 1 - (x/xw)**2 )^-1 }
    x = x.ravel()
    xw, b, a = map(lambda x: np.atleast_2d(x).T, p)

    x_arg = 1 - (x/xw)**2
    fx = np.zeros( (xw.shape[0], len(x)) )

    x_bump = np.abs(x) <= xw
    fx[x_bump] = np.exp(-1/x_arg[x_bump])
    fx *= b
    fx += a
    return fx.squeeze()

def bump_fit(y, x):

    y = np.atleast_2d(y)
    pf = np.zeros( (y.shape[0], 4) )
    res = list()
    if x.ndim == 1:
        x = np.tile(x, (y.shape[0], 1))

    def _costfn(p, xx, yy, resids=True):
        errs = yy - bump1d(xx, p)
        if resids:
            return errs
        else:
            return np.sum( errs**2 )
        
    for xi, yi in zip(x, y):
        mx = np.max(yi)
        mn = np.min(yi)
        p0 = np.array([2, mx, mn])

        # parameters are width, gain, bias
        pbnd = [ (1, xi.max()),
                 (0, mx), (mn, mx) ]
        #print pbnd
        #r = leastsq(_costfn, p0, args=(xi, yi))
        r = fmin_tnc(_costfn, p0, args=(xi, yi, False), 
                     approx_grad=1, bounds=pbnd, messages=0)
    
        res.append(r[0])
    return np.asarray(res)

def gauss_fit(y, x=None):

    y = np.atleast_2d(y)
    pf = np.zeros( (y.shape[0], 4) )
    res = list()
    if x is None:
        x = np.arange(y.shape[-1])
    if x.ndim == 1:
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
        
        pbnd = [ (0, len(yi)-1), (0., len(yi)/3.0),
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
    

def mahal_distance(population, test_points, tol=1-1e-2):
    """

    Parameters
    ----------
    
    population : ndarray (n_samps, p)
        A collection of samples from a multivariate population in R^p

    test_points : ndarray ([n_tests], p)
        A 1- or 2D collection of test samples in R^p

    Returns
    -------

    dists
        The Mahalanobis distances of the test points w.r.t. the population.

    """

    test_points = np.atleast_2d(test_points)

    n_tests = test_points.shape[0]
    n_samps = population.shape[0]
    # do usual PCA thing.. 
    # Cxx is estimated by: (1/n)(P' * P) = V * (S/sqrt(n))^2 * V'
    p_mean = np.mean(population, axis=0)
    pop0 = (population - p_mean) / np.sqrt(n_samps-1)
    U, S, Vt = la.svd(pop0, full_matrices=False)

    # project to proper non-degenerate subspace to 
    # avoid ill-conditioned inverse
    if tol < 1:
        portion_variance = np.cumsum(S**2) / np.sum(S**2)
        dims = portion_variance.searchsorted(tol) + 1
    else:
        dims = len(S)
    
    # The projection to subpsace is Vt[:dims]
    P = Vt[:dims]
    icov = 1 / S[:dims]**2

    # subtract mean and project to (dims, n_tests)
    test_points = P.dot( (test_points - p_mean).T )

    signal_dists = np.diag( np.dot( test_points.T * icov, test_points ) )
    return np.sqrt(signal_dists)


def bootstrap_stat(*arrays, **kwargs):
    """
    This method parallelizes simple bootstrap resampling over the 
    1st axis in the arrays. This can be used only with methods that 
    are vectorized over one dimension (e.g. have an "axis" keyword 
    argument).

    kwargs must include the method keyed by "func"

    func : the method to reduce the sample

    n_boot : number of resampling steps

    rand_seed : seed for random state

    args : method arguments to concatenate to the array list

    extra : any further arguments are passed directly to func

    """
    # If axis is given as positive it will have to be increased
    # by one slot
    axis = kwargs.setdefault('axis', -1)
    if axis >= 0:
        kwargs['axis'] = axis + 1

    func = kwargs.pop('func', None)
    n_boot = kwargs.pop('n_boot', 1000)
    rand_seed = kwargs.pop('rand_seed', None)
    args = kwargs.pop('args', [])
    splice_args = kwargs.pop('splice_args', None)
    
    if func is None:
        raise ValueError('func must be set')

    
    np.random.RandomState(rand_seed)

    b_arrays = list()
    for arr in arrays:
        r = len(arr)
        resamp = np.random.randint(0, r, r*n_boot)
        b_arr = np.take(arr, resamp, axis=0)
        b_arr.shape = (n_boot, r) + arr.shape[1:]
        b_arrays.append(b_arr)

    if not splice_args:
        # try to determine automatically
        
        test_input = [b[0] for b in b_arrays] + list(args)

        test_output = func(*test_input, **kwargs)
        if isinstance(test_output, tuple):
            outputs = range(len(test_output))
        else:
            outputs = (0,)
    else:
        outputs = splice_args
        print outputs

    p_func = split_at(
        split_arg=range(len(b_arrays)), 
        splice_at=outputs
        )(func)

    inputs = b_arrays + list(args)
    b_output = p_func(*inputs, **kwargs)
    return b_output
