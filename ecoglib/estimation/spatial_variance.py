import numpy as np
from ecoglib.numutil import fenced_out

__all__ = ['semivariogram']

def adapt_bins(bsize, dists, return_map=False):

    bins = [dists.min()]
    while bins[-1] + bsize < dists.max():
        bins.append( bins[-1] + bsize )
    bins = np.array(bins)
    converged = False
    n = 0
    while not converged:
        diffs = np.abs( dists - bins[:, None] )
        bin_assignment = diffs.argmin(0)
        new_bins = [ dists[ bin_assignment==b ].mean()
                     for b in xrange(len(bins)) ]
        new_bins = np.array(new_bins)
        new_bins = new_bins[ np.isfinite(new_bins) ]
        if len(new_bins) == len(bins):
            dx = np.linalg.norm( bins - new_bins )
            converged = dx < 1e-5
        bins = new_bins
        if n > 20:
            break
        n += 1
    # sometimes the maximum distance to a bin can be greater than the
    # bin size -- seems to only happen on last distance group (probably
    # because of very low weighting). 
    ## diffs = np.abs( dists - bins[:, None] )
    ## print diffs.min(0).max()
    ## mx_diff = diffs.min(0).argmax()
    ## print dists[mx_diff]
    if return_map:
        diffs = np.abs( dists - bins[:, None] )
        bin_assignment = diffs.argmin(0)
        return bins, bins[bin_assignment]
    return bins

def semivariogram(
        F, combs, xbin=None, robust=True,
        trimmed=True, cloud=False, counts=False
        ):
    """
    Classical semivariogram estimator with option for Cressie's robust
    estimator. Can also return a semivariogram "cloud".

    Parameters
    ----------

    F : ndarray, (N, ...)
        One or more samples of N field values.
    combs : Bunch
        Object representing the site-site pairs between N field
        points. combs.p1 and combs.p2 are (site_i, site_j) indices.
        combs.dist is the distance ||site_i - site_j||. This object
        can be found from the ChannelMap.site_combinations attribute.
    xbin : float (optional)
        Bin site distances with this spacing, rather than the default
        of using all unique distances on the grid.
    robust : Bool
        If True, use Cressie's formula for robust semivariance
        estimation. Else use mean-square difference.
    trimmed : Bool
        Perform outlier detection for extreme values.
    cloud : Bool
        Return (robust, trimmed) estimates for all pairs.
    counts : Bool
        If True, then return the bin-counts for observations at each
        lag in x. If cloud is True, then Nd is the count of inlier
        differences for each pair.

    Returns
    -------
    x : ndarray
        lags
    sv : ndarray,
        semivariance
    Nd : ndarray
        bin counts (only if counts==True)
    
    """
    # F is an n_site field of values
    # combs is a channel combination bunch
    if cloud:
        # xxx: this relies on comb.dist and triu_diffs both using
        # upper-triangle convention
        x = combs.dist
        sv, Nd = _pairwise_semivariance(F, robust=robust, trimmed=trimmed)
        if counts:
            return x, sv, Nd
        return x, sv
    else:
        if xbin is None:
            x = np.unique(combs.dist)
            Nd = np.zeros(len(x), 'i')
            sv = np.empty_like(x)
        else:
            x, assignment = adapt_bins(xbin, combs.dist, return_map=True)
            Nd = np.zeros(len(x), 'i')
            sv = np.empty(len(x))
    for n in xrange(len(x)):
        if xbin is None:
            m = combs.dist == x[n]
        else:
            m = assignment == x[n]
        x_s1 = F[ combs.p1[m] ].ravel()
        x_s2 = F[ combs.p2[m] ].ravel()
        if trimmed:
            # trim outliers from the population of samples at this lag
            m = fenced_out( np.r_[x_s1, x_s2] ).reshape(2, len(x_s1))
            m = m[0] & m[1]
            # keep only pairs where both samples are inliers
            x_s1 = x_s1[m]
            x_s2 = x_s2[m]
        Nd[n] = len(x_s1)
        if not Nd[n]:
            if not cloud:
                sv[n] = np.nan
            continue
        if robust:
            avg_var = np.power(np.abs( x_s1 - x_s2 ), 0.5).mean() ** 4
            sv[n] = avg_var / 2 / (0.457 + 0.494 / Nd[n])
        else:
            sv[n] = 0.5 * np.mean( (x_s1 - x_s2)**2 )
    if counts:
        return x, sv, Nd
    return x, sv

try:
    from ._semivariance import triu_diffs
    def _pairwise_semivariance(F, robust=False, trimmed=False):

        N, P = F.shape
        diffs = triu_diffs(F, axis=0)
        if trimmed:
            m = fenced_out(diffs)
            diffs = np.ma.masked_array(diffs, mask=~m)
            Nd = P - diffs.mask.sum(1)
        else:
            Nd = np.ones(N, 'i') * P
        if robust:
            avg_var = np.power( np.abs(diffs), 0.5 ).mean(1) ** 4
            sv = avg_var / 2 / (0.457 + 0.494 / Nd)
        else:
            sv = 0.5 * np.mean( diffs**2, axis=1 )

        return sv, Nd
except ImportError:
    def _pairwise_semivariance(*args, **kwargs):
        raise NotImplementedError('Cythonized "triu_diffs" method required.')
    
def ergodic_semivariogram(data, normed=True, mask_outliers=False):
    #data = data - data.mean(1)[:,None]
    if mask_outliers:
        pwr = np.apply_along_axis(np.linalg.norm, 0, data)
        m = fenced_out(pwr)
        data = data[:, m]
    if normed:
        data = data / np.std(data, axis=1, keepdims=1)
    #data = data - data.mean(0)
    data = data - data.mean()
    cxx = np.einsum('ik,jk->ij', data, data)
    cxx /= data.shape[1]
    var = cxx.diagonal()
    ## if normed:
    ##     cxx /= np.outer(var, var) ** 0.5
    ##     var = np.ones_like(var)
    return 0.5 * (var[:,None] + var) - cxx
    
    ## # zero mean for temporal expectation
    ## dzm = data - data.mean(-1)[:,None]
    ## #dzm = data - data.mean(0)
    ## # I think this adjustment is ok -- this enforces the assumption
    ## # that every site has similar marginal statistics
    ## ## var = dzm.var(-1)
    ## ## dzm = dzm / np.sqrt(var)[:,None]
    ## cxx = np.cov(dzm, bias=1)
    ## var = cxx.diagonal()
    ## var = 0.5 * (var[:,None] + var)
    ## return var - cxx
