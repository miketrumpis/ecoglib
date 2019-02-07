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
        trimmed=True, cloud=False, counts=False, se=False, slowcloud=False
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
    se : Bool
        Return the standard error of the mean.

    Returns
    -------
    x : ndarray
        lags
    sv : ndarray,
        semivariance
    Nd : ndarray
        bin counts (only if counts==True)
    se : ndarray
        standard error (only if se==True)
    
    """
    # F is an n_site field of values
    # combs is a channel combination bunch
    if cloud:
        x_set = []
        y_set = []
    if xbin is None:
        x = np.unique(combs.dist)
        Nd = np.zeros(len(x), 'i')
        sv = np.empty_like(x)
    else:
        x, assignment = adapt_bins(xbin, combs.dist, return_map=True)
        Nd = np.zeros(len(x), 'i')
        sv = np.empty(len(x))
    serr = np.empty(len(x))
    for n in xrange(len(x)):
        if xbin is None:
            pair_mask = combs.dist == x[n]
        else:
            pair_mask = assignment == x[n]
        x_s1 = F[combs.p1[pair_mask]]
        x_s2 = F[combs.p2[pair_mask]]
        diffs = x_s1 - x_s2
        if trimmed:
            # trim outliers from the population of samples at this lag
            t = 4 if isinstance(trimmed, bool) else trimmed
            # mask differences (not raw samples)
            trim_mask = fenced_out(diffs, thresh=t)
            if cloud:
                diffs = np.ma.masked_array(diffs, mask=~trim_mask)
            else:
                # this should flatten the diffs
                diffs = diffs[trim_mask]
        if cloud:
            diff_var = 0.5 * diffs.var(1)
            y_set.append(diff_var)
            Nd[n] = len(diff_var)
            x_set.append([x[n]] * Nd[n])
            continue
        Nd[n] = len(diffs)
        if not Nd[n]:
            continue
        if robust:
            avg_var = np.power(np.abs(diffs), 0.5).mean() ** 4
            sv[n] = avg_var / 2 / (0.457 + 0.494 / Nd[n])
        else:
            sv[n] = 0.5 * np.mean(diffs ** 2)
        serr[n] = np.std(0.5 * (diffs ** 2)) / np.sqrt(Nd[n])
    if cloud:
        x = np.concatenate(x_set)
        sv = np.concatenate(y_set)
    if counts and se:
        return x, sv, Nd, serr
    if se:
        return x, sv, serr
    if counts:
        return x, sv, Nd
    return x, sv

try:
    from ._semivariance import triu_diffs
    def _pairwise_semivariance(F, robust=False, trimmed=False):

        N, P = F.shape
        diffs = triu_diffs(F, axis=0)
        if trimmed:
            # trim outliers from the population of samples at this lag
            t = 4 if isinstance(trimmed, bool) else trimmed
            m = fenced_out(diffs, thresh=t)
            diffs = np.ma.masked_array(diffs, mask=~m)
            Nd = P - diffs.mask.sum(1)
        else:
            Nd = np.ones(diffs.shape[0], 'i') * P
        if robust:
            avg_var = np.power( np.abs(diffs), 0.5 ).mean(1) ** 4
            sv = avg_var / 2 / (0.457 + 0.494 / Nd)
        else:
            sv = 0.5 * np.mean( diffs**2, axis=1 )

        return sv, Nd
except ImportError:
    def _pairwise_semivariance(*args, **kwargs):
        raise NotImplementedError('Cythonized "triu_diffs" method required.')
    

def fast_semivariogram(
        F, combs, xbin=None, trimmed=True,
        cloud=False, counts=False, se=False, **kwargs
        ):
    """
    Semivariogram estimator with stationarity assumptions, enabling
    faster "flipped" covariance computation.

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
    trimmed : Bool
        Perform outlier detection for extreme values.
    cloud : Bool
        Return (robust, trimmed) estimates for all pairs.
    counts : Bool
        If True, then return the bin-counts for observations at each
        lag in x. If cloud is True, then Nd is the count of inlier
        differences for each pair.
    se : Bool
        Return the standard error of the mean.

    Returns
    -------
    x : ndarray
        lags
    sv : ndarray,
        semivariance
    Nd : ndarray
        bin counts (only if counts==True)
    se : ndarray
        standard error (only if se==True)
    
    """
    # F is an n_site field of values
    # combs is a channel combination bunch


    sv_matrix = ergodic_semivariogram(F, normed=False,
                                      mask_outliers=trimmed, **kwargs)
    x = combs.dist
    sv = sv_matrix[ np.triu_indices(len(sv_matrix), k=1) ]
    sv = np.ma.masked_invalid(sv)
    x = np.ma.masked_array(x, sv.mask).compressed()
    sv = sv.compressed()
    if cloud:
        if counts:
            return x, sv, 1
        return x, sv

    if xbin is None:
        xb = np.unique(x)
        yb = [ sv[ x == u ] for u in xb ]
    else:
        xb, assignment = adapt_bins(xbin, x, return_map=True)
        yb = [ sv[ assignment == u ] for u in xb ]
    Nd = np.array(map(len, yb))

    semivar = np.array( map(np.mean, yb) )
    serr = np.array( map(lambda x: np.std(x) / np.sqrt(len(x)), yb) )
    if counts and se:
        return xb, semivar, Nd, serr
    if se:
        return xb, semivar, serr
    if counts:
        return xb, semivar, Nd
    return xb, semivar

def ergodic_semivariogram(data, normed=False, mask_outliers=True, zero_field=True, covar=False):
    #data = data - data.mean(1)[:,None]
    if zero_field:
        data = data - data.mean(0)
    if mask_outliers:
        if isinstance(mask_outliers, bool):
            thresh = 4.0
        else:
            thresh = mask_outliers
        ## pwr = np.apply_along_axis(np.linalg.norm, 0, data)
        ## m = fenced_out(pwr)
        ## data = data[:, m]
        m = fenced_out(data, thresh=thresh)
        dm = np.zeros_like(data)
        np.putmask(dm, m, data)
        data = dm
        
    if normed:
        data = data / np.std(data, axis=1, keepdims=1)
    cxx = np.einsum('ik,jk->ij', data, data)
    if mask_outliers:
        m = m.astype('i')
        N = np.einsum('ik,jk->ij', m, m)
    else:
        N = data.shape[1]
    cxx /= N
    var = cxx.diagonal()
    if covar:
        return cxx
    return 0.5 * (var[:,None] + var) - cxx
