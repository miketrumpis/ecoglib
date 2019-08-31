import numpy as np
from ecogdata.numutil import fenced_out


__all__ = ['semivariogram', 'fast_semivariogram', 'ergodic_semivariogram', 'semivariogram', 'adapt_bins',
           'binned_variance', 'binned_variance_aggregate', 'resample_bins', 'subsample_bins', 'concat_bins']


def semivariogram(F, combs, xbin=None, robust=True, trimmed=True, cloud=False, counts=False, se=False):
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
    for n in range(len(x)):
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
            # this should be compressed -- in the event that a pair has no valid differences
            diff_var = (0.5 * diffs.var(1)).compressed()
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
            avg_var = np.power(np.abs(diffs), 0.5).mean(1) ** 4
            sv = avg_var / 2 / (0.457 + 0.494 / Nd)
        else:
            sv = 0.5 * np.mean(diffs**2, axis=1)

        return sv, Nd
except ImportError:
    def _pairwise_semivariance(*args, **kwargs):
        raise NotImplementedError('Cythonized "triu_diffs" method required.')


def fast_semivariogram(F, combs, xbin=None, trimmed=True, cloud=False, counts=False, se=False, **kwargs):
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
    sv = sv_matrix[np.triu_indices(len(sv_matrix), k=1)]
    sv = np.ma.masked_invalid(sv)
    x = np.ma.masked_array(x, sv.mask).compressed()
    sv = sv.compressed()
    if cloud:
        if counts:
            return x, sv, 1
        return x, sv

    xb, yb = binned_variance(x, sv, binsize=xbin)
    Nd = np.array(list(map(len, yb)))
    xb, semivar, serr = binned_variance_aggregate(xb, yb)
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
    return 0.5 * (var[:, None] + var) - cxx


# Not sure if this belongs here -- this functionality overlaps pretty well with the output preparation of
# fast_semivariogram
def cxx_to_pairs(cxx, chan_map, **kwargs):
    if cxx.ndim < 3:
        cxx = cxx[np.newaxis, :, :]
    chan_combs = chan_map.site_combinations
    pairs = zip(chan_combs.p1, chan_combs.p2)
    ix = [x for x, y in sorted(enumerate(pairs), key=lambda x: x[1])]
    idx1 = chan_combs.idx1[ix]
    idx2 = chan_combs.idx2[ix]
    dist = chan_combs.dist[ix]
    tri_x = np.triu_indices(cxx.shape[1], k=1)
    cxx_pairs = np.array([c_[tri_x] for c_ in cxx])
    return dist, cxx_pairs.squeeze()


def adapt_bins(bsize, dists, return_map=False):

    bins = [dists.min()]
    while bins[-1] + bsize < dists.max():
        bins.append(bins[-1] + bsize)
    bins = np.array(bins)
    converged = False
    n = 0
    while not converged:
        diffs = np.abs(dists - bins[:, None])
        bin_assignment = diffs.argmin(0)
        new_bins = [dists[bin_assignment == b].mean()
                    for b in range(len(bins))]
        new_bins = np.array(new_bins)
        new_bins = new_bins[np.isfinite(new_bins)]
        if len(new_bins) == len(bins):
            dx = np.linalg.norm(bins - new_bins)
            converged = dx < 1e-5
        bins = new_bins
        if n > 20:
            break
        n += 1
    # sometimes the maximum distance to a bin can be greater than the
    # bin size -- seems to only happen on last distance group (probably
    # because of very low weighting).
    # diffs = np.abs( dists - bins[:, None] )
    # print diffs.min(0).max()
    # mx_diff = diffs.min(0).argmax()
    # print dists[mx_diff]
    if return_map:
        diffs = np.abs(dists - bins[:, None])
        bin_assignment = diffs.argmin(0)
        return bins, bins[bin_assignment]
    return bins


def binned_variance(x, y, binsize=None):
    if binsize is None:
        xu = np.unique(x)
    else:
        xu, x = adapt_bins(binsize, x, return_map=True)
    return xu, [y[x == u] for u in xu]


def subsample_bins(xb, yb, min_bin=-1, max_bin=-1):
    """
    Sub-sample bins to equalize the group sizes. The minimum group
    size can be specified by min_bin, or will be set by the minimum
    of the provided group sizes. Bins with sizes smaller than the
    minimum will be dropped.
    """

    group_sizes = map(len, yb)
    if min_bin < 0:
        min_bin = min(group_sizes)

    if any([g < min_bin for g in group_sizes]):
        nb = len(group_sizes)
        xb = [xb[i] for i in range(nb) if group_sizes[i] >= min_bin]
        yb = [yb[i] for i in range(nb) if group_sizes[i] >= min_bin]

    if max_bin > 0:
        min_bin = max_bin

    yb_r = [np.random.choice(y_, min_bin, replace=False) for y_ in yb]
    return xb, yb_r


def resample_bins(xb, yb, min_bin=10, beta=0.5):
    """
    Do a bootstrap resample within the len(xb) bins in the list yb.
    Only resample if there are at least min_bin elements in a bin,
    otherwise reject the entire bin with probability (1-beta).
    """

    xb = np.asarray(xb)
    bin_size = np.array(map(len, yb))
    b_mask = bin_size >= min_bin
    yb_r = [y_[np.random.randint(0, len(y_), len(y_))]
            for y_, b in zip(yb, b_mask) if b]
    yb_sm = [y_ for y_, b in zip(yb, b_mask) if not b]
    xb_r = xb[b_mask]
    keep_small = np.random.rand(len(xb) - b_mask.sum()) < beta
    xb_r = np.r_[xb_r, xb[~b_mask][keep_small]]
    yb_r.extend([y_ for y_, k in zip(yb_sm, keep_small) if k])
    return xb_r, yb_r


def concat_bins(xb, yb):
    sizes = map(len, yb)
    x = np.concatenate([[x_] * g for x_, g in zip(xb, sizes)])
    y = np.concatenate(yb)
    return x, y


def _get_scale_method(scale_type):
    def iqr(sample):
        return np.nanpercentile(sample, [25, 75])

    def sd(sample):
        return np.std(sample)

    def sem(sample):
        return sd(sample) / np.sqrt(len(sample))
    # XXX: bootstrap TODO

    methods = {'iqr': iqr, 'sd': sd, 'std': sd, 'sem': sem}
    if scale_type.lower() not in methods:
        raise ValueError('scale function {} not implemented'.format(scale_type))
    return methods[scale_type.lower()]


def binned_variance_aggregate(xb, y_tab, mid_type='mean', scale_type='sem'):
    mid_fn = {'mean': np.mean,
              'median': np.median}
    if mid_type.lower() not in mid_fn:
        raise ValueError('middle function {} not implemented'.format(mid_type))
    mid_fn = mid_fn.get(mid_type.lower())
    scale_fn = _get_scale_method(scale_type)

    y_mid = np.array([mid_fn(y) for y in y_tab if len(y) > 1])
    y_scale = np.array([scale_fn(y) for y in y_tab if len(y) > 1])
    xb = np.array([xb[i] for i in range(len(xb)) if len(y_tab[i]) > 1])
    return xb, y_mid, y_scale
