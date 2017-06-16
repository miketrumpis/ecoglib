import numpy as np
from ecoglib.numutil import fenced_out

__all__ = ['semivariogram']

def semivariogram(F, combs, robust=True, trimmed=True):
    # F is an n_site field of values
    # combs is a channel combination bunch
    x = np.unique(combs.dist)
    sv = np.empty_like(x)
    for n in xrange(len(x)):
        m = combs.dist == x[n]
        x_s1 = F[ combs.p1[m] ]
        x_s2 = F[ combs.p2[m] ]
        if trimmed:
            # trim outliers from the population of samples at this lag
            m = fenced_out( np.r_[x_s1, x_s2] ).reshape(2, len(x_s1))
            m = m[0] & m[1]
            # keep only pairs where both samples are inliers
            x_s1 = x_s1[m]
            x_s2 = x_s2[m]
        Nd = len(x_s1)
        if not Nd:
            sv[n] = np.nan
            continue
        if robust:
            avg_var = np.power(np.abs( x_s1 - x_s2 ), 0.5).mean() ** 4
            sv[n] = avg_var / 2 / (0.457 + 0.494 / Nd)
        else:
            sv[n] = 0.5 * np.mean( (x_s1 - x_s2)**2 )
    return x, sv

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
