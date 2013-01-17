import numpy as np
from sklearn.cluster import k_means
from sandbox import kmedians as kmd

def gap_stat(x, Kmax, nsurr=20, p=2, svd=False):
    wk, r_wk = unsupervised_clusters(x, Kmax, nsurr=nsurr, p=p, svd=svd)
    lwk = np.log(wk)
    lr_wk = np.log(r_wk)
    gk = np.mean(lr_wk, axis=0) - lwk
    sk = np.std(lr_wk, axis=0) * np.sqrt(1+1/float(nsurr))
    return gk, sk

def unsupervised_clusters(x, Kmax, nsurr=20, p=2, svd=False):
    ## wk = [ k_means(x, k, n_init=10)[-1]
    ##        for k in xrange(1,Kmax+1) ]
    wk = [ pooled_dispersion(x, k, p=p, n_init=20)
           for k in xrange(1, Kmax+1) ]
    wk = np.array(wk)
    ## bbox_hi = x.max(axis=0)
    ## bbox_lo = x.min(axis=0)

    if svd:
        bbox_hi, bbox_lo, vt = bounding_box(x, svd=True)
    else:
        bbox_hi, bbox_lo = bounding_box(x)

    rand_wk = np.zeros( (nsurr, Kmax) )
    for n in xrange(nsurr):
        print n
        rbox = np.random.rand(*x.shape)
        rbox *= (bbox_hi - bbox_lo)
        rbox += bbox_lo
        if svd:
            rbox = np.dot(rbox, vt)
        ## rand_wk[n] = np.array(
        ##     [ k_means(rbox, k, init='random', n_init=1)[-1]
        ##       for k in xrange(1,Kmax+1) ]
        ##       )
        rand_wk[n] = np.array(
            [ pooled_dispersion(rbox, k, p=p, n_init=1)
              for k in xrange(1, Kmax+1) ]
              )
    #e_wk = np.mean(rand_wk, axis=0)
    return wk, rand_wk

def bounding_box(x, svd=False):
    if svd:
        xd = x - x.mean(axis=0)
        [u, s, vt] = np.linalg.svd(xd, full_matrices=0)
        xd = np.dot(x, vt.T)
    else:
        xd = x

    bbox_hi = x.max(axis=0)
    bbox_lo = x.min(axis=0)
    if svd:
        return bbox_hi, bbox_lo, vt
    else:
        return bbox_hi, bbox_lo

def pooled_dispersion(x, k, p=2, **kws):
    if p==2:
        _, _, wk = k_means(x, k, **kws)
    elif p==1:
        _, _, wk = kmd.kmedians(x, k, **kws)
    else:
        raise ValueError('Only l1, l2 distance implemented')
    return wk
    ## cr = [ np.where(labels==i)[0] for i in xrange(k) ]
    ## nr = np.array( [len(c) for c in cr] )
    ## # --this is not scatter--
    ## # dr = [ np.sum((x[c] - loc)**2) for c, loc in zip(cr, locs) ]
    ## # scatter is sum of pairwise distances
    ## dr = np.empty(k)
    ## for r in xrange(k):
    ##     x_c = x[ cr[r] ]
    ##     d_ii = x_c[:,None,:] - x_c[None,:,:]
    ##     np.power(d_ii, 2, d_ii)
    ##     dr[r] = d_ii.sum()
    ## wk = np.array(dr) / (2*nr)
    ## return wk.sum()
