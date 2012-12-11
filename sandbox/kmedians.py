from __future__ import division
import numpy as np
import sklearn.cluster.k_means_ as km

def _slow_disp(X):
    # calculate pairwise distances the long way
    Xd = X[:,None,:] - X[None,:,:]
    return np.abs(Xd).sum()

def _fast_disp(X):
    # calculate pairwise distance the fast way
    Xd = np.sort(X, axis=0)
    Xd = np.diff(Xd, axis=0)
    nd = Xd.shape[0]
    cumidx = np.arange(1, nd+1).reshape(nd, 1)
    np.multiply(Xd, cumidx, Xd)
    np.multiply(Xd, cumidx[::-1], Xd)
    return 2*np.sum(Xd)

def kmedians(X, k, restarts=2, tol=1e-4, max_iter = 100):

    m, n = X.shape

    #best_locs, best_inertia, best_labeling
    best_locs = None
    best_inertia = 1e10
    best_labeling = None

    # some pre-allocated memory
    X_scratch = np.empty_like(X)
    group_inertia = np.zeros(k)
    X_dist = np.empty( (m, k), 'd' )
    np.power(X, 2, X_scratch)
    sq_dist = np.sum(X_scratch, axis=1)

    for n in xrange(restarts):
        locs = km.k_init(X, k, x_squared_norms=sq_dist)

        # perform modified Lloyd's algorithm to
        # 1) assign points to current locations
        # 2) update locations by finding median of provisional clusters
        # repeat steps until the "inertia" value converges
        old_locs = np.ones_like(locs) * 1e10
        for i in xrange(max_iter):
            # find labels
            for r in xrange(k):
                np.subtract(X, locs[r], X_scratch)
                np.abs(X_scratch, X_scratch)
                X_dist[:,r] = np.sum(X_scratch, axis=1)

            assignments = np.argmin(X_dist, axis=1)

            # update medians
            for r in xrange(k):
                # indicator set
                Cr = (assignments == r)
                nr = np.sum(Cr)
                cluster = X[ Cr ]
                cluster.sort(axis=0)
                if (nr % 2):
                    locs[r] = cluster[ nr // 2 ]
                else:
                    midx = nr // 2
                    locs[r] = (cluster[midx-1] + cluster[midx]) / 2

            movement = np.sum( (old_locs - locs)**2 )
            if movement < tol:
                break
            old_locs = locs.copy()

        for r in xrange(k):
            Cr = (assignments == r)
            nr = np.sum(Cr)
            cluster = X[ Cr ]
            cluster.sort(axis=0)
            # do quick O(nr) calculation of
            # within-cluster dispersion
            cdiff = np.diff(cluster, axis=0)
            cumidx = np.arange(1,nr).reshape(nr-1, 1)
            np.multiply(cdiff, cumidx, cdiff)
            np.multiply(cdiff, cumidx[::-1], cdiff)
            group_inertia[r] = 2*np.sum(cdiff)

        inertia = np.sum(group_inertia)
        if inertia < best_inertia:
            best_inertia = inertia
            best_locs = locs.copy()
            best_labeling = assignments.copy()

    return best_locs, best_labeling, best_inertia


if __name__=='__main__':
    n_pts = 50
    n_bkg = 100
    model_locs = (
        np.array([5,0]), np.array([-5,-5])
        )

    import scipy.stats.distributions as dist
    import matplotlib.pyplot as pp
    # create a test set with points scattered around the model locations
    # with a heavy tailed dist (cauchy). Also sprinkle in some
    # uniformly distributed "background" points as outliers
    test_set1 = dist.cauchy.rvs(loc=model_locs[0], size=(n_pts,2))
    test_set2 = dist.cauchy.rvs(loc=model_locs[1], size=(n_pts,2))
    test_set1 = dist.laplace.rvs(loc=model_locs[0], scale=(2,2), size=(n_pts,2))
    test_set2 = dist.laplace.rvs(loc=model_locs[1], scale=(2,2), size=(n_pts,2))

    d = np.vstack( (test_set1, test_set2) )
    dmx = d.max(axis=0); dmn = d.min(axis=0)

    bkg_set = dist.uniform.rvs(loc=dmn, scale=dmx-dmn, size=(n_bkg,2))

    d = np.vstack( (d, bkg_set) )

    pp.figure()
    true_labels = np.r_[np.zeros(n_pts), np.ones(n_pts), np.ones(n_bkg)*0.5]
    pp.scatter(d[:,0], d[:,1], c=true_labels)

    c1, g1, d1 = km.k_means(d, 2)

    if np.linalg.norm(c1[0] - model_locs[0]) > \
       np.linalg.norm(c1[1] - model_locs[0]):
       c1_loc_err = np.linalg.norm(c1[0] - model_locs[1])**2 + \
         np.linalg.norm(c1[1] - model_locs[0])**2
       g1_1 = (g1==1)
       g1[g1_1] = 0
       g1[~g1_1] = 1
    else:
       c1_loc_err = np.linalg.norm(c1[0] - model_locs[0])**2 + \
         np.linalg.norm(c1[1] - model_locs[1])**2
    g1_1 = (g1==1)
    cls_err1 = (g1_1[:n_pts].sum() + (~g1_1[n_pts:2*n_pts]).sum())
    cls_err1 = cls_err1 / float(2*n_pts)

    c2, g2, d2 = kmedians(d, 2, restarts=10)

    if np.linalg.norm(c2[0] - model_locs[0]) > \
       np.linalg.norm(c2[1] - model_locs[0]):
       c2_loc_err = np.linalg.norm(c2[0] - model_locs[1])**2 + \
         np.linalg.norm(c2[1] - model_locs[0])**2
       g2_1 = (g2==1)
       g2[g2_1] = 0
       g2[~g2_1] = 1
    else:
       c2_loc_err = np.linalg.norm(c2[0] - model_locs[0])**2 + \
         np.linalg.norm(c2[1] - model_locs[1])**2
    g2_1 = (g2==1)
    cls_err2 = (g2_1[:n_pts].sum() + (~g2_1[n_pts:2*n_pts]).sum())
    cls_err2 = cls_err2 / float(2*n_pts)


    pp.figure()
    pp.scatter(d[:,0], d[:,1], c=g1)
    pp.title('kmeans')
    pp.figure()
    pp.scatter(d[:,0], d[:,1], c=g2)
    pp.title('kmedians')

    print 'kmeans))) loc err: %1.4f'%c1_loc_err, 'cls err: %1.2f'%cls_err1
    print 'kmedos))) loc err: %1.4f'%c2_loc_err, 'cls err: %1.2f'%cls_err2

    pp.show()
