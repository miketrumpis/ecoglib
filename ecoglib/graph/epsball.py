import numpy as np
import scipy.sparse as sparse

from kernels import gauss_affinity

def eps_graph(dists, nbs, eps, sigma_sq=1.0):
    """
    Construct a graph whose edges are defined by the relationship

    E = { (i,j) : d(i,j) < eps }, d(i,j) = Euclidean dist

    By default, the edges weights are calculated according to the
    Gaussian kernel

    W(i,j) = exp{-d(i,j)**2 / (2*sigma_sq)}

    If sigma_sq is 0, then the connectivity graph is created instead.

    """


    n_vert, n_samps = dists.shape
    ix, jx = np.where( dists < eps )

    nz_idx = ix*n_samps + jx
    nz_nbs = np.take(nbs, nz_idx)

    idxptr = np.where(np.diff(ix) > 0)[0]
    idxptr += 1
    idxptr = np.r_[0, idxptr, len(ix)]

    if sigma_sq > 0:
        eps_weights = np.take(dists, nz_idx)
        eps_weights = gauss_affinity(eps_weights**2, sigma_sq)
    else:
        eps_weights = np.ones(jx.shape, 'd')

    W = sparse.csr_matrix(
        (eps_weights, nz_nbs, idxptr), (n_vert, n_vert), dtype='d'
        )
    return W



