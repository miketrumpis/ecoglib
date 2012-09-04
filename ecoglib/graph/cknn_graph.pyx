import numpy as np
import scipy.sparse as sparse

cimport numpy as np
from knn_graph import gauss_affinity

def knn_graph(
        np.ndarray[np.int32_t, ndim=2] neighbors,
        np.ndarray[np.float64_t, ndim=2] dists=None,
        scale=1.0, auto_scale=0, mutual=False):

    cdef int n_vert = neighbors.shape[0]
    cdef int n_nb = neighbors.shape[1]
    cdef int diag = 1 if (neighbors[0,0] == 0) else 0
    cdef int weighted = 0 if (dists is None) else 1

    cdef np.ndarray[np.int32_t, ndim=1] u_idx_ptr = \
      np.zeros((n_vert+1,), 'i')
    cdef np.ndarray[np.int32_t, ndim=1] l_idx_ptr = \
      np.zeros((n_vert+1,), 'i')
    cdef np.ndarray[np.float64_t, ndim=1] ascale = np.ones((n_vert,), 'd')

    # upper bounds for the size of the adjacency lists, etc
    cdef np.ndarray[np.float64_t, ndim=1] u_weight = \
      np.empty( (neighbors.size,), 'd' )
    cdef np.ndarray[np.float64_t, ndim=1] l_weight = \
      np.empty( (neighbors.size,), 'd' )

    cdef np.ndarray[np.int32_t, ndim=1] u_idx = \
      np.empty( (neighbors.size,), 'i' )
    cdef np.ndarray[np.int32_t, ndim=1] l_idx = \
      np.empty( (neighbors.size,), 'i' )



    # form upper and lower index sets and weights

    cdef int m, n
    cdef int u_idx_cnt = 0
    cdef int l_idx_cnt = 0
    cdef int nabe
    cdef float wt
    for m in xrange(n_vert):
        u_idx_ptr[m] = u_idx_cnt
        l_idx_ptr[m] = l_idx_cnt
        for n in xrange(diag, n_nb):
            nabe = neighbors[m,n]
            if weighted:
                wt = dists[m,n]
            else:
                wt = 1
            if nabe > m:
                u_idx[u_idx_cnt] = nabe
                u_weight[u_idx_cnt] = wt
                u_idx_cnt += 1
            else:
                l_idx[l_idx_cnt] = nabe
                l_weight[l_idx_cnt] = wt
                l_idx_cnt += 1
            if n==auto_scale:
                ascale[m] = wt
        # end for n
        u_idx_ptr[m+1] = u_idx_cnt
        l_idx_ptr[m+1] = l_idx_cnt
    # end for m
    u_idx = u_idx[:u_idx_cnt]
    u_weight = u_weight[:u_idx_cnt]
    l_idx = l_idx[:l_idx_cnt]
    l_weight = l_weight[:l_idx_cnt]
    # auto-scale distances and compute kernel values
    if weighted:
        if auto_scale:
            u_sig_sq = np.repeat(ascale, np.diff(u_idx_ptr))
            u_sig_sq *= np.take(ascale, u_idx)
            l_sig_sq = np.repeat(ascale, np.diff(l_idx_ptr))
            l_sig_sq *= np.take(ascale, l_idx)
        else:
            u_sig_sq = scale**2; l_sig_sq = scale**2
        u_weight = gauss_affinity(u_weight**2, u_sig_sq)
        l_weight = gauss_affinity(l_weight**2, l_sig_sq)

    W_lower = sparse.csr_matrix(
        (l_weight, l_idx, l_idx_ptr), (n_vert, n_vert), dtype='d'
        )
    W_upper = sparse.csr_matrix(
        (u_weight, u_idx, u_idx_ptr), (n_vert, n_vert), dtype='d'
        )

    Wl_mask = sparse.csr_matrix(
        (np.ones_like(l_weight), l_idx, l_idx_ptr),
        (n_vert, n_vert), dtype='d'
        )
    Wu_mask = sparse.csr_matrix(
        (np.ones_like(u_weight), u_idx, u_idx_ptr),
        (n_vert, n_vert), dtype='d'
        )

    Wl_dbl = Wl_mask.multiply(W_upper.T)
    Wu_dbl = Wu_mask.multiply(W_lower.T)

    W = W_lower + W_upper
    W = W + W_lower.T
    W = W + W_upper.T
    W = W - Wl_dbl
    W = W - Wu_dbl
    if diag:
        W_diag = sparse.csr_matrix(
            (np.ones(n_vert), np.arange(n_vert), np.arange(n_vert+1)),
            (n_vert, n_vert), 'd'
            )
        W = W + W_diag
    return W
