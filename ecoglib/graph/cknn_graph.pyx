import numpy as np
import scipy.sparse as sparse

cimport numpy as np
cimport cython
from kernels import gauss_affinity

@cython.boundscheck(False)
@cython.wraparound(False)
def knn_graph(
        np.ndarray[np.int32_t, ndim=2] neighbors,
        np.ndarray[np.float64_t, ndim=2] dists=None,
        scale=1.0, auto_scale=0, mutual=False
        ):
    """
    Build a sparse adjacency matrix of a k-nearest neighbor graph.
    By common definition, this graph includes all vertices

    { (i,j) : i \in N_k(j) OR j \in N_k(i) }

    To construct the mutual k-nearest neighbor graph, see the
    "mutual" parameter below.

    Self-loops are not restricted, and will be encoded in the graph
    if the first column of neighbors is identical to the index order.

    Parameters
    ----------

    neighbors: ndarray (n_vertices, k)
      The vertices are ordered by index, and the values of each
      row indicate the index of k-nearest neighboring vertices.

    dists: ndarray (n_vertices, k)
      The optional distances to the k-nearest neighbors of each vertex.
      If given, the graph is weighted. Otherwise, it is a connectivity
      graph (binary edge weights).

    scale: float (Default 1.0)
      The characteristic distance scale used in the Gaussian affinity
      kernel. Alternatively use auto_scale for adaptive scale.

    auto_scale: int (Default 0)
      If distances are given, then automatically tune the characteristic
      distance scale between vertices based on their respective ith
      nearest neighbors. The given value of auto_scale is used as "i".

    mutual: boolean (Default False)
      Construct a mutual k-nearest neighbors graph, which includes
      all vertices (i,j) such that i is a neighbor of j AND j is
      a neighbor of i.

    """

    cdef int n_vert = neighbors.shape[0]
    cdef int n_nb = neighbors.shape[1]
    # check for whether the nearest-neighbors and distance tables
    # include self-references
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
        if auto_scale >= 0:
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

    if mutual:
        # find the intersection (in the lower triangle) of the two
        # connectivity maps, then mask out the lower triangle edge
        # weights and symmetrize the weights.
        W_mask = Wl_mask.multiply(Wu_mask.T)
        W = W_mask.multiply(W_lower)
        W = W + W.T
    else:
        # add the upper and lower triangles, plus the transposes
        # of these triangles. Then subtract out the edge weights
        # that are double-counted
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
