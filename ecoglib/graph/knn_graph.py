import numpy as np
import scipy.sparse as sparse

def gauss_affinity(d_sq, sig_sq):
    """
    The Gaussian affinity function is defined as

    a(i,j) = exp{-d(i,j)**2/(2*sig**2)}

    Parameters
    ----------

    d_sq: ndarray
      A flat array of all d(i,j)**2 values to compute
    sig_sq: float or array
      The characteristic distance scale, or possibly an adaptive scale
      for to each (i,j) pair, listed in the same order as d_sq
    """
    return np.exp(-d_sq/(2*sig_sq))


def knn_graph(neighbors, dists=None, scale=1.0, auto_scale=0, mutual=False):
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
      a neighbor of i. (NOT YET IMPLEMENTED)

    """

    Nk = set()
    n_vert = neighbors.shape[0]
    if auto_scale:
        scale = np.empty(n_vert)
    connectivity = (dists is None)
    if connectivity:
        for i, i_nb in enumerate(neighbors):
            Nk.update( ( (i,j) for j in i_nb ) )
            Nk.update( ( (j,i) for j in i_nb ) )
    else:
        for i, (i_nb, i_dist) in enumerate(zip(neighbors, dists)):
            Nk.update( ( (i, j, w) for j, w, in zip(i_nb, i_dist) ) )
            Nk.update( ( (j, i, w) for j, w, in zip(i_nb, i_dist) ) )
            if auto_scale:
                scale[i] = i_dist[auto_scale]

    if mutual:
        # Prune the kNN set to include pairs of vertices that
        # are mutually k-nearest. Note that this does not change
        # the auto-tuned characteristic distance for each point.
        # This should exhibit the property of small clusters
        # and outlier points having relatively weak connectivity,
        # since their surviving edge-connected neighbors may be
        # significantly closer than the ith neighbor that defines
        # the characteristic distance.
        if connectivity:
            Nk_sym = set( ( (i,j) for (i,j) in Nk if (j,i) in Nk ) )
        else:
            Nk_sym = set( ( (i,j,w) for (i,j,w) in Nk if (j,i,w) in Nk ) )
        Nk = Nk_sym

    if connectivity:
        csr_ind = np.array( [ [i,j] for i,j in Nk ] )
        csr_dat = np.ones(len(Nk))
    else:
        csr_ind = np.array( [ [i,j,w] for i,j,w in Nk ] )
        csr_dist = csr_ind[:,2]
        csr_ind = csr_ind[:,:2].astype('i')
        if auto_scale:
            sig_sq = np.take(scale, csr_ind)
            sig_sq = sig_sq[:,0] * sig_sq[:,1]
        else:
            sig_sq = scale**2
        csr_dat = gauss_affinity(csr_dist**2, sig_sq)
    W = sparse.csr_matrix((csr_dat, csr_ind.T), shape=(n_vert, n_vert))
    return W
