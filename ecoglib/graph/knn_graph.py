import numpy as np
import scipy.sparse as sparse

def knn_graph(neighbors, dists=None):
    Nk = set()
    # if no self-loops, then zip on (neighbors[:,1:], dists[:,1:])
    if dists is None:
        for i, i_nb in enumerate(neighbors[:,1:]):
            Nk.update( ( (i,j) for j in i_nb ) )
            Nk.update( ( (j,i) for j in i_nb ) )
    else:
        for i, (i_nb, i_dist) in enumerate(zip(neighbors[:,1:], dists[:,1:])):
            Nk.update( ( (i, j, w) for j, w, in zip(i_nb, i_dist) ) )
            Nk.update( ( (j, i, w) for j, w, in zip(i_nb, i_dist) ) )

    ## Nk_sym = set( ( (i,j,d) for (i,j,d) in Nk if (j,i,d) in Nk ) )
    ## Nk = Nk_sym

    if dists is None:
        csr_ind = np.array( [ [i,j] for i,j in Nk ] )
        csr_dat = np.ones(len(Nk))
    else:
        csr_ind = np.array( [ [i,j,w] for i,j,w in Nk ] )
        csr_dat = np.exp(-csr_ind[:,-1]**2)
        csr_ind = csr_ind[:,:2]
    npts = len(neighbors)
    W = sparse.csr_matrix((csr_dat, csr_ind.T), shape=(npts, npts))
    return W
