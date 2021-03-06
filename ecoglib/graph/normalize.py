import numpy as np
import scipy.sparse as sparse

def degree_matrix(W, p=1):
    """
    Computes a diagonal matrix G of the form

    G_ii = (D_ii) ^ {p}

    where D is the degree matrix of adjacency matrix W.
    """
    deg = W.sum(1)
    diags = np.power(deg.A.T, p)
    return sparse.dia_matrix( (diags, [0]), W.shape )

def laplacian(W):
    """
    Computes the graph laplacian:

    L = D - W

    where D = diag(sum(W, axis=1))
    """
    deg = W.sum(1)
    L = sparse.dia_matrix( (deg.A.T, [0]), W.shape ) - W
    return L

def markov(W):
    """
    Computes the markov normalization:

    P = D^(-1) * W

    where D = diag(sum(W, axis=1))
    """
    #deg = W.sum(1)
    #Di = sparse.dia_matrix( (1/deg.A.T, [0]), W.shape )
    Di = degree_matrix(W, p=-1)
    return Di*W

def normalized_laplacian(W):
    """
    Computes the normalized graph laplacian:

    L = I - D^(-1) * W

    where D = diag(sum(W, axis=1))
    """
    P = markov(W)
    L = sparse.dia_matrix( (np.ones((1,npts), 'd'), [0]), W.shape ) - P
    return L

def bimarkov(W):
    """
    Computes the so-called "bimarkov" normalization:

    P = D^(-1/2) * W * D^(-1/2)

    where D = diag(sum(W, axis=1))
    """
    ## deg = W.sum(1)
    ## Di = sparse.dia_matrix( (1/np.sqrt(deg.A.T), [0]), W.shape )
    Di = degree_matrix(W, p=-1/2.)
    return Di*(W*Di)

def anisotropic(W, alpha=0.5):
    """
    Computes the anisotropic diffusion normalization. The symmetic
    edge weights are defined by a kernel

    W_{i,j} = K(x_i, x_j)

    Also,

    p(x_i) = sum_j K(x_i, x_j)

    This method computes the new graph whose edge weights are defined
    by a reweighted anisotropic diffusion kernel matrix

    Wa_{i,j} = K(x_i, x_j) / (p(x_i)p(x_j))^(\alpha)

    This method would normally be followed by computing the markov
    normalization of the anisotropic diffusion graph.

    M = D^(-1) * Wa

    Note: M is adjoint to the following symmetric matrix

    Ms = D^(1/2) * M * D^(-1/2) = D^(-1/2) * Wa * D^(-1/2)

    which is the bimarkov normalization of Wa. Since it can be more
    stable to compute eigen-values/vectors of the symmetric matrix,
    one may wish to work with the bimarkov normalized matrix.
    In this case, the eigenvectors {u} of Ms relate
    to the eigenvectors {v} of M by:

    u = D^(1/2)*v

    R. Coifman et al, "Diffusion maps, reduction coordinates and low
    dimensional representation of stochastic systems,"
    SIAM Multiscale modeling and simulation, vol. 7, no. 2,
    pp. 842-864, 2008.

    """

    # W better be symmetric
    if np.abs(alpha) < 1e-8:
        return markov(W)
    deg = np.power(W.sum(0).A[0], alpha)
    ix, jx = W.nonzero()
    px = np.take(deg, ix)
    px *= np.take(deg, jx)

    if sparse.issparse(W):
        weights = W.data / px
        return sparse.csr_matrix((weights, W.indices, W.indptr), W.shape)
    else:
        Wa = np.matrix( (W.A.ravel() / px).reshape(W.shape) )
        return Wa
        


