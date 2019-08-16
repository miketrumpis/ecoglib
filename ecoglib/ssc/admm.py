import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

def shrink_thresh(x, alpha):
    # definitely operates in-place! with low memory consumption
    xl = x < -alpha
    xu = x > alpha
    x[xl] += alpha
    x[xu] -= alpha
    xl = ~xl & ~xu
    x[xl] = 0

def diag_plus_loaded_inverse(Y, lamz, rho, YYt=None, YtY=None, matrix=False):
    # return a LinearOperator that applies the inverse of
    # lamz*YtY + rho*(I + ee^T)

    m, N = Y.shape

    # if m >> N, then just compute inverse directly on N x N matrix
    if m > N:
        if YtY is None:
            YtY = Y.T.dot(Y)
        T = YtY * lamz
        T += rho
        T.flat[0:N*N:(N+1)] += rho
        return np.linalg.inv(T)

    if YYt is None:
        YYt = Y.dot(Y.T)

    Ynu = np.sum(Y, axis=1)
    Ynu

    # B is the (m, m) matrix to actually invert
    B = np.outer(Ynu, Ynu)
    B *= -1.0 / (N+1)
    B += YYt
    B /= rho
    # add 1/lamz to the diagonal
    B.flat[0:m*m:(m+1)] += 1/lamz

    Binv = np.linalg.inv(B)

    #X = Y - Ynu[:,None]/(N+1)
    X = Y.copy()
    X -= Ynu[:,None]/(N+1)
    X /= rho

    # construct 1/rho*(I - ee^t/(N+1)) - X^T * Binv * X
    # either as a dense matrix, or as an operator
    if matrix:

        # matrix products first
        T = X.T.dot(Binv)
        T = T.dot(X)

        # now invert sign and subtract constant from each entry
        T *= -1
        T -= 1/(rho*(N+1))
        # and add constant to the diagonal
        T.flat[0:N*N:(N+1)] += 1/rho
    else:

        def op(U):
            # the first action on U is to sum the columns as in
            # (e^T * U) / (rho*(N+1))

            R = U/rho
            Rsum = np.sum(R, axis=0) / (N+1)
            R -= Rsum

            # next action is to apply operators X^T * Binv * X
            R2 = X.dot(U)
            R2 = Binv.dot(R2)
            R -= X.T.dot(R2)
            return R
        T = LinearOperator( (N,N), op, matmat=op, dtype='d' )
    return T

def auto_mu(Y, YtY=None):
    if YtY is None:
        aYtY = np.abs(Y.T.dot(Y))
    else:
        aYtY = np.abs(YtY)

    m, N = Y.shape
    aYtY.flat[0:N*N:(N+1)] = 0
    muz = np.min( np.max(aYtY, axis=-1) )
    ys = np.sort(np.sum(np.abs(Y), axis=0))
    mue = ys[-2]
    return (muz, mue)


def admm_one(Y, lamz, lamr, rho, X=None, YYt=None, YtY=None, max_it = 1e3):
    # Y is (m, N) where m is feature length and N is # of pts

    # If X is provided, then X is (m, N1) and the problem is to
    # respresent X (not Y) in terms of Y and a sparse error matrix E.
    # In this case,
    # C, A are (N, N1)
    # E is (m, N1)
    # Dlta is (N, N1)
    # dlta is (N1, 1)

    m, N = Y.shape

    if X is None:
        N1 = N
    else:
        N1 = X.shape[1]
        YtX = Y.T.dot(X)

    # initialization ( >= 4 NxN matrices!!)
    C = np.zeros( (N,N1) )
    Akm1 = np.zeros( (N,N1) )
    E = np.zeros_like(Y) if X is None else np.zeros_like(X)
    Ekm1 = np.zeros_like(Y) if X is None else np.zeros_like(X)
    Dlta = np.zeros( (N,N1) )
    dlta = np.zeros( (N1,1) )
    if YtY is None:
        YtY = Y.T.dot(Y)
    rhs = np.empty( (N,N1) )

    # construct the Woodbury (or direct) inverse
    T = diag_plus_loaded_inverse(Y, lamz, rho, YYt=YYt, YtY=YtY, matrix=True)

    it = 0
    while True:
        # update A -- first compute crazy RHS matrix
        rhs[:] = YtY if X is None else YtX
        rhs -= Y.T.dot(E)
        rhs *= lamz

        # check (XXX)
        C += 1
        C *= rho

        rhs += C
        rhs -= dlta.T
        rhs -= Dlta
        # now apply T to rhs
        A = T.dot(rhs)

        # update C
        C[:] = Dlta
        C /= rho
        C += A
        shrink_thresh(C, 1/rho)
        if X is not None:
            dC = C.diagonal()
            C.flat[0:N*N:(N+1)] -= dC

        # update E
        E = Y.dot(A)
        if X is None:
            E -= Y
        else:
            E -= X
        shrink_thresh(E, lamr/lamz)

        # step up the Langrange multipliers
        Asum = np.sum(A, axis=0)
        Asum -= 1
        dlta += rho*Asum[:,None]

        tmp = A - C
        mx2 = np.max(tmp)
        tmp *= rho
        Dlta += tmp

        # check convergence criteria

        mx1 = np.max(Asum)
        # moved up above
        ## tmp /= rho
        ## mx2 = np.max(tmp)

        tmp[:] = A
        tmp -= Akm1
        mx3 = np.max(tmp)

        tmp = E.copy()
        tmp -= Ekm1
        mx4 = np.max(tmp)

        itcap = it > max_it
        errs = np.array([mx1, mx2, mx3, mx4])
        if itcap or (np.max(errs) <= 1e-4):
            break
        Akm1[:] = A
        Ekm1[:] = E
        it += 1
        print(it, errs)

    return C, E

