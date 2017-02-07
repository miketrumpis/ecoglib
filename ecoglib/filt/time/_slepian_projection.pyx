# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as np
cimport cython

def lowpass_moving_projection(
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=2] dpss,
        np.ndarray[np.float64_t, ndim=1] weights_f,
        np.ndarray[np.float64_t, ndim=1] weights_n,
        np.ndarray[np.float64_t, ndim=1] yp
        ):

    """Calculate one lowpass moving projection-reconstruction.

    Cython helper function for moving_projection() method.

    Parameters
    ----------

    x : ndarray, length-M
        Input signal.
    dpss : ndarray, K x N
        Discrete prolate spheroidal sequences (projection operator)
    weights_f : ndarray, length-K
        Reconstruction weights for the DPSS expansion.
    weights_n : ndarray, length-N
        Temporal weights for the moving-projection filter.
    yp : ndarray, length-M
        Array to store the filtered signal.

    """

    #cdef int R = x.shape[0]
    cdef int M = x.shape[0]

    cdef int K = dpss.shape[0]
    cdef int N = dpss.shape[1]

    # other temps
    cdef int k, b, n, t
    cdef float y, y_nb

    # pad x with N-1 zeros beginning and end
    cdef np.ndarray[np.float64_t, ndim=1] xp = np.zeros( (M + 2*N - 2,), 'd' )
    b = 0
    while b < M:
        xp[b + N - 1] = x[b]
        b += 1

    # calculate y_k(b) = inner( dpss(k), x(b) )
    # = sum_n dpss(k,n) * x(b+n) n = 0, N-1
    cdef np.ndarray[np.float64_t, ndim=2] Y = np.empty( (K, M + N - 1), 'd' )
    for k in xrange(K):
        for b in xrange(M + N - 1):
            y = 0
            for n in xrange(N):
                y += dpss[k,n] * xp[b + n]
            Y[k,b] = y

    # recreate Y(n, b) = inner( dpss(:,n), Y(:, b) )
    # and y(t) = sum( Y(n, b) * win(n) ) for b = t - N + 1, b = t

    #t = N-1
    t = 0
    while t < M: #M + N - 1:
        # e.g. the first valid reconstruction point y(N-1) of x(N-1)
        # was touched by:
        # * dpss(N-1) (block 0)
        # * dpss(N-2) (block 1)
        # ...
        # * dpss(0) (block N-1)
        #b = t - N + 1 # + (N - 1)
        b = t
        #while b <= t:
        while b < t + N:
            y_nb = 0
            for k in xrange(K):
                #y_nb += dpss[k, t - b] * Y[k, b] * weights_f[k]
                y_nb += dpss[k, t + N - 1 - b] * Y[k, b] * weights_f[k]

            yp[t] += y_nb * weights_n[t + N - 1 - b]
            b += 1
        t += 1
    #return yp
    
                

    
    
## def bandpass_moving_projection(
##         np.ndarray[np.float64_t, ndim=1] x,
##         np.ndarray[np.float64_t, ndim=2] dpss,
##         np.ndarray[np.float64_t, ndim=1] weights_f,
##         np.ndarray[np.float64_t, ndim=1] weights_n,
##         f0
##         ):
##     pass
