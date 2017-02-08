# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
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

@cython.boundscheck(False)
def bandpass_moving_projection(
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=2] dpss,
        np.ndarray[np.float64_t, ndim=1] weights_f,
        np.ndarray[np.float64_t, ndim=1] weights_n,
        np.ndarray[np.float64_t, ndim=1] yp_r,
        np.ndarray[np.float64_t, ndim=1] yp_i,
        f0, baseband=True
        ):

    """Calculate one bandpass moving projection-reconstruction.

    Cython helper function for moving_projection() method. This method
    projects x onto the subspace spanned by dpss, but shifted in
    frequency to f0.

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
    yp_r : ndarray, length-M
        Array to store the filtered signal, or real-part if baseband is True
    yp_i : ndarray, length-M
        Array to store the imaginary part of the baseband signal.
    f0 : float
        Center frequency.
    baseband : bool (default True)
        Reconstruct signal at baseband (centered at DC) or centered at f0.

    """
    
    cdef int M = x.shape[0]

    cdef int K = dpss.shape[0]
    cdef int N = dpss.shape[1]

    # other temps
    cdef int k, b, n, t
    cdef float y_re, y_im, y_nb, cs_t, sn_t

    # pad x with N-1 zeros beginning and end
    cdef np.ndarray[np.float64_t, ndim=1] xp = np.zeros( (M + 2*N - 2,), 'd' )
    b = 0
    while b < M:
        xp[b + N - 1] = x[b]
        b += 1

    # create the "complex" exponentials at f0
    cdef np.ndarray[np.float64_t, ndim=1] sn = np.empty( (N,), 'd' )
    cdef np.ndarray[np.float64_t, ndim=1] cs = np.empty( (N,), 'd' )
    for n in xrange(N):
        sn[n] = np.sin(2*np.pi*f0*n)
        cs[n] = np.cos(2*np.pi*f0*n)
            
    # calculate y_k(b) = inner( dpss*(k), x(b) )
    # = sum_n dpss(k,n) * x(b+n) n = 0, N-1
    cdef np.ndarray[np.float64_t, ndim=2] Y_re = np.empty( (K, M + N - 1), 'd' )
    cdef np.ndarray[np.float64_t, ndim=2] Y_im = np.empty( (K, M + N - 1), 'd' )
    for k in xrange(K):
        for b in xrange(M + N - 1):
            y_re = 0
            y_im = 0
            for n in xrange(N):
                y_re += dpss[k,n] * xp[b + n] * cs[n]
                y_im += dpss[k,n] * xp[b + n] * -sn[n]
            Y_re[k,b] = y_re
            Y_im[k,b] = y_im

    #       
    # Recreate Y(n, b) = y_k(b)-dot-dpss(:,n) + y_k*(b)-dot-dpss*(:,n)
    #                  = 2 * Re{ y_k(b)-dot-dpss(:,n) }
    # If bandpass reconstruction (dpss includes complex exponential),
    # Y(n, b) = Re{y_k(b)}-dot-Re{dpss(:,n)} - Im{y_k(b)}-dot-Im{dpss(:,n)}
    #
    # Finally, y(t) = sum( Y(n, b) * win(n) ) for b = t - N + 1, b = t
    #
    # If baseband reconstruction, first reconstruct the complex-valued
    # single-side band signal, then multiply by a complex exponential
    # to shift to baseband. The SSB signal has to be reconstructed
    # first so that the phases of the multiple blocks are aligned.
    # 

    t = 0
    if baseband:
        while t < M:
            # e.g. the first valid reconstruction point y(N-1) of x(N-1)
            # was touched by:
            # * dpss(N-1) (block 0)
            # * dpss(N-2) (block 1)
            # ...
            # * dpss(0) (block N-1)
            b = t
            yp_r[t] = 0
            yp_i[t] = 0
            while b < t + N:
                y_re = 0
                y_im = 0
                n = t + N - 1 - b
                for k in xrange(K):
                    y_re += dpss[k, n] * Y_re[k, b] * cs[n] * weights_f[k]
                    y_re -= dpss[k, n] * Y_im[k, b] * sn[n] * weights_f[k]
                    y_im += dpss[k, n] * Y_re[k, b] * sn[n] * weights_f[k]
                    y_im += dpss[k, n] * Y_im[k, b] * cs[n] * weights_f[k]
                yp_r[t] += y_re * weights_n[n]
                yp_i[t] += y_im * weights_n[n]
                b += 1
            # need to multiply by 2exp{-i*2PI*f*t} ? 
            cs_t = np.sqrt(2) * np.cos(2*np.pi*f0*t)
            sn_t = -np.sqrt(2) * np.sin(2*np.pi*f0*t)
            y_re = (yp_r[t] * cs_t - yp_i[t] * sn_t)
            y_im = (yp_r[t] * sn_t + yp_i[t] * cs_t)
            yp_r[t] = y_re
            yp_i[t] = y_im
            t += 1
    else:
        while t < M:
            b = t
            yp_r[t] = 0
            while b < t + N:
                y_nb = 0
                n = t + N - 1 - b
                for k in xrange(K):
                    y_nb += dpss[k, n] * cs[n] * Y_re[k, b] * weights_f[k]
                    y_nb -= dpss[k, n] * sn[n] * Y_im[k, b] * weights_f[k]
                yp_r[t] += 2 * y_nb * weights_n[n]
                b += 1
            t += 1
    # done
