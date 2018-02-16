"""Cython-ized bispectrum calculations"""

import numpy as np
cimport numpy as np
cimport cython

cdef inline void cplx_prod( double a, double b, double c, double d,
                            double *re, double *im ):
    re[0] = a*c - b*d
    im[0] = a*d + b*c
    return

cdef inline void cplx_tri_prod( double a, double b,
                                double c, double d,
                                double e, double f,
                                double *re, double *im ):
    # multiply (a + ib)(c + id)(e + if)
    re[0] = (a*c*e) - (b*d*e) - (a*d*f) - (b*c*f)
    im[0] = (a*d*e) + (b*c*e) + (a*c*f) - (b*d*f)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_bispectrum(
        np.ndarray[np.float64_t, ndim=2] tf_re,
        np.ndarray[np.float64_t, ndim=2] tf_im
    ):
    """Calculate frequencies (f1, f2) of the bispectrum, defined as

    B(f1, f2) = E{ dX(f1)dX(f2)dX*(f1+f2) }

    Due to real symmetries in the power spectrum, we can limit
    computation to one quarter of the freq-freq grid:

    (f1, f2) : { f2 <= f1 and f1 + f2 <= min(max(freqs), 0.5) }

    Paramters
    ---------
    tf_re : ndarray (K, nf)
        real-part of multitaper complex demodulate estimates
    tf_im : ndarray (K, nf)
        imaginary-part of multitaper complex demodulate estimates
    freqs : ndarray
        frequency grid (digital units)

    Returns
    -------
    B : ndarray (K, N)
        K multitaper estimates of bispectrum at N frequency pairs.
        Each B[k] can be used in scipy.sparse.csr_matrix.
    row : ndarray (N,)
    col : ndarray (N,)
        row, column vectors to use for scipy.sparse.csr_matrix
        
    """

    cdef int K, nf
    K = tf_re.shape[0]
    nf = tf_re.shape[1]

    f1, f2 = np.meshgrid(np.arange(nf), np.arange(nf))
    cdef int N = ( (f1 >= f2) & (f1 + f2 < nf) ).sum()

    cdef np.ndarray[np.float64_t, ndim=2] B = np.zeros( (K, N*2), 'd' )
    cdef np.ndarray[np.int32_t, ndim=1] row = np.zeros( (N,), 'i' )
    cdef np.ndarray[np.int32_t, ndim=1] col = np.zeros( (N,), 'i' )
    cdef int k, i, j, n
    cdef double foo1, foo2
    foo1 = 0
    foo2 = 0
    # need to "alloc" these pointers
    cdef double *re = &foo1
    cdef double *im = &foo2
		
    # compute frequencies on the "lower" quarter, such that
    # f1 <--> col(j) and f2 <--> row(i)
    for k in xrange(K):
        n = 0
        for i in xrange(nf):
            if nf <= 2*i:
                break
            for j in xrange(i, nf-i):
                if k == 0:
                    row[n] = i
                    col[n] = j

                # B[k,n] = x(j)x(i)x*(i+j)
                cplx_tri_prod( tf_re[k, j], tf_im[k, j],
                               tf_re[k, i], tf_im[k, i],
                               tf_re[k, i+j], -tf_im[k, i+j],
                               re, im )
                B[k,2*n] = re[0]
                B[k,2*n+1] = im[0]
                ## pass
                n += 1

    return B, row, col
			
