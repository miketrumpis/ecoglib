"""Cython-ized semivariance calculations"""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def triu_diffs(np.ndarray[np.float64_t, ndim=2] x, axis=0):

	while axis < 0:
		axis += x.ndim
	N = x.shape[axis]
	P = x.shape[1-axis]
	n_pairs = N * (N-1) / 2
	cdef np.ndarray[np.float64_t, ndim=2] pairs = np.zeros( (n_pairs, P), 'd' )
	cdef int i, j, k, m
	m = 0
	for i in xrange(N):
		for j in xrange(i+1, N):
			for k in xrange(P):
				if axis == 0:
					pairs[m, k] = x[i, k] - x[j, k]
				else:
					pairs[m, k] = x[k, i] - x[k, j]
			m += 1

	return pairs

	
