import numpy as np
from nitime.algorithms import dpss_windows
from numpy.lib.stride_tricks import as_strided

def slepian_projection(
        data, BW, Fs, Kmax=None, w0=0, baseband=False, 
        dpss=None, save_dpss=False, min_conc=None
        ):
    """
    Perform bandpass filtering by projection onto the bandpass space
    supported by "discrete prolate spheroidal sequences" (i.e. Slepian
    functions).

    Parameters
    ----------

    data : ndarray
        Last dimension is timeseries
    BW : float
        Bandwidth in Hz of the Slepian functions 
        (and therefore the bandpass)
    Fs : float
        Sampling rate of the timeseries
    Kmax : int
        Highest order Slepian sequence to use (default is K=2TW).
    w0 : float
        Center frequency of the bandpass (defaut is lowpass)
    baseband : bool
        Reconstruct bandpass signal in baseband
    dpss : ndarray
        Pre-computed Slepian functions for given time-bandwidth product
    save_dpss : bool
        Return the Slepian functions for current time-bandwidth product
        along with the filtered timeseries
    """
    
    shp = data.shape
    if len(shp) < 2:
        data = np.atleast_2d(data)
    if len(shp) > 2:
        data = data.reshape( shp[0] * shp[1], -1 )
    nchan, npts = data.shape
    if dpss is None:
        # find NW which is also TW
        T = npts / Fs
        # round to the nearest multiple of 1/2
        TW = round( 2 * T * BW ) / 2.0
        K = 2 * TW
        if K < 1:
            min_bw = 0.5 / T
            err = 'BW is too small for the window size: ' \
              'minimum BW={0}'.format(min_bw)
            raise ValueError(err)
        if Kmax is not None:
            K = min(Kmax, K)
        dpss, eigs = dpss_windows(npts, TW, K)
        if min_conc is not None:
            keep = eigs > min_conc
            dpss = dpss[keep]
    if w0 == 0:
        # shortcut for lowpass only
        w = data.dot( dpss.T )
        # w is (nchan x K)
        # expand Slepians as (nchan x npts)
        bp =  w.dot( dpss )
    else:
        t = np.arange(npts)
        dpss_pf = np.exp(2j * np.pi * w0 * t / Fs) * dpss
        dpss_nf = np.exp(-2j * np.pi * w0 * t / Fs) * dpss
        nrm = np.sqrt( ( dpss_nf * dpss_nf.conj() ).sum(-1) )
        dpss_nf /= nrm[:, None]
        wp = data.dot( dpss_pf.conj().T )
        wn = data.dot( dpss_nf.conj().T )
        if baseband:
            bp = ( wp.dot( dpss ) + wn.dot( dpss ) ).real.copy()
        else:
            bp = ( wp.dot( dpss_pf ) + wn.dot( dpss_nf ) ).real.copy()
    if save_dpss:
        return bp.reshape(shp), dpss
    return bp.reshape(shp)

def _stagger_array(x, N):
    """
    Stack x into N rows, with each row staggered (and zero padded)
    by one sample.
    """
    M = len(x)
    x_pad = np.r_[ np.zeros(N-1), x, np.zeros(N-1) ]
    B = x_pad.dtype.itemsize
    x_strided = as_strided(x_pad, shape=(N, M+N-1), strides=(B, B))
    return x_strided.copy()

def moving_projection(
        x, N, BW, Fs=1.0, Kmax=None, weight_eigen=True, window=np.hanning
        ):
    """
    Perform the "moving" projection filter on a (relatively short)
    signal x. The projection basis is computed for an N-dimensional 
    space, and staggered length-N blocks of x are filtered by the 
    projection. Since blocks are offset by one sample, each point in 
    the output is represented N times (with zero padding at the
    beginning and end of the signal). The final output is a weighted
    sum of these estimates. 

    Parameters
    ==========
    x : ndarray (1D)
        Signal to filter.
    N : int
        Length of blocks (N < len(x))
    BW : float
        Bandwidth of the lowpass projection (with respect to Fs).
    Fs : float, optional
        Sampling frequency of x.
    Kmax : int, optional
        Maximum number of basis vectors for projection. Kmax < 2*N*BW/Fs - 1
    weight_eigen : bool, optional
        Weight the reconstruction vectors by eigenvalue (default True).
        The lowpass energy concentration of a particular vector is
        reflected by its eigenvalue (higher is better).
    window : callable, optional
        The method, if given, returns a window the length of its argument.
        This window will be used to weight the projection values.

    Returns
    =======
    y : ndarray
        Lowpass projection of x.

    Notes
    =====
    Since this method depends on projections of staggered blocks of 
    length N, expect poorer reconstruction accuracy within N samples
    of the beginning and end of the signal window.
    
    Method from "Projection Filters for Data Analysis", D.J. Thomson, 1996

    Todo
    ====
    Adapt to n-dimensional input.
    """

    M = len(x)
    T = N / Fs
    TW = round( 2 * T * BW ) / 2.0
    K = 2 * TW - 1
    dpss, eigs = dpss_windows(N, TW, K)

    # X: shape (N, M + N - 1)
    # columns are the length-N windows of x beginning at offset b
    # where b in [ -(N-1), M-1 ]
    X = _stagger_array(x, N)
    # Y: shape (K, M + N - 1)
    # Y is the lowpass coefficients indexed by k and b
    # Y_ij = y_i(j) for ith taper and jth block
    Y = dpss.dot(X)
    if weight_eigen:
        w = eigs / eigs.sum()
        Yh = (dpss.T * eigs).dot(Y)
    else:
        Yh = dpss.T.dot(Y)
    # Yh is the projection of staggered blocks.
    # Use a strided array view to unstagger the blocks
    Yh_ = np.zeros( (N, M + 2*(N-1)) )
    B = Yh_.dtype.itemsize
    v = as_strided(Yh_, shape=Yh.shape, strides=( (Yh_.shape[1]+1) * B, B ))
    v[:] = Yh
    # Each output sample y(t) is a weighted combination of N estimates
    # from every block in X that overlapped with x(t). The previous
    # array maneuver took these staggered estimates and aligned them into
    # the same column. The final step is to make a weighted sum of estimates.
    w = window(N)
    w /= w.sum()
    y = (Yh_[:, (N-1):-(N-1)] * w[:,None]).sum(0)
    return y
    
