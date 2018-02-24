import numpy as np
from nitime.algorithms import dpss_windows
from numpy.lib.stride_tricks import as_strided
from ecoglib.util import input_as_2d
from sandbox.array_split import split_at

__all__ = ['slepian_projection', 'moving_projection']

@input_as_2d(out_arr=0)
def slepian_projection(
        data, BW, Fs, Kmax=None, w0=0, baseband=False, onesided=False,
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
        Half bandwidth of the lowpass projection (with respect to Fs).
        (In other words, BW sets the corner frequency of a lowpass.)
    Fs : float
        Sampling rate of the timeseries
    Kmax : int
        Highest order Slepian sequence to use (default is K=2TW).
    w0 : float
        Center frequency of the bandpass (defaut is lowpass)
    baseband : bool
        Reconstruct bandpass signal in baseband
    onesided : bool
        If making a centered bandpass projection, do one-sided or not.
        For example, a one-sided baseband reconstruction is similar
        to a combination of bandpass filtering and Hilbert transform.
    dpss : ndarray
        Pre-computed Slepian functions for given time-bandwidth product
    save_dpss : bool
        Return the Slepian functions for current time-bandwidth product
        along with the filtered timeseries
    """
    
    nchan, npts = data.shape
    if dpss is None:
        # find NW which is also TW
        T = npts / Fs
        # round to the nearest multiple of 1/2
        TW = int( round( 2 * T * BW ) / 2.0 )
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
        # this really should already be normalized
        nrm = np.sqrt( ( dpss_pf * dpss_pf.conj() ).sum(-1) )
        #dpss_nf = np.exp(-2j * np.pi * w0 * t / Fs) * dpss
        #nrm = np.sqrt( ( dpss_nf * dpss_nf.conj() ).sum(-1) )
        #dpss_nf /= nrm[:, None]
        dpss_pf /= nrm[:, None]
        wp = data.dot( dpss_pf.conj().T )
        #wn = data.dot( dpss_nf.conj().T )
        if baseband:
            #bp = ( wp.dot( dpss ) + wn.dot( dpss ) ).real.copy()
            bp = 2 * wp.dot( dpss )
        else:
            bp = 2 * wp.dot( dpss_pf )
            #bp = ( wp.dot( dpss_pf ) + wn.dot( dpss_nf ) ).real.copy()
        if not onesided:
            bp = bp.real
    if save_dpss:
        return bp, dpss
    return bp

def _stagger_array(x, N):
    """
    Stack x into N rows. The first row is padded with N-1 zeros. The
    remaining N-1 rows are circularly shifted one sample back. If x
    is 2D, then the stacked matrices comprise the first two dimensions
    of the resulting array.
    """
    shp = x.shape
    s = shp[:-1]
    M = shp[-1]
    zp = np.zeros( s + (N-1,) )
    x_pad = np.concatenate( (zp, x, zp), axis=-1 )
    B = x_pad.dtype.itemsize
    st = x_pad.strides[:-1]
    x_strided = as_strided(x_pad, shape=s + (N, M+N-1), strides=st + (B, B))
    if x_strided.ndim > 2:
        return x_strided.transpose(1, 2, 0).copy()
    return x_strided.copy()

@input_as_2d(out_arr=0)
def _moving_projection_preserve(
        x, N, BW, Fs=1.0, f0=0, Kmax=None, baseband=True,
        weight_eigen=True, window=np.hanning, 
        dpss=None, save_dpss=False
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
    x : ndarray 
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
    f0 : float (dummy var)
        Non-functional
    baseband : bool (dummy var)
        Non-functional

    Returns
    =======
    y : ndarray
        Lowpass projection of x.

    Notes
    =====
    Since this method depends on projections of staggered blocks of 
    length N, expect poorer reconstruction accuracy within N samples
    of the beginning and end of the signal window.

    Memory consumption scales rather high with many (~1000) input vectors.
    
    Method from "Projection Filters for Data Analysis", D.J. Thomson, 1996
    """

    M = x.shape[-1]
    T = N / Fs
    TW = int( round( 2 * T * BW ) / 2.0 )
    K = 2 * TW - 1
    if K < 1:
        min_bw = 0.5 / T
        err = 'BW is too small for the window size: ' \
          'minimum BW={0}'.format(min_bw)
        raise ValueError(err)
    
    if dpss is not None:
        dpss, eigs = dpss
    else:
        dpss, eigs = dpss_windows(N, TW, K)

    # X: shape (N, M + N - 1, [R]) (for R input vectors)
    # columns are the length-N windows of x beginning at offset b
    # where b in [ -(N-1), M-1 ]
    X = _stagger_array(x, N)
    # Y: shape (K, M + N - 1, [R])
    # Y is the lowpass coefficients indexed by k and b
    # Y_ij = y_i(j) for ith taper and jth block
    Y = np.tensordot(dpss, X, (1, 0))
    del X
    if weight_eigen:
        w = K * eigs / eigs.sum()
        Yh = np.tensordot( dpss.T * w, Y, (1, 0) )
    else:
        Yh = np.tensordot(dpss.T, Y, (1, 0))
    # Yh is the projection of staggered blocks.
    # Use a strided array view to unstagger the blocks
    ndim = x.ndim
    Yh_ = np.zeros( (N, M + 2*(N-1)) + Yh.shape[2:] )
    B = Yh_.dtype.itemsize
    d = Yh.shape[2] if ndim > 1 else 1
    strides = ( (Yh_.shape[1] + 1) * d * B, d * B, B )[:ndim+1]
    v = as_strided(Yh_, shape=Yh.shape, strides=strides)
    v[:] = Yh
    # Each output sample y(t) is a weighted combination of N estimates
    # from every block in X that overlapped with x(t). The previous
    # array maneuver took these staggered estimates and aligned them into
    # the same column. The final step is to make a weighted sum of estimates.
    w = window(N)
    w /= w.sum()
    y = (Yh_[:, (N-1):-(N-1)].T * w).sum(-1)
    if save_dpss:
        return y, (dpss, eigs)
    return y

try:
    from ._slepian_projection import \
         lowpass_moving_projection, bandpass_moving_projection

    # parallel appears safe! (also input as 2d should wrap input splitting)
    @input_as_2d(out_arr=0)
    @split_at()
    def moving_projection(
            x, N, BW, Fs=1.0, f0=0, Kmax=None, baseband=True,
            weight_eigen=True, window=np.hanning, 
            dpss=None, save_dpss=False
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
        x : ndarray 
            Signal to filter.
        N : int
            Length of blocks (N < len(x))
        BW : float
            Half bandwidth of the lowpass projection (with respect to Fs).
            (In other words, BW sets the corner frequency of a lowpass.)
        Fs : float, optional
            Sampling frequency of x.
        f0 : float, optional
            Center frequency of the bandpass (defaut is 0 for lowpass)
        Kmax : int, optional
            Maximum number of basis vectors for projection. Kmax < 2*N*BW/Fs - 1
        baseband : {True | False}
            Return the complex baseband (lowpass) reconstruction of a
            bandpass signal (only applies to f0 > 0).
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

        MT: moving_projection_test.py demos various projections
        """

        M = x.shape[-1]
        T = N / Fs
        TW = int( round( 2 * T * BW ) / 2.0 )
        K = 2 * TW - 1
        if K < 1:
            min_bw = 0.5 / T
            err = 'BW is too small for the window size: ' \
              'minimum BW={0}'.format(min_bw)
            raise ValueError(err)

        # if f0 > 0, then it also has to be > BW/2
        if abs(f0) > 0:
            ## if abs(f0) < BW/2.0:
            ##     raise ValueError('A bandpass center has to satisfy abs(f0) > BW/2')
            f0 = f0 / float(Fs)
        else:
            f0 = False

        if dpss is not None:
            dpss, eigs = dpss
        else:
            dpss, eigs = dpss_windows(N, TW, K)

        if weight_eigen:
            wf = K * eigs / eigs.sum()
        else:
            wf = np.ones(K, 'd')

        wt = window(N)
        wt = wt / wt.sum()
        if f0:
            if baseband:
                y = np.zeros( x.shape, 'D' )
                y_flat = y.view(dtype='d')
                y_re = y_flat[..., 0::2]
                y_im = y_flat[..., 1::2]
            else:
                y_re = np.zeros_like(x)
                # make dummy array to satisfy Cython signature
                y_im = np.empty( (x.shape[0], 1), 'd' )
            for i in xrange(x.shape[0]):
                bandpass_moving_projection(
                    x[i].astype('d'), dpss, wf, wt, y_re[i], y_im[i],
                    f0, baseband=baseband
                    )
            if not baseband:
                y = y_re
        else:
            y = np.zeros(x.shape, 'd')
            for i in xrange(x.shape[0]):
                lowpass_moving_projection(x[i].astype('d'), dpss, wf, wt, y[i])
        if save_dpss:
            return y, (dpss, eigs)
        return y

except ImportError:
    moving_projection = _moving_projection_preserve

