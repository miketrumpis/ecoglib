from __future__ import division
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import nitime.algorithms as alg
import nitime.utils as nt_utils

from numpy.lib.stride_tricks import as_strided
from sandbox.split_methods import multi_taper_psd
from sandbox.array_split import shared_ndarray, split_at
import ecoglib.filt.blocks as blocks

from .jackknife import Jackknife

__all__ = ['mtm_spectrogram_basic',
           'mtm_spectrogram',
           'mtm_complex_demodulate',
           'bispectrum',
           'normalize_spectrogram']

def _parse_mtm_args(kw_dict):
    NFFT = kw_dict.pop('NFFT', None)
    BW = kw_dict.pop('BW', None)
    Fs = kw_dict.pop('Fs', 1)
    NW = kw_dict.pop('NW', None)
    if BW is not None:
        # BW wins in a contest (since it was the original implementation)
        norm_BW = np.round(BW * N / Fs)
        NW = norm_BW / 2.0
    elif NW is None:
        # default NW
        NW = 4
    # (else BW is None and NW is not None) ... all set
    lb = kw_dict.pop('low_bias', True)
    return NW, NFFT, lb

def mtm_spectrogram_basic(x, n, pl=0.25, detrend='', **mtm_kwargs):
    """
    Make spectrogram using the multitaper spectral estimation method at
    each block.

    Parameters
    ----------
    n: int
        number of points per block
    pl: float
        percent overlap between blocks
    detrend: ''
        detrend each block as 'linear', 'constant',  (or not at all)
    mtm_args: dict
        keyword arguments for the multitaper family of methods

    Returns
    -------
    tx, fx, psd_matrix
    
    """

    mtm_kwargs.setdefault('adaptive', True)
    mtm_kwargs.setdefault('jackknife', False)
    xb = blocks.BlockedSignal(x, n, overlap=pl, partial_block=False)
    x_lapped = xb._x_blk.copy()
    if detrend:
        x_lapped = signal.detrend(x_lapped, type=detrend, axis=-1).copy()

    
    fx, psds, nu = multi_taper_psd(x_lapped, **mtm_kwargs)

    # infer time-resolution
    lag = round( (1-pl) * n )
    Fs = 2*fx[-1]
    spec_res = lag / Fs

    tx = np.arange(0, xb.nblock) * spec_res
    # align the bins at the middle of the strips
    tx += 0.5 * n / Fs

    return tx, fx, psds.transpose(0, 2, 1)

def mtm_spectrogram(
        x, n, pl=0.25, detrend='', Fs=1.0, adaptive=True, samp_factor=1,
        freqs=None, **mtm_kwargs
        ):
    """
    Make spectrogram from baseband complex-demodulates computed for
    each frequency over moving blocks.

    Parameters
    ----------
    n: int
        number of points per block
    pl: float
        percent overlap between blocks
    detrend: ''
        detrend each block as 'linear', 'constant',  (or not at all)
    adaptive: bool
        weight each spectral estimator adaptively
    samp_factor: int
        Each complex demodulate has a temporal resolution of 1/2W, and will
        be resampled by default. The time resolution will be calculated
        to ensure that shifting windows overlap correctly (i.e. the 
        window step time will be ensured to be a multiple of the time 
        resolution). Set samp_factor to compute complex demodulates within
        a window at a higher time resolution.
    freqs: list
        If given, only keep the power envelopes at these frequencies
        (instead of the full spectrogram)
    mtm_args: dict
        keyword arguments for the multitaper family of methods, viz:

        * NW
        * low_bias
        * NFFT

    Returns
    -------
    tx, fx, psd_matrix
    
    """

    NW, nfft, lb = _parse_mtm_args(mtm_kwargs)
    if pl > 1:
        # re-specify in terms of a fraction
        pl = float(pl) / n

    # set up time-resolution and adjust overlap to a more
    # convenient number if necessary
    K = 2*NW
    if samp_factor == 0:
        user_delta = 1.0
    else:
        user_delta = float(n) / K / samp_factor
    m = round( (1-pl) * n )
    if m < user_delta:
        delta = m
    else:
        p = np.ceil(m / user_delta)
        if (m//p) * p < m:
            m = p * (m//p)
            print 'resetting pl from %0.2f'%pl,
            pl = 1 - float(m)/n
            print ' to %0.2f'%pl
        delta = m//p
        #delta -= delta % 2

    print user_delta, delta, m

    # check contiguous    
    if not x.flags.c_contiguous:
        x = x.copy(order='C')
    blk_x = blocks.BlockedSignal(
        x, n, overlap=pl, partial_block = False
        )

    nblock = blk_x.nblock

    # going to compute the family of complex demodulates for each block
    if not nfft:
        nfft = 2**int(np.ceil(np.log2(n)))
    if freqs is None:
        nfreq = nfft//2 + 1
    else:
        nfreq = len(freqs)

    fx = np.linspace(0, Fs/2, nfft//2 + 1)
        
    # calculate the number of psd matrix time points        
    # total size of time-frequency array
    #pts_per_block = n // delta
    pts_per_block = int( np.ceil( (n - delta//2) / delta ) )
    overlap = pts_per_block - m // delta
    psd_len = int( nblock * pts_per_block - (nblock-1)*overlap )
    psd_pl = float(overlap) / pts_per_block
    psd_matrix = np.zeros( x.shape[:-1] + (nfreq, psd_len), 'd' )
    print pts_per_block, overlap, psd_len
    # need to make sure psd overlap is
    blk_psd = blocks.BlockedSignal(
        psd_matrix, pts_per_block, overlap=psd_pl, 
        partial_block = False, axis=-1
        )
    # this array mirrors the psd matrix blocks and counts accumulation
    n_avg = np.zeros(psd_len)
    blk_n = blocks.BlockedSignal(
        n_avg, pts_per_block, overlap=psd_pl, partial_block = False
        )

    print blk_n.nblock, blk_x.nblock

    dpss, eigs = alg.dpss_windows(n, NW, 2*NW)
    if lb:
        lb = 0.99 if float(lb)==1.0 else lb
        print 'low_bias:', lb
        keepers = eigs > lb
        dpss = dpss[keepers]
        eigs = eigs[keepers]

    print 'n_tapers:', len(dpss)
    dpss_sub = dpss[..., int(delta//2)::int(delta)]
    weight = delta * dpss_sub.T.dot( dpss_sub.dot(np.ones(pts_per_block)) )
    #weight **= 2
    ## window = np.power( 
    ##     np.cos(np.linspace(-np.pi/2, np.pi/2, pts_per_block)), 0.1
    ##     )
    window = np.hamming(pts_per_block)
    weight *= window
    weight = window
    print 'weight max:', weight.max()
    
    for b in xrange(blk_n.nblock):
        # it's possible to exceed the data blocks, since we're not
        # using fractional blocks in the signal (??)
        if b >= nblock:
            break
        nwin = blk_n.block(b)
        #nwin[:] += 1 # just keep count of how many times we hit these points
        # oddly sqrt *looks* more right
        #nwin[:] += weight**.5
        nwin[:] += weight
        #nwin[:] = weight
        dwin = blk_x.block(b)
        if detrend:
            dwin = signal.detrend(dwin, type=detrend)
        mtm_pwr, ix, weighting = mtm_complex_demodulate(
            dwin, NW, nfft=nfft, adaptive=adaptive, dpss=dpss, eigs=eigs,
            samp_factor=1.0/delta if delta > 1 else 0
            )
        if freqs is None:
            mtm_pwr = 2 * np.abs(mtm_pwr)**2
        else:
            f_idx = fx.searchsorted(freqs)
            mtm_pwr = 2 * np.abs(mtm_pwr[f_idx])**2

        ## mtm_pwr *= mtm_pwr
        if np.iterable(weighting):
            mtm_pwr /= weighting[...,None]
        else:
            mtm_pwr /= weighting
        psd_win = blk_psd.block(b)
        #psd_win[:] = psd_win + mtm_pwr
        psd_win[:] = psd_win + window * mtm_pwr

    n_avg[n_avg==0] = 1
    #n_avg = np.convolve(n_avg, np.ones(overlap)/overlap, mode='same')
    #print n_avg
    psd_matrix /= n_avg
    if samp_factor == 0:
        tx = np.arange(psd_matrix.shape[-1], dtype='d')
    else:
        tx = (np.arange(psd_matrix.shape[-1]) + 0.5) * delta
    tx /= Fs

    # scale by freq so that total power(t) = \int{ psd(t,f) * df }
    df = Fs / nfft 
    psd_matrix /= df
    if freqs is not None:
        fx = freqs
        
    return tx, fx, psd_matrix

def bw2nw(bw, n, fs):
    """Full BW to NW, given sequence length n"""
    # nw = tw = t(bw)/2 = (n/fs)(bw)/2
    bw, n, fs = map(float, (bw, n, fs))
    return (n/fs) * (bw/2)

def nw2bw(nw, n, fs):
    """NW to full BW, given sequence length n"""
    # bw = 2w = 2(tw)/t = 2(nw)/t = 2(nw) / (n/fs) = 2(nw)(fs/n)
    nw, n, fs = map(float, (nw, n, fs))
    return 2 * nw * fs / n

def mtm_complex_demodulate(
        x, NW, nfft=None, adaptive=True, low_bias=True,
        dpss=None, eigs=None, samp_factor = 1, fmax=0.5

        ):
    """
    Computes the complex demodulate of x. The complex demodulate is
    essentially a time-frequency matrix that holds the complex "baseband"
    of x, as if demodulated from each frequency band (f +/- W). The width
    of the carrier band is of course determined by the Slepian concentration
    band W.

    Parameters
    ----------
    x : array
        Array with signal sequence in the last dimension
    NW : float
        DPSS parameter
    samp_factor: int
        By default, the complex demodulate will be resampled to its temporal
        resolution of 1/2W. Setting samp_factor > 1 samples at that many times
        the temporal resolution. 
        If a specific (integer) sample-resolution is desired, then set
        samp_factor = 1.0 / sample_resolution.
        Setting samp_factor == 0 disables resampling.
    dpss: array
        pre-computed Slepian sequences
    eigs: array
        eigenvalues for pre-computed Slepians

    Returns
    -------
    x_tf : array
        complex demodulates at baseband for each frequency
    ix : array
        ix is an array of resampled points, relative to the full indexing of
        the input array.
    weight : array
        weight is the matrix of weights **(why??)**

    """

    N = x.shape[-1]
    
    if dpss is None:
        dpss, eigs = alg.dpss_windows(N, NW, 2*NW)
        if low_bias:
            low_bias = 0.99 if float(low_bias)==1.0 else low_bias
            keepers = eigs > low_bias
            dpss = dpss[keepers]
            eigs = eigs[keepers]

            
    K = len(eigs)
    if nfft is None:
        nfft = int(2**np.ceil(np.log2(N)))
    fmax = int(round(fmax * nfft)) + 1
    xk = alg.tapered_spectra(x, dpss, NFFT=nfft)
    if adaptive:
        if xk.ndim == 2:
            xk.shape = (1,) + xk.shape
        weight = np.empty( (xk.shape[0], nfft // 2+1), 'd' )
        for m in xrange(xk.shape[0]):
            w, _ = nt_utils.adaptive_weights(xk[m], eigs, sides='onesided')
            weight[m] = np.sum(w**2, axis=0)
            xk[m, :, :fmax] = xk[m, :, :fmax] * w[:, :fmax]
        xk = np.squeeze(xk)[..., :fmax]
        weight = np.squeeze(weight)
    else:
        xk = xk[..., :fmax]
        weight = float(K)

    xk *= np.sqrt(eigs[:,None])


    # XXX: the interpolated tensor product is the same as
    # the tensor product with an interpolated sequence?
    if samp_factor == 0:
        dpss_sub = dpss
        ix = np.arange(N)
    elif samp_factor < 1:
        t_res = max(2, int(1/samp_factor))
        # force t_res to be even??
        # t_res -= t_res%2
        ix = (np.arange( N//t_res ) + 0.5) * t_res
        #ix = np.arange(0, N, t_res) + t_res/2
        if ix[-1] + t_res < N:
            ix = np.r_[ix, ix[-1]+t_res]
        dpss_sub = np.take(dpss, ix.astype('i'), axis=1)
    else:
        samp_factor = min(samp_factor, float(N)/(4*NW))
        t_res = np.floor(float(N) / (2*NW) / samp_factor)
        t_res = max(2.0, t_res)
        ix = (np.arange(2*NW*samp_factor) + 0.5) * t_res
        #ix = np.arange(0, N, t_res) + t_res/2
        dpss_interp = interpolate.interp1d(np.arange(N), dpss, axis=-1)
        dpss_sub = dpss_interp(ix)

    xk_win_ax = 0 + x.ndim - 1
    x_tf = np.tensordot(xk, dpss_sub, axes=(xk_win_ax,0))

    return x_tf, ix, weight

def bispectrum(
        x, NW, low_bias=True, nfft=None, fmax=0.5,
        bic=True, se=True, jackknife=True, all_samps=False,
        return_symmetric=False
        ):
    """
    Estimate the bispectrum of x using complex demodulates.

    Parameters
    ----------
    x : ndarray (1D)
        Compute bispectrum (or bicoherence) on this signal. Due to the
	memory consumption of this method, only 1D is implemented.
    NW : float
        Time-bandwidth product (2NW should be an integer).
    low_bais : {True | False | p < 1.0}
        Only use eigenvalues above 0.99 (by default), or the value
	given.
    nfft : int
        Number of FFT grid points
    bic : bool {True | False}
        Normalize the bispectrum (to yield the bicoherence).
    se : bool {True | False}
        Return the jack-knife standard error (sets jackknife to True).
    jackknife : bool {True | False}
        Use jackknife resampling for estimating the bispectrum mean
        and standard error. The Jackknife is ALWAYS used as the
        bicoherence ratio estimator.
    all_samps : bool {True | False}
        Skip all estimates and return all samples.
    return_symmetric : bool {True | False}
        If True, return the full (f1 + f2 <= 0.5) area of the bispectrum.
        (This option is implied if the cython-ized bispectrum method
        cannot be loaded.)

    Returns
    ------
    B : ndarray (n_freq, n_freq)
        The bispectrum (or bicoherence).
    se : ndarray (n_freq, n_freq)
        The jackknife standard error (if se==True)

    """

    N = x.shape[-1]
    dpss, eigs = alg.dpss_windows(N, NW, 2*NW)
    if low_bias:
        low_bias = 0.99 if float(low_bias)==1.0 else low_bias
        keepers = eigs > low_bias
        dpss = dpss[keepers]
        eigs = eigs[keepers]
        
    x_tf, ix, w = mtm_complex_demodulate(
        x, NW, nfft=nfft, dpss=dpss, eigs=eigs, samp_factor=1, fmax=fmax
        )

    nf, K = x_tf.shape
    
    dpss = dpss * np.sqrt(eigs[:,None])
    P = np.einsum('im,jm,km->ijk', dpss, dpss, dpss)
    v = np.sum(P**2)

    x_tf = x_tf.T
    try:
        from ._bispectrum import calc_bispectrum
        from scipy.sparse import csr_matrix
        samps, row, col = calc_bispectrum(x_tf.real, x_tf.imag)
        samps = samps.view('D')
        samps *= (N / v)
        def r(X):
            if X.ndim > 1:
                dm = np.empty( (len(X), nf, nf), dtype=X.dtype )
                for k in xrange(len(X)):
                    sm = csr_matrix( (X[k], (row, col)), shape=(nf, nf) )
                    dm[k] = sm.todense()
                    if return_symmetric:
                        dm[k].flat[::nf+1] /= 2.0
                        dm[k] = dm[k] + dm[k].T
                return dm
            sm = csr_matrix( (X, (row, col)), shape=(nf, nf) )
            if return_symmetric:
                dm = sm.todense()
                dm.flat[::nf+1] /= 2.0
                return dm + dm.T
            return sm.todense()
    except ImportError:
        samps = shared_ndarray( (K, nf, nf), typecode='D' )
        np.einsum('...i,...j->...ij', x_tf, x_tf, out=samps)

        b_tr = np.zeros( (nf, nf), dtype=x_tf.dtype )
        tr_i, tr_j = np.tril_indices(nf)

        # reflect the row indices to give the upper-left triangle:
        # i + j < nf for all entries in this triangle
        tr_i = nf - 1 - tr_i

        #@split_at(split_arg=(0,1))
        def _apply_third_product(x, tf, nf, tr_i, tr_j):
            b_tr = np.zeros( (nf, nf), dtype=x.dtype )
            for k in xrange(x.shape[0]):
                b_tr[ (tr_i, tr_j) ] = np.take(tf[k], tr_i + tr_j)
                x[k] *= b_tr.conj()
            return x

        samps = _apply_third_product(samps, x_tf, nf, tr_i, tr_j)
        samps *= (N / v)
        def r(X):
            return X

    if all_samps:
        return r(samps)

    # bi-coherence estimator
    def _BIC_ratio(P, axis=0):
        D = np.mean(np.abs(P)**2, axis=axis) ** 0.5
        return np.abs(P.mean(axis=axis)) / D

    if se:
        jackknife = True
    
    if not jackknife:
        # not jackknife is synonymous with not se
        est = _BIC_ratio(samps) if bic else samps.mean(0)
        return r(est)

    # But jackknife == True does not imply se == True
    if bic:
        pv = Jackknife(samps, axis=0).pseudovals(_BIC_ratio)
    else:
        pv = Jackknife(samps, axis=0).pseudovals(np.mean)
    return (r(pv.mean(0)), r(pv.std(0) / K**0.5)) if se else r(pv.mean(0))

def normalize_spectrogram(x, baseline):
    # normalize based on the assumption of stationarity in baseline
    nf = x.shape[1]
    y = np.log(x[..., baseline]).transpose(1, 0, 2).reshape(nf, -1)
    b_spec = y.mean(-1)
    b_rms = y.std(-1)

    z = ( np.log(x).mean(0) - b_spec[:,None]) / b_rms[:,None]
    return z
