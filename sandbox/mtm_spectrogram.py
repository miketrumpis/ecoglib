from __future__ import division
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import nitime.algorithms as alg
import nitime.utils as nt_utils

from numpy.lib.stride_tricks import as_strided

import ecoglib.filt.blocks as blocks

def mtm_specgram(x, n, pl=0.25, detrend='', **mtm_kwargs):
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
    assert x.ndim == 1, 'Only taking spectrograms of single timeseries'

    lx = len(x)
    # so x need not be contiguous in memory, find the elementwise stride
    x_stride = x.strides[0]

    # reshape x into overlapping sections -- omit last block if necessary
    blk_stride = int(np.round((1-pl) * n))
    nblocks = lx // blk_stride

    x_lapped = as_strided(
        x, shape=(nblocks, n), strides=(x_stride*blk_stride, x_stride)
        )
    if detrend:
        x_lapped = signal.detrend(x_lapped, type=detrend, axis=1)

    fx, psds, nu = alg.multi_taper_psd(x_lapped, **mtm_kwargs)

    # infer freq sampling from f
    Fs = 2*fx[-1] / blk_stride

    tx = np.arange(0, nblocks) / Fs

    return tx, fx, psds

def mtm_coherogram(x, n, pl=0.25, axis=0, **mtm_kwargs):
    blk_x = blocks.BlockedSignal(
        x, n, overlap=pl, axis=axis, partial_block = False
        )
    nblock = blk_x.nblock
    nfreq = mtm_kwargs.get('NFFT', n)//2 + 1
    avg_coh = np.zeros( (nblock, nfreq) )

    NW, NFFT, lb = parse_mtm_args(mtm_kwargs)
    if not NFFT:
        NFFT = n
    # get the tapers up front..
    dpss, eigvals = alg.dpss_windows(n, NW, 2*NW)
    if lb:
        keepers = eigvals > 0.9
        dpss = dpss[keepers]
        eigvals = eigvals[keepers]
    lb = False
    K = len(eigvals)
    M = x.shape[1]
    weights = np.tile(np.sqrt(eigvals), M).reshape(M, K, 1)

    for b, win in enumerate(blk_x.fwd()):
        # win is (n, n_chan)
        print 'block %03d of %03d'%(b, blk_x.nblock)
        avg_coh[b] = avg_coherence(
            win.T, dpss, eigvals, NFFT=NFFT, low_bias=False
            )
    avg_coh /= M
    return avg_coh

def avg_coherence(x, dpss, eigs, NFFT=None, low_bias=True):
    # dpss is (dpss, eigs) or (NW, Kmax)
    M = x.shape[0]
    if not NFFT:
        NFFT = x.shape[-1]
    spectra = alg.tapered_spectra(x, dpss, NFFT=NFFT, low_bias=low_bias)
    K = len(eigs)
    weights = np.tile(np.sqrt(eigs), M).reshape(M, K, 1)

    spec2 = np.rollaxis(spectra, 1, start=0)
    w2 = np.rollaxis(weights, 1, start=0)
    sdf_est = alg.mtm_cross_spectrum(spec2, spec2, w2, sides='onesided')
    nf = sdf_est.shape[-1]
    coh = np.zeros(nf)
    for i in xrange(M):
        for j in xrange(i):
            csd = alg.mtm_cross_spectrum(
                spectra[i], spectra[j], (weights[i], weights[j]),
                sides='twosided'
                )
            coh += np.abs(csd[:nf])**2 / sdf_est[i] / sdf_est[j]
    coh /= (M*(M-1)/2.)
    return coh

def mtm_psd_brief(x, dpss, eigs, NFFT):
    spectra = alg.tapered_spectra(
        x, dpss, NFFT=NFFT, low_bias=False
        )
    M = x.shape[0]
    K = len(eigs)
    weights = np.tile(np.sqrt(eigs), M).reshape(M, K, 1)
    spectra = np.rollaxis(spectra, 1, start=0)
    weights = np.rollaxis(weights, 1, start=0)
    sdf_est = alg.mtm_cross_spectrum(
        spectra, spectra, weights, sides='onesided'
        )
    return sdf_est

def parse_mtm_args(kw_dict):
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



def mtm_spectrogram(
        x, n, pl=0.25, detrend='', Fs=1.0, adaptive=True, samp_factor=None,
        freqs=None, **mtm_kwargs
        ):
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

    adaptive: bool
      weight each spectral estimator adaptively

    samp_factor: int
      Each complex demodulate has a temporal resolution of 1/2W, and will
      be resampled by default.
      Setting samp_factor >= 1 samples at that many times the temporal
      resolution.
      Setting samp_factor < 1 disables resampling.
      Leaving samp_factor == None will allow the algorithm to set the
      samp_factor based on the amount of overlap between windows. I.e.
      samp_factor <-- ceil(1/(1 - pl))

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

    if pl > 1:
        # re-specify in terms of a fraction
        pl = float(pl) / n
    if not x.flags.c_contiguous:
        x = x.copy(order='C')
    blk_x = blocks.BlockedSignal(
        x, n, overlap=pl, partial_block = False
        )

    nblock = blk_x.nblock

    # going to compute the family of complex demodulates for each block
    NW, nfft, lb = parse_mtm_args(mtm_kwargs)
    if not nfft:
        nfft = 2**int(np.ceil(np.log2(n)))
    if freqs is None:
        nfreq = nfft//2 + 1
    else:
        nfreq = len(freqs)

    fx = np.linspace(0, Fs/2, nfft//2 + 1)
        
    # calculate the number of psd matrix time points
    if samp_factor is None:
        samp_factor = np.ceil( 1/(1-pl) )
    samp_factor = int(samp_factor)
    if samp_factor < 1:
        n_psd_times = n
    else:
        n_psd_times = 2*NW*samp_factor

    #psd_len = int( np.ceil( len(x) * float(n_psd_times) / n ) )
    psd_len = int( np.ceil( x.shape[-1] * float(n_psd_times) / n ) )
    #psd_matrix = np.zeros( x.shape[:-1] + (nfreq, psd_len), 'D' )
    psd_matrix = np.zeros( x.shape[:-1] + (nfreq, psd_len), 'd' )
    blk_psd = blocks.BlockedSignal(
        psd_matrix, n_psd_times, overlap=pl, partial_block = False, axis=-1
        )
    # this array mirrors the psd matrix blocks and counts accumulation
    n_avg = np.zeros(psd_len)
    blk_n = blocks.BlockedSignal(
        n_avg, n_psd_times, overlap=pl, partial_block = False
        )

    dpss, eigs = alg.dpss_windows(n, NW, 2*NW)
    if lb:
        keepers = eigs > 0.9
        dpss = dpss[keepers]
        eigs = eigs[keepers]

    for b in xrange(blk_n.nblock):
        # it's possible to exceed the data blocks, since we're not
        # using fractional blocks in the signal (??)
        if b >= blk_x.nblock:
            break
        nwin = blk_n.block(b)
        nwin[:] += 1 # just keep count of how many times we hit these points

        dwin = blk_x.block(b)
        if detrend:
            dwin = signal.detrend(dwin, type=detrend)
        mtm_pwr, _, weighting = mtm_complex_demodulate(
            dwin, NW, nfft=nfft, adaptive=adaptive, dpss=dpss, eigs=eigs,
            samp_factor=samp_factor
            )
        if freqs is None:
            mtm_pwr = np.sqrt(2) * np.abs(mtm_pwr)
        else:
            f_idx = fx.searchsorted(freqs)
            mtm_pwr = np.sqrt(2) * np.abs(mtm_pwr[f_idx])

        ## mtm_pwr *= mtm_pwr
        if np.iterable(weighting):
            mtm_pwr /= weighting[:,None]
        else:
            mtm_pwr /= weighting
        psd_win = blk_psd.block(b)
        psd_win[:] = psd_win + mtm_pwr

    n_avg[n_avg==0] = 1
    psd_matrix /= n_avg
    if samp_factor < 1:
        tx = np.arange(psd_matrix.shape[-1])
    else:
        t_res = float(n) / (2*NW) / samp_factor
        tx = (np.arange(psd_matrix.shape[-1]) + 0.5) * t_res
    tx /= Fs

    # scale by freq so that total power(t) = \int{ psd(t,f) * df }
    df = 2 * Fs / nfft 
    psd_matrix /= df
    if freqs is not None:
        fx = freqs
        
    return tx, fx, psd_matrix


def mtm_complex_demodulate(
        x, NW, nfft=None, adaptive=True, low_bias=True,
        dpss=None, eigs=None, samp_factor = 1

        ):
    """
    Computes the complex demodulate of x. The complex demodulate is
    essentially a time-frequency matrix that holds the complex "baseband"
    of x, as if demodulated from each frequency band (f +/- W). The width
    of the carrier band is of course determined by the Slepian concentration
    band W.

    samp_factor: int
      By default, the complex demodulate will be resampled to its temporal
      resolution of 1/2W. Setting samp_factor > 1 samples at that many times
      the temporal resolution. Setting samp_factor < 1 disables resampling.

    dpss: array
      pre-computed Slepian sequences

    eigs: array
      eigenvalues for pre-computed Slepians

    Returns
    -------

    x_tf, ix, weight

    ix is an array of resampled points, relative to the full indexing of
    the input array.

    weight is the matrix of weights **(why??)**

    """

    N = x.shape[-1]

    if dpss is None:
        dpss, eigs = alg.dpss_windows(N, NW, 2*NW)
        if low_bias:
            keepers = eigs > 0.9
            dpss = dpss[keepers]
            eigs = eigs[keepers]

    K = len(eigs)
    if nfft is None:
        nfft = int(2**np.ceil(np.log2(N)))

    xk = alg.tapered_spectra(x, dpss, NFFT=nfft)
    if adaptive:
        if xk.ndim == 2:
            xk.shape = (1,) + xk.shape
        weight = np.empty( (xk.shape[0], nfft/2+1), 'd' )
        for m in xrange(xk.shape[0]):
            w, _ = nt_utils.adaptive_weights(xk[m], eigs, sides='onesided')
            weight[m] = np.sum(w**2, axis=0)
        xk = xk[..., :nfft/2+1] * w
        xk = np.squeeze(xk)
        weight = np.squeeze(weight)
    else:
        xk = xk[..., :nfft/2+1]
        weight = float(K)

    xk *= np.sqrt(eigs[:,None])


    # XXX: the interpolated tensor product is the same as
    # the tensor product with an interpolated sequence?
    if samp_factor < 1:
        dpss_sub = dpss
        ix = np.arange(N)
    else:
        t_res = float(N) / (2*NW) / samp_factor
        ix = (np.arange(2*NW*samp_factor) + 0.5) * t_res
        dpss_interp = interpolate.interp1d(np.arange(N), dpss, axis=-1)
        dpss_sub = dpss_interp(ix)

    xk_win_ax = 0 + x.ndim - 1
    x_tf = np.tensordot(xk, dpss_sub, axes=(xk_win_ax,0))

    return x_tf, ix, weight


def _jn(fn, sample, *fnarg, **fnkw):
    # quick jackknife among rows of sample
    M = sample.shape[0]
    xtest = fn(sample[1:], *fnarg, **fnkw)
    pseudovals = np.zeros( (M,) + xtest.shape, dtype=xtest.dtype )
    pseudovals[0] = xtest
    for m in xrange(1,M):
        pseudovals[m] = fn(
            np.row_stack( (sample[:m], sample[m+1:]) ), *fnarg, **fnkw
            )
    return pseudovals

def jackknife_avg(samples, axis=0):

    jnr = _jn(np.mean, samples, axis=axis)

    M = jnr.shape[0]
    r1 = np.mean(samples, axis=axis)
    r2 = np.mean(jnr, axis=0)

    r = M*r1 - (M-1)*r2
    bias = (M-1) * (r2 - r1)

    err = np.sqrt( (float(M-1)/M) * np.sum( (jnr - r2)**2, axis=0 ) )

    return r, bias, err
