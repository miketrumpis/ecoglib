from __future__ import division
import numpy as np
import scipy.signal as signal
import nitime.algorithms as alg

from numpy.lib.stride_tricks import as_strided

import ecoglib.filt.blocks as blocks

def mtm_specgram(x, n, pl=0.25, detrend='', **mtm_kwargs):

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



