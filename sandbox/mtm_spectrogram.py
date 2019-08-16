
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import nitime.algorithms as alg
import nitime.utils as nt_utils

from numpy.lib.stride_tricks import as_strided
from sandbox.split_methods import multi_taper_psd
import ecoglib.filt.blocks as blocks

def mtm_spectrogram_basic(
        x, n, pl=0.25, detrend='', **mtm_kwargs
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
        print('block %03d of %03d'%(b, blk_x.nblock))
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
    for i in range(M):
        for j in range(i):
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
        x, n, pl=0.25, detrend='', Fs=1.0, adaptive=True, samp_factor=1,
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

    NW, nfft, lb = parse_mtm_args(mtm_kwargs)
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
            print('resetting pl from %0.2f'%pl, end=' ')
            pl = 1 - float(m)/n
            print(' to %0.2f'%pl)
        delta = m//p
        #delta -= delta % 2

    print(user_delta, delta, m)

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
    print(pts_per_block, overlap, psd_len)
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

    print(blk_n.nblock, blk_x.nblock)

    dpss, eigs = alg.dpss_windows(n, NW, 2*NW)
    if lb:
        lb = 0.99 if float(lb)==1.0 else lb
        print('low_bias:', lb)
        keepers = eigs > lb
        dpss = dpss[keepers]
        eigs = eigs[keepers]

    print('n_tapers:', len(dpss))
    dpss_sub = dpss[..., int(delta//2)::int(delta)]
    weight = delta * dpss_sub.T.dot( dpss_sub.dot(np.ones(pts_per_block)) )
    #weight **= 2
    ## window = np.power( 
    ##     np.cos(np.linspace(-np.pi/2, np.pi/2, pts_per_block)), 0.1
    ##     )
    window = np.hamming(pts_per_block)
    weight *= window
    weight = window
    print('weight max:', weight.max())
    
    for b in range(blk_n.nblock):
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
    # nw = tw = t(bw)/2 = (n/fs)(bw)/2
    bw, n, fs = list(map(float, (bw, n, fs)))
    return (n/fs) * (bw/2)

def nw2bw(nw, n, fs):
    # bw = 2w = 2(tw)/t = 2(nw)/t = 2(nw) / (n/fs) = 2(nw)(fs/n)
    nw, n, fs = list(map(float, (nw, n, fs)))
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

    x_tf, ix, weight

    ix is an array of resampled points, relative to the full indexing of
    the input array.

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
        for m in range(xk.shape[0]):
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


def _jn(fn, sample, *fnarg, **fnkw):
    # quick jackknife among rows of sample
    M = sample.shape[0]
    xtest = fn(sample[1:], *fnarg, **fnkw)
    pseudovals = np.zeros( (M,) + xtest.shape, dtype=xtest.dtype )
    pseudovals[0] = xtest
    for m in range(1,M):
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


def normalize_spectrogram(x, baseline):
    # normalize based on the assumption of stationarity in baseline
    nf = x.shape[1]
    y = np.log(x[..., baseline]).transpose(1, 0, 2).reshape(nf, -1)
    b_spec = y.mean(-1)
    b_rms = y.std(-1)

    z = ( np.log(x).mean(0) - b_spec[:,None]) / b_rms[:,None]
    return z
