"""Spectral estimation methods for nonstationary/nonlinear timeseries"""

import numpy as np
import scipy.signal as signal
import scipy.stats.distributions as dists
try:
    import scipy.fft as fft
    POCKET_FFT = True
except ImportError:
    import scipy.fftpack as fft
    POCKET_FFT = False
import nitime.utils as nt_utils

from ecogdata.parallel.sharedmem import shared_ndarray
import ecogdata.filt.blocks as blocks
from ecogdata.util import nextpow2, dpss_windows

from .resampling import Jackknife


__all__ = ['bw2nw',
           'nw2bw',
           'MultitaperEstimator',
           'mtm_spectrogram_basic',
           'mtm_spectrogram',
           'mtm_complex_demodulate',
           'bispectrum',
           'normalize_spectrogram']


def _parse_mtm_args(N, kw_dict):
    # parse args for some retro signatures
    if 'NFFT' in kw_dict:
        nfft = kw_dict.pop('NFFT', None)
    elif 'nfft' in kw_dict:
        nfft = kw_dict.pop('nfft', None)
    if nfft is None:
        nfft = N
    elif nfft == 'auto':
        nfft = nextpow2(N)
    BW = kw_dict.pop('BW', None)
    Fs = kw_dict.pop('Fs', 1)
    NW = kw_dict.pop('NW', None)
    if BW is not None:
        # BW wins in a contest (since it was the original implementation)
        NW = bw2nw(BW, N, Fs, halfint=True)
    elif NW is None:
        # default NW
        NW = 4
    # (else BW is None and NW is not None) ... all set
    lb = kw_dict.pop('low_bias', True)
    return NW, nfft, lb


def bw2nw(bw, n, fs, halfint=True):
    """Full BW to NW, given sequence length n"""
    # nw = tw = t(bw)/2 = (n/fs)(bw)/2
    bw, n, fs = list(map(float, (bw, n, fs)))
    nw = (n / fs) * (bw / 2)
    if halfint:
        # round 2NW to the closest integer and then halve again
        nw = round(2 * nw) / 2.0
    return nw


def nw2bw(nw, n, fs):
    """NW to full BW, given sequence length n"""
    # bw = 2w = 2(tw)/t = 2(nw)/t = 2(nw) / (n/fs) = 2(nw)(fs/n)
    nw, n, fs = list(map(float, (nw, n, fs)))
    return 2 * nw * fs / n


# These estimator functions used to be closures.
# They need to be importable for spawn-based multiprocessing.
def _psd_from_direct_spectra(Y, w, logout=False):
    sk = np.sum(w * Y, axis=1) / np.sum(w, axis=1)
    if logout:
        return np.log(sk)
    else:
        return sk


def _csd_from_direct_spectra(y, w):
    # weighted sdfs
    yw = y * w
    numer = np.einsum('mkl,nkl->mnl', yw, yw.conj())
    wsum = np.sum(w ** 2, axis=1) ** 0.5
    denom = wsum[:, None, :] * wsum[None, :, :]
    return numer / denom


class _DPSScache:

    cache = dict()

    @classmethod
    def prepare_dpss(cls, N, NW, low_bias=True, Kmax=None):
        if (N, NW) in cls.cache:
            dpss_c, eigs_c = cls.cache[(N, NW)]
            dpss = dpss_c.copy()
            eigs = eigs_c.copy()
        else:
            dpss, eigs = dpss_windows(N, NW, int(2 * NW))
            # always store 2NW eigenvectors/values
            cls.cache[(N, NW)] = (dpss.copy(), eigs.copy())
        if low_bias:
            low_bias = 0.99 if float(low_bias) == 1.0 else low_bias
            keepers = eigs > low_bias
            dpss = dpss[keepers]
            eigs = eigs[keepers]
        if Kmax is not None and len(eigs) > Kmax:
            dpss = dpss[:Kmax]
            eigs = eigs[:Kmax]
        return dpss, eigs


class MultitaperEstimator:
    """
    Class to handle multitaper method spectral estimation.
    """

    def __init__(self, N, NW, fs=1.0, nfft=None, low_bias=True, dpss=None):
        """
        Constructs a multitaper estimator based on length-bandwidth parameters. Constructs discrete prolate
        spheroidal sequences (DPSS) and the estimator's frequency grid.

        Parameters
        ----------
        N: int
            Length of sequences
        NW: float
            Time-bandwidth product to define DPSS. 2NW DPSS will be constructed and K <= 2NW will be used based on
            spectral concentration and the low_bias argument.
        fs: float
            Sampling frequency (or 1 for normalized digital frequencies).
        nfft: int
            Compute the FFT with this many points, rather than N. If the value is 'auto', use the next highest power
            of two.
        low_bias: bool, float
            Restrict DPSS tapers based on bandpass concentration ratios, given by eigenvalues. If True, then restrict
            to tapers with eigenvalues > 0.99. If a number is given, then use that threshold.
        dpss: 2-tuple
            If DPSS and eigenvalues were pre-created, use them instead. Specify as dpss=(dpss_vecs, eigs)

        """
        if dpss is not None:
            self.dpss, self.eigs = dpss
        else:
            self.dpss, self.eigs = _DPSScache.prepare_dpss(N, NW, low_bias=low_bias)
        if nfft is None:
            self.nfft = N
        elif nfft == 'auto':
            self.nfft = nextpow2(N)
        else:
            self.nfft = nfft
        self.NW = NW
        self.freq = np.linspace(0, fs / 2.0, self.nfft // 2 + 1)

    @property
    def BW(self):
        return nw2bw(self.NW, self.dpss.shape[-1], self.freq[-2] * 2)

    def direct_sdfs(self, x, adaptive_weights=False, dof=False):
        """
        Compute uncorrelated direct spectral density functions under each of K tapers for array(s) in x.

        Parameters
        ----------
        x: ndarray
            1- or 2-dimension timeseries array. SDFs are computed for series in the last dimension.
        adaptive_weights: bool
            Compute adaptive weightings per frequency. If False, then use standard weights based on eigenvalues.
        dof: bool
            Return "approximate" chi-square degrees of freedom based on adaptive weights. Otherwise nu=2K

        Returns
        -------
        yk: ndarray
            Complex array of SDFs shaped (K, nfreq) for 1D input or (M, K, nfreq) for 2D input.
        w: ndarray
            SDF weights. If adaptive weighting was used, the shape of w matches yk.shape. Otherwise,
            dummy-dimensions are used in place of M and/or nfreq.

        Notes
        -----
        Since DPSS tapers are orthonormal, these SDFs can also be used as linear coefficients of narrowband
        subspaces. The subspaces are defined by frequency-shifted DPSS tapers: v(k, n) * exp(-j * w[b] * n)
        where w[b] is the FFT bin at b. The bandwidth of each subspace centered at w[b] is determined by the
        NW taper parameter.

        """
        shp = x.shape
        # x must be 2D (even if shaped (1, T))
        x = np.atleast_2d(x)
        M = x.shape[0]
        K = len(self.dpss)
        tapered = x[..., np.newaxis, :] * self.dpss
        fft_args = dict(axis=-1, n=self.nfft, overwrite_x=True)
        if POCKET_FFT:
            fft_args['workers'] = -1
        yk = fft.fft(tapered, **fft_args)
        half_pts = self.nfft // 2 + 1
        nu = np.zeros((M, half_pts))
        if adaptive_weights:
            w = np.zeros((M, K, half_pts))
            for m in range(M):
                w[m], nu[m] = nt_utils.adaptive_weights(yk[m], self.eigs)
        else:
            w = np.sqrt(self.eigs[None, :, None])
            nu[:] = 2 * K
            # w = np.tile(w, (M, 1, half_pts))
        # TODO: tiling weights and cutting/copying yk are fairly expensive (e.g. 700 ms for (300, 2 ** 15) input)
        yk = yk[..., :half_pts].copy()
        # if input was 1D then only return SDFs for single timeseries (get rid of dummy dimension)
        if len(shp) == 1:
            yk = yk[0]
            w = w[0]
            nu = nu[0]
        if dof:
            return yk, w, nu
        else:
            return yk, w

    def compute_psd(self, x, adaptive_weights=False, jackknife=False, jn_jobs=1, detrend=None, ci=False):
        """
        Compute the multitaper psd estimate(s) of series in x. PSD normalization is defined such that the integral of
        spectral power per Hz equals to the total variance in x:

        E{(x - E{x}) ** 2} = integral P(f) * df (eval from f=0 to f=Fs / 2)

        Parameters
        ----------
        x: ndarray
            1- or 2-dim array with timeseries in last dimension.
        adaptive_weights: bool
            Use adaptive weights in combining multitaper estimates.
        jackknife: bool
            Use jackknife resampling in combining multitaper estimates. In this mode, standard error is calculated
            after log-transforming jackknifed estimates. The confidence interval is calculated for the log domain
            based on Student's t distribution for K - 1 degrees of freedom, but is then exponentiated before return.
        jn_jobs: int
            Perform Jackknife resampling on this many processes.
        detrend: str, bool
            Remove order-1 (detrend='linear') or order-0 (detrend='constant') trends. If detrend=True, remove constant.
        ci: bool or float
            Compute the confidence interval at (100 * ci) percent or 95% by default. If jackknife is not used,
            then the standard chi-squared assumption is used. If adaptive weights are used, the minimum chi-squared
            DOF will be clipped to 1.

        Returns
        -------
        freqs: ndarray
            Frequency grid
        psds: ndarray
            Power spectral density estimate(s) in power / Hz.

        """
        # TODO: return jackknife or chi2 confidence interval
        if detrend is not None:
            if isinstance(detrend, bool):
                detrend = 'constant'
            x = signal.detrend(x, axis=-1, type=detrend)
        shp = x.shape
        x = np.atleast_2d(x).reshape(-1, x.shape[-1])
        yk, w, nu = self.direct_sdfs(x, adaptive_weights=adaptive_weights, dof=True)

        # funny -- this is faster (in one step) than doing inplace squares
        yk = np.abs(yk) ** 2
        # np.power(yk, 2, yk)
        np.power(w, 2, w)
        if jackknife:
            pxx, se = Jackknife([yk, w], axis=1, n_jobs=jn_jobs).estimate(_psd_from_direct_spectra,
                                                                          correct_bias=True,
                                                                          se=True,
                                                                          logout=True)
        else:
            pxx = _psd_from_direct_spectra(yk, w)

        # PSD needs normalization:
        # P(f) <-- 2 * Pf / samp_rate (for 0 < f < f_nyq)
        # P(f) <-- Pf / samp_rate (for DC and nyquist)
        # Save normalization for different scenarios, but reshape to output here
        pxx = pxx.reshape(shp[:-1] + (pxx.shape[-1],))
        if ci:
            if isinstance(ci, float):
                p = 1 - ci
            else:
                p = 1 - 0.95
            if jackknife:
                # normalize pxx from log domain
                pxx[..., 1:-1] += np.log(2) - np.log(self.freq[-1] * 2)
                pxx[..., 0] -= np.log(self.freq[-1] * 2)
                pxx[..., -1] -= np.log(self.freq[-1] * 2)
                se.shape = pxx.shape
                # Since SE is computed under log transform, it doesn't need to be scaled. The PSD normalization here
                # is just a shift in the mean
                t_iv = dists.t.ppf([p / 2, 1 - p / 2], yk.shape[1] - 1)
                conf_iv = np.array([pxx + t_iv[0] * se, pxx + t_iv[1] * se])
                np.exp(pxx, out=pxx)
                np.exp(conf_iv, out=conf_iv)
            else:
                # normalize
                pxx[..., 1:-1] /= self.freq[-1]
                pxx[..., 0] /= (self.freq[-1] * 2)
                pxx[..., -1] /= (self.freq[-1] * 2)
                nu.shape = pxx.shape
                np.putmask(nu, nu < 1, 1)
                chi2_iv_lo = dists.chi2.ppf(p / 2, nu)
                chi2_iv_hi = dists.chi2.ppf(1 - p / 2, nu)
                conf_iv = np.array([nu * pxx / chi2_iv_hi, nu * pxx / chi2_iv_lo])
            return self.freq, pxx, conf_iv
        else:
            pxx[..., 1:-1] /= self.freq[-1]
            pxx[..., 0] /= (self.freq[-1] * 2)
            pxx[..., -1] /= (self.freq[-1] * 2)
            return self.freq, pxx

    @classmethod
    def psd(cls, x, NW=2.5, fs=1.0, nfft=None, low_bias=True, dpss=None,
            adaptive_weights=False, jackknife=False, jn_jobs=1, detrend=None, ci=False):
        """
        Shortcut to create a MultitaperEstimator and then compute the psd for x. See arguments for
        MultitaperEstimator construction and compute_psd

        Returns
        -------
        freqs: ndarray
            Frequency grid
        psds: ndarray
            Power spectral density estimate(s) in power / Hz.

        """
        N = x.shape[-1]
        mt_estimator = cls(N, NW, fs=fs, nfft=nfft, low_bias=low_bias, dpss=dpss)
        return mt_estimator.compute_psd(x, adaptive_weights=adaptive_weights, jackknife=jackknife,
                                        jn_jobs=jn_jobs, detrend=detrend, ci=ci)

    def compute_csd(self, x, y=None, adaptive_weights=False, jackknife=False, jn_jobs=1, detrend=None):
        """
        Compute cross-spectra either between timeseries x and y, or between
        all the combinations in the vector timeseries x. Other arguments follow compute_psd.

        Returns
        -------
        freqs: ndarray
            Frequency grid
        csds: ndarray
            (M, M, nfreq) matrix of cross spectral densities. M=2 if x and y are specified separarely, else M=len(x).

        """
        # TODO: maybe implement "seed" kwargs -- basically computes only 1 row of the matrix
        if x.ndim == 1 and y is None:
            raise ValueError('Need two or more vectors')
        if y is not None:
            x = np.vstack([x, y])
        if detrend is not None:
            if isinstance(detrend, bool):
                detrend = 'constant'
            x = signal.detrend(x, axis=-1, type=detrend)
        yk, w, nu = self.direct_sdfs(x, adaptive_weights=adaptive_weights, dof=True)

        if jackknife:
            cxy = Jackknife([yk, w], axis=1, n_jobs=jn_jobs).estimate(_csd_from_direct_spectra,
                                                                      correct_bias=True,
                                                                      se=False)
        else:
            cxy = _csd_from_direct_spectra(yk, w)
        cxy /= (self.freq[-1] * 2)
        cxy[..., 1:] *= 2
        return self.freq, cxy

    @classmethod
    def csd(cls, x, y=None, NW=2.5, fs=1.0, nfft=None, low_bias=True, dpss=None,
            adaptive_weights=False, jackknife=False, jn_jobs=1, detrend=None):
        """
        Shortcut to create a MultitaperEstimator and then compute the csds for x. See arguments for
        MultitaperEstimator construction and compute_csd

        Returns
        -------
        freqs: ndarray
            Frequency grid
        csds: ndarray
            (M, M, nfreq) matrix of cross spectral densities. M=2 if x and y are specified separarely, else M=len(x).


        """
        N = x.shape[-1]
        mt_estimator = cls(N, NW, fs=fs, nfft=nfft, low_bias=low_bias, dpss=dpss)
        return mt_estimator.compute_csd(x, y=y, adaptive_weights=adaptive_weights,
                                        jackknife=jackknife, jn_jobs=jn_jobs, detrend=detrend)


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

    if x.ndim < 2:
        x = x.reshape((1,) + x.shape)
    NW, nfft, low_bias = _parse_mtm_args(x.shape[-1], mtm_kwargs)
    mtm_kwargs.setdefault('adaptive_weights', True)
    mtm_kwargs.setdefault('jackknife', False)
    xb = blocks.BlockedSignal(x, n, overlap=pl, partial_block=False)
    x_lapped = xb._x_blk.copy()
    if detrend:
        x_lapped = signal.detrend(x_lapped, type=detrend, axis=-1).copy()

    # fx, psds, nu = multi_taper_psd(x_lapped, **mtm_kwargs)
    fx, psds = MultitaperEstimator.psd(x_lapped, NW=NW, nfft=nfft, low_bias=low_bias, **mtm_kwargs)

    # infer time-resolution
    lag = round((1 - pl) * n)
    Fs = 2 * fx[-1]
    spec_res = lag / Fs

    tx = np.arange(0, xb.nblock) * spec_res
    # align the bins at the middle of the strips
    tx += 0.5 * n / Fs

    return tx, fx, psds.transpose(0, 2, 1).squeeze()


def mtm_spectrogram(
        x, n, pl=0.25, detrend='', Fs=1.0, adaptive_weights=True, samp_factor=1,
        freqs=None, pad=False, **mtm_kwargs
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
    adaptive_weights: bool
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

    # clean up possible call signature difference
    if 'adaptive' in mtm_kwargs:
        adaptive_weights = mtm_kwargs.pop('adaptive')
    NW, nfft, lb = _parse_mtm_args(n, mtm_kwargs)
    if pl > 1:
        # re-specify in terms of a fraction
        pl = float(pl) / n

    # set up time-resolution and adjust overlap to a more
    # convenient number if necessary
    K = 2 * NW
    if samp_factor == 0:
        user_delta = 1.0
    else:
        user_delta = float(n) / K / samp_factor
    m = round((1 - pl) * n)
    if m < user_delta:
        delta = m
    else:
        p = np.ceil(m / user_delta)
        if (m // p) * p < m:
            m = p * (m // p)
            pl = 1 - float(m) / n
        delta = m // p

    # check contiguous
    if not x.flags.c_contiguous:
        x = x.copy(order='C')
    blk_x = blocks.BlockedSignal(x, n, overlap=pl, partial_block=False)

    nblock = len(blk_x)

    # calculate the number of psd matrix time points
    # total size of time-frequency array
    pts_per_block = int(n // delta)
    overlap = pts_per_block - m // delta
    psd_len = int(nblock * pts_per_block - (nblock - 1) * overlap)
    psd_pl = float(overlap) / pts_per_block

    pad_n = mtm_complex_demodulate(n, NW, nfft=None, pad=pad, return_pad_length=True,
                                   samp_factor=(1.0 / delta if delta > 1 else 0))
    nfft = nextpow2(pad_n)
    dpss, eigs = _DPSScache.prepare_dpss(pad_n, float(pad_n * NW) / n, low_bias=lb)
    fx = np.linspace(0, Fs / 2, nfft // 2 + 1)

    if freqs is None:
        nfreq = len(fx)
    else:
        nfreq = len(freqs)

    psd_matrix = np.zeros(x.shape[:-1] + (nfreq, psd_len), 'd')
    blk_psd = blocks.BlockedSignal(
        psd_matrix, pts_per_block, overlap=psd_pl,
        partial_block=False, axis=-1
    )
    # this array mirrors the psd matrix blocks and counts accumulation
    n_avg = np.zeros(psd_len)
    blk_n = blocks.BlockedSignal(
        n_avg, pts_per_block, overlap=psd_pl, partial_block=False
    )

    ind = (np.arange(pts_per_block) + 0.5) * delta
    dpss_sub = dpss[..., ind.astype('i')]
    weight = delta * dpss_sub.T.dot(dpss_sub.dot(np.ones(pts_per_block)))
    window = np.hamming(pts_per_block)
    weight *= window
    weight = window

    for b in range(len(blk_n)):
        # it's possible to exceed the data blocks, since we're not
        # using fractional blocks in the signal (??)
        if b >= nblock:
            break
        nwin = blk_n.block(b)
        # nwin[:] += 1 # just keep count of how many times we hit these points
        # oddly sqrt *looks* more right
        # nwin[:] += weight**.5
        nwin[:] += weight
        dwin = blk_x.block(b)
        if detrend:
            dwin = signal.detrend(dwin, type=detrend)
        mtm_pwr, ix, weighting = mtm_complex_demodulate(
            dwin, NW, nfft=nfft, adaptive_weights=adaptive_weights, dpss=dpss, eigs=eigs,
            samp_factor=1.0 / delta if delta > 1 else 0, pad=pad
        )
        if freqs is None:
            mtm_pwr = 2 * np.abs(mtm_pwr)**2
        else:
            f_idx = fx.searchsorted(freqs)
            mtm_pwr = 2 * np.abs(mtm_pwr[f_idx])**2

        if np.iterable(weighting):
            mtm_pwr /= weighting[..., None]
        else:
            mtm_pwr /= weighting
        psd_win = blk_psd.block(b)
        psd_win[:] = psd_win + window * mtm_pwr

    n_avg[n_avg == 0] = 1
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


def mtm_complex_demodulate(x, NW, nfft=None, adaptive_weights=True, low_bias=True,
                           dpss=None, eigs=None, samp_factor=1,
                           fmax=0.5, pad=False, return_pad_length=False):
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
    pad: bool
        pad the timeseries to reflect the edges
    return_pad_length: bool
        Only compute the (nontrivial) padded length and return.
        Helpful for pre-computing DPSS and freq axis.

    Returns
    -------
    x_tf: array
        complex demodulates at baseband for each frequency
    ix: array
        ix is an array of resampled points, relative to the full indexing of
        the input array.
    weight: array
        weight is the matrix of weights **(why??)**

    """

    if return_pad_length and isinstance(x, int):
        N = x
    else:
        N = x.shape[-1]
    # Find the time basis
    if samp_factor == 0:
        t_res = 1
        # resample on samples (i.e. don't resample)
        resample_point = 1.0
    elif samp_factor < 1:
        t_res = max(2, int(1 / samp_factor))
        # resample midway between samples separated by t_res
        resample_point = 0.5
    else:
        samp_factor = min(samp_factor, float(N) / (4 * NW))
        t_res = int(np.floor(float(N) / (2 * NW) / samp_factor))
        t_res = max(2, t_res)
        resample_point = 0.5
    if pad:
        # Append an exact multiple of t_res for the pre-pad record so that
        # the downsampled time base aligns correctly between [0, N-1]
        neg_segs = ((N - 1) // t_res)
        neg_length = neg_segs * t_res
        N_pad = neg_length + 2 * N - 1
        if return_pad_length:
            return N_pad
        x = np.hstack([x[..., 1:1 + neg_length][..., ::-1], x, x[..., :-1][..., ::-1]])
        ix = ((np.arange(N_pad // t_res) - neg_segs) + resample_point) * t_res
    else:
        if return_pad_length:
            return N
        N_pad = N
        neg_segs = 0
        ix = (np.arange(N // t_res) + resample_point) * t_res

    BW = float(NW) / N
    NW_pad = BW * N_pad
    if dpss is None:
        dpss, eigs = _DPSScache.prepare_dpss(N_pad, NW_pad, low_bias=low_bias)
    if nfft is None:
        nfft = nextpow2(N_pad)
    K = len(eigs)

    mtm = MultitaperEstimator(N_pad, NW_pad, nfft=nfft, dpss=(dpss, eigs))
    xk, weight = mtm.direct_sdfs(x, adaptive_weights=adaptive_weights)
    if adaptive_weights:
        xk *= weight
        # repurpose weight as the sum of squared weights across K direct SDFs
        weight = np.sum(weight ** 2, axis=-2)
    else:
        weight = float(K)

    xk *= np.sqrt(eigs[:, None])

    xk_win_ax = x.ndim - 1
    # expand the complex coefficients at each frequency with the baseband DPSS vectors
    x_tf = np.tensordot(xk, dpss, axes=(xk_win_ax, 0))
    if resample_point == 0.5:
        t1 = (ix + neg_segs * t_res).astype('i')
        x_tf1 = np.take(x_tf, t1, axis=-1)
        # print('taking time axis at', t1)
        if t_res % 2:
            x_tf1 += np.take(x_tf, t1 + 1, axis=-1)
            # print('averaging with times at', t1 + 1)
            x_tf1 /= 2
        x_tf = x_tf1
    if (ix < 0).any():
        if resample_point == 0.5:
            # limit last point to an interior point between samples
            ix_mask = (ix >= 0) & (ix < N - 0.5 * t_res)
        else:
            # limit last point to actual last point
            ix_mask = (ix >= 0) & (ix < N)
        x_tf = x_tf[..., ix_mask]
        ix = ix[ix_mask]

    return x_tf, ix, weight


# bi-coherence estimator -- needs to be importable for pickle & spawn
def _BIC_ratio(P):
    D = np.mean(np.abs(P) ** 2, axis=0) ** 0.5
    return np.abs(P.mean(axis=0)) / D


def bispectrum(
        x, NW, low_bias=True, nfft=None, fmax=0.5,
        bic=True, se=True, jackknife=True, jn_jobs=1, all_samps=False,
        return_symmetric=False, return_sparse=False
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
    low_bias : {True | False | 0 < p < 1}
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
    jn_jobs: int
            Perform Jackknife resampling on this many processes.
    all_samps : bool {True | False}
        Skip all estimates and return all samples.
    return_symmetric : bool {True | False}
        If True, return the full (f1 + f2 <= 0.5) area of the bispectrum.
        (This option is implied if the cython-ized bispectrum method
        cannot be loaded.)
    return_sparse : bool {True | False}
        If True, return the sparse matrix (matrices). (This option is
        only applies to the cython-ized bispectrum method.)

    Returns
    ------
    B : ndarray (n_freq, n_freq)
        The bispectrum (or bicoherence).
    se : ndarray (n_freq, n_freq)
        The jackknife standard error (if se==True)

    """

    N = x.shape[-1]
    dpss, eigs = _DPSScache.prepare_dpss(N, NW, low_bias=low_bias)

    x_tf, ix, w = mtm_complex_demodulate(
        x, NW, nfft=nfft, dpss=dpss, eigs=eigs, samp_factor=1, fmax=fmax, pad=False
    )

    nf, K = x_tf.shape

    dpss = dpss * np.sqrt(eigs[:, None])
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
                sm = [csr_matrix((X_, (row, col)), shape=(nf, nf))
                      for X_ in X]
                if return_sparse:
                    return sm
                dm = np.empty((len(X), nf, nf), dtype=X.dtype)
                for k in range(len(X)):
                    dm[k] = sm[k].todense()
                    if return_symmetric:
                        dm[k].flat[::nf + 1] /= 2.0
                        dm[k] = dm[k] + dm[k].T
                return dm
            sm = csr_matrix((X, (row, col)), shape=(nf, nf))
            if return_sparse:
                return sm
            if return_symmetric:
                dm = sm.todense()
                dm.flat[::nf + 1] /= 2.0
                return dm + dm.T
            return sm.todense()
    except ImportError:
        samps = shared_ndarray((K, nf, nf), typecode='D')
        np.einsum('...i,...j->...ij', x_tf, x_tf, out=samps)
        tr_i, tr_j = np.tril_indices(nf)

        # reflect the row indices to give the upper-left triangle:
        # i + j < nf for all entries in this triangle
        tr_i = nf - 1 - tr_i

        # @split_at(split_arg=(0,1))
        def _apply_third_product(x, tf, nf, tr_i, tr_j):
            b_tr = np.zeros((nf, nf), dtype=x.dtype)
            for k in range(x.shape[0]):
                b_tr[(tr_i, tr_j)] = np.take(tf[k], tr_i + tr_j)
                x[k] *= b_tr.conj()
            return x

        samps = _apply_third_product(samps, x_tf, nf, tr_i, tr_j)
        samps *= (N / v)

        def r(X):
            return X

    if all_samps:
        return r(samps)

    if se:
        jackknife = True

    if not jackknife:
        # not jackknife is synonymous with not se
        est = _BIC_ratio(samps) if bic else samps.mean(0)
        return r(est)

    # But jackknife == True does not imply se == True
    if bic:
        pv = Jackknife(samps, axis=0, n_jobs=jn_jobs).pseudovals(_BIC_ratio)
    else:
        pv = Jackknife(samps, axis=0, n_jobs=jn_jobs).pseudovals(np.mean)
    return (r(pv.mean(0)), r(pv.std(0) / K**0.5)) if se else r(pv.mean(0))


def _circular_clip(x, eps=0):
    if x.dtype not in np.sctypes['complex']:
        return np.clip(x, -1, 1)
    x_mag = np.abs(x)
    x_phs = np.angle(x)
    np.putmask(x, x_mag > 1 - eps, (1 - eps) * np.exp(1j * x_phs))
    return x

# coherence estimator: needs to be importable for pickle & spawn
def _coh_estimator(Y, seed, msc, fisher):
    # Y is (M, K, Nf)
    if seed is not None:
        coh = (Y * Y[seed].conj()).sum(-2)
    else:
        coh = np.einsum('mkn,pkn->mpn', Y, Y.conj())
    y_auto = np.sum(np.abs(Y) ** 2, axis=-2)
    np.sqrt(y_auto, y_auto)
    if seed is not None:
        y_auto *= y_auto[seed]
        coh /= y_auto
    else:
        coh = coh / y_auto[None, :, :]
        coh = coh / y_auto[:, None, :]
    # Need to "circularly clip" imaginary values by shrinking them to the unit circle
    coh = _circular_clip(coh, eps=1e-16)
    if msc:
        coh = np.abs(coh) ** 2
        if fisher:
            coh = np.arctanh(coh)
    elif fisher:
        # Fisher's transform, but complex
        # apply transformation separately to real and imag values?
        coh.real[:] = np.arctanh(coh.real)
        coh.imag[:] = np.arctanh(coh.imag)
    return coh


def coherence(
        x, NW, msc=True, dpss=None, eigs=None, ci=False, jn_jobs=1, fisher=True,
        low_bias=True, nfft='auto', seed=None
):
    """Estimate the coherence spectrum between different signals.

    Parameters
    ----------
    x : ndarray (M, N)
        Vector timeseries of sources
    NW : float
        Time-bandwidth product defining the Slepian tapers
    msc : {True | False}
        Compute the magnitude-square coherence (MSC) of the
        dual spectrum.
    dpss : None
        Precomputed Slepian tapers vk(N,W)
    eigs : None
        Eigenvalues of precomputed tapers
    ci : bool or float
        Return a confidence interval based on the jackknife variance of the estimator.
    jn_jobs: int
            Perform Jackknife resampling on this many processes.
    fisher: bool
        Use arc hyperbolic tangent (Fisher) normalization for jackknifing (does not seem to work well!!)
    low_bias : {True | False | 0 < p < 1}
        Only use eigenvalues above 0.99 (by default), or the value
    nfft : {None, int, 'auto'}
        Number of FFT points to use (default is next-power-of-2)

    Returns
    -------
    spec : ndarray (..., nfreq, nfreq)
        Dual spectrum (or MSC)
    se : ndarray (..., nfreq, nfreq)
        Standard error (if se==True)

    """

    N = x.shape[-1]
    if dpss is None:
        dpss, eigs = _DPSScache.prepare_dpss(N, NW, low_bias=low_bias)

    mtm = MultitaperEstimator(N, NW, nfft=nfft, dpss=(dpss, eigs))
    xk, weight = mtm.direct_sdfs(x, adaptive_weights=False)
    if not ci:
        fisher = False

    d_spec = _coh_estimator(xk, seed, msc, fisher)
    if ci:
        err = Jackknife(xk, axis=-2, n_jobs=jn_jobs).variance(_coh_estimator, e_args=(seed, msc, fisher))
        if isinstance(ci, float):
            p = 1 - ci
        else:
            p = 1 - 0.95
        t_iv = dists.t.ppf([p / 2, 1 - p / 2], len(xk) - 1)
        conf_iv = np.array([d_spec + t_iv[0] * err, d_spec + t_iv[1] * err])
        if fisher:
            if msc:
                conf_iv = np.tanh(conf_iv)
            else:
                # don't know??
                pass
    if fisher:
        if msc:
            d_spec = np.tanh(d_spec)
        else:
            d_spec.real[:] = np.tanh(d_spec.real)
            d_spec.imag[:] = np.tanh(d_spec.imag)

    if msc:
        d_spec = np.clip(d_spec, 0, 1)
    else:
        d_spec = _circular_clip(d_spec)
    return (d_spec, conf_iv) if ci else d_spec


def _semiherence_estimator(Y, seed, real):
    # Y is (M, K, Nf)
    if seed is not None:
        semih = (Y * Y[seed].conj()).mean(-2)
    else:
        semih = np.einsum('mkn,pkn->mpn', Y, Y.conj())
        K = Y.shape[1]
        semih /= K
    y_auto = np.mean(np.abs(Y)**2, axis=-2)
    if real:
        semih = np.abs(semih.real)
    else:
        semih = np.abs(semih)

    semih *= -1
    semih += 0.5 * y_auto[:, None, :]
    semih += 0.5 * y_auto[None, :, :]
    return semih


def semiherence(
        x, NW, real=True, dpss=None, eigs=None, jackknife=True, jn_jobs=1, se=False,
        low_bias=True, nfft='auto', fmax=0.5, seed=None
):
    """Estimate the coherence spectrum between different signals.

    Parameters
    ----------
    x : ndarray (M, N)
        Vector timeseries of sources
    NW : float
        Time-bandwidth product defining the Slepian tapers
    msc : {True | False}
        Compute the magnitude-square coherence (MSC) of the
        dual spectrum.
    dpss : None
        Precomputed Slepian tapers vk(N,W)
    eigs : None
        Eigenvalues of precomputed tapers
    jackknife : bool
        Use the jackknife to estimate the dual spectrum (or dual MSC)
    jn_jobs: int
            Perform Jackknife resampling on this many processes.
    se : bool
        Also return standard error of the estimator
        (sets jackknife to True)
    low_bias : {True | False | 0 < p < 1}
        Only use eigenvalues above 0.99 (by default), or the value
    nfft : {None, int, 'auto'}
        Number of FFT points to use (default is next-power-of-2)

    Returns
    -------
    spec : ndarray (..., nfreq, nfreq)
        Dual spectrum (or MSC)
    se : ndarray (..., nfreq, nfreq)
        Standard error (if se==True)

    """

    N = x.shape[-1]
    if dpss is None:
        dpss, eigs = _DPSScache.prepare_dpss(N, NW, low_bias=low_bias)

    mtm = MultitaperEstimator(N, NW, nfft=nfft, dpss=(dpss, eigs))
    xk, weight = mtm.direct_sdfs(x, adaptive_weights=False)

    if se:
        jackknife = True

    if jackknife:
        d_spec, err = Jackknife(xk, axis=-2, n_jobs=jn_jobs).estimate(_semiherence_estimator,
                                                                      se=True,
                                                                      e_args=(seed, real))
    else:
        d_spec = _semiherence_estimator(xk, seed, real)
    np.clip(d_spec, 0, d_spec.max(), out=d_spec)
    return (d_spec, err) if se else d_spec


# Freq x Freq mean-square coherence estimator: importable for pickle & spawn
def _dual_msc_estimator(Y):
    # Y is packed with y1, y2 = Y
    y1, y2 = Y
    #d_spec = y1[..., :, :, None] * y2[..., :, None, :].conj()
    d_spec = np.einsum('...ki,...kj->...kij', y1, y2.conj())
    d_spec = np.abs(d_spec.mean(axis=-3))**2
    # get marginal spectral densities
    S1f = np.mean(np.abs(y1)**2, axis=-2)
    S2f = np.mean(np.abs(y2)**2, axis=-2)
    denom = np.einsum('...i,...j->...ij', S1f, S2f)
    return d_spec / denom


def dual_spectrum(
        x1, x2, NW, msc=True, dpss=None, eigs=None, jackknife=True, jn_jobs=1, se=False,
        low_bias=True, nfft='auto', fmax=0.5
):
    """Estimate the dual frequency spectrum between different signals.

    Parameters
    ----------
    x1 : ndarray (..., N)
        (Vector) timeseries from source 1
    x2 : ndarray (..., N)
        (Vector) timeseries from source 2
    NW : float
        Time-bandwidth product defining the Slepian tapers
    msc : {True | False}
        Compute the magnitude-square coherence (MSC) of the
        dual spectrum.
    dpss : None
        Precomputed Slepian tapers vk(N,W)
    eigs : None
        Eigenvalues of precomputed tapers
    jackknife : bool
        Use the jackknife to estimate the dual spectrum (or dual MSC)
    jn_jobs: int
            Perform Jackknife resampling on this many processes.
    se : bool
        Also return standard error of the estimator
        (sets jackknife to True)
    low_bias : {True | False | 0 < p < 1}
        Only use eigenvalues above 0.99 (by default), or the value
    nfft : {None, int, 'auto'}
        Number of FFT points to use (default is next-power-of-2)

    Returns
    -------
    spec : ndarray (..., nfreq, nfreq)
        Dual spectrum (or MSC)
    se : ndarray (..., nfreq, nfreq)
        Standard error (if se==True)

    """

    N = x1.shape[-1]
    assert x2.shape[-1] == N, 'need same length sequences'
    if dpss is None:
        dpss, eigs = _DPSScache.prepare_dpss(N, NW, low_bias=low_bias)

    mtm = MultitaperEstimator(N, NW, nfft=nfft, dpss=(dpss, eigs))
    x1k, weight1 = mtm.direct_sdfs(x1, adaptive_weights=False)
    x2k, weight2 = mtm.direct_sdfs(x2, adaptive_weights=False)

    if se:
        jackknife = True

    if not msc:
        #       x_{1,k}(f1)            x_{2,k}(f2)
        #samps = x1k[..., :, :, None] * x2k[..., :, None, :].conj()
        samps = np.einsum('...ki,...kj->...kij', x1k, x2k.conj())
        if jackknife:
            d_spec, err = Jackknife(samps, axis=-3, n_jobs=jn_jobs).estimate(np.mean, se=True)
        else:
            d_spec = np.mean(samps, axis=-3)
        return (d_spec, err) if se else d_spec

    samps = np.array([x1k, x2k])

    # seems like a slight abuse of the jackknife machinery
    if jackknife:
        msc, err = Jackknife(samps, axis=-2, n_jobs=jn_jobs).estimate(_dual_msc_estimator, se=True)
        np.clip(msc, 0, 1, msc)
    else:
        msc = _dual_msc_estimator(samps)
    return (msc, err) if se else msc


def normalize_spectrogram(x, baseline):
    # normalize based on the assumption of stationarity in baseline
    nf = x.shape[1]
    y = np.log(x[..., baseline]).transpose(1, 0, 2).reshape(nf, -1)
    b_spec = y.mean(-1)
    b_rms = y.std(-1)

    z = (np.log(x).mean(0) - b_spec[:, None]) / b_rms[:, None]
    return z


if __name__ == '__main__':
    arr = np.random.randn(300, 2 ** 15)
    freqs, psds = MultitaperEstimator.psd(arr, NW=5, fs=1.0, adaptive_weights=False, jackknife=True, low_bias=0.9)
