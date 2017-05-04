"""
Simple filter design wrappings
"""
import numpy as np
import scipy.signal.filter_design as fdesign
import scipy.signal as signal
from scipy import poly

__all__ = [ 'butter_bp', 
            'cheby1_bp', 
            'cheby2_bp', 
            'notch', 
            'plot_filt',
            'continuous_amplitude_linphase' ]

def _bandpass_params(lo, hi):
    (lo, hi) = map(float, (lo, hi))
    if not (lo > 0 or hi > 0):
        raise ValueError('no cutoff frequencies set')
    if lo and not hi > 0:
        return lo, 'highpass'
        ## return sig.filter_design.butter(
        ##     ord, 2 * lo / Fs, btype='highpass'
        ##     )
    if hi and not lo > 0:
        return hi, 'lowpass'
        ## return sig.filter_design.butter(
        ##     ord, 2 * hi / Fs, btype='lowpass'
        ##     )
    return np.array([lo, hi]), 'bandpass'

def butter_bp(lo=0, hi=0, Fs=2.0, ord=3):

    # note: "lo" corresponds to highpass cutoff
    #       "hi" corresponds to lowpass cutoff
    freqs, btype = _bandpass_params(lo, hi)
    return fdesign.butter(ord, 2*freqs/Fs, btype=btype)
    
def cheby1_bp(ripple, lo=0, hi=0, Fs=2.0, ord=3):
    freqs, btype = _bandpass_params(lo, hi)
    return fdesign.cheby1(ord, ripple, 2*freqs/Fs, btype=btype)
    
def cheby2_bp(rstop, lo=0, hi=0, Fs=2.0, ord=3):
    freqs, btype = _bandpass_params(lo, hi)
    return fdesign.cheby2(ord, rstop, 2*freqs/Fs, btype=btype)

def notch(fcut, Fs=2.0, nwid=3.0, npo=None, nzo=3):

    f0 = fcut * 2 / Fs
    fw = nwid * 2 / Fs

    z = [np.exp(1j*np.pi*f0), np.exp(-1j*np.pi*f0)]
    
    # find the polynomial with the specified (multiplicity of) zeros
    b = poly( np.array( z * int(nzo) ) )
    # the polynomial with the specified (multiplicity of) poles
    if npo is None:
        npo = nzo
    a = poly( (1-fw) * np.array( z * int(npo) ) )
    return (b, a)

def plot_filt(
        b, a, Fs=2.0, n=2048, log=True, logx=False, db=False, 
        filtfilt=False, phase=False, ax=None, **plot_kws
        ):
    import matplotlib.pyplot as pp

    if logx:
        hi = np.log10( Fs/2. )
        lo = hi - 4
        w = np.logspace(lo, hi, n)
    else:
        w = np.linspace(0, Fs/2.0, n) 
    _, f = signal.freqz(b, a, worN=w * (2*np.pi/Fs))
    if ax:
        pp.sca(ax)
    else:
        pp.figure()
    if filtfilt:
        m = np.abs(f)**2
    else:
        m = np.abs(f)

    if log and db:
        # assume dB actually preferred
        log = False

    if db:
        m = 20*np.log(m)
    if logx and log:
        pp.loglog( w, m, **plot_kws )
        pp.ylabel('Magnitude')
    elif log:
        pp.semilogy( w, m, **plot_kws )
        pp.ylabel('Magnitude')
    elif logx:
        pp.semilogx( w, m, **plot_kws )
        pp.ylabel('Magnitude (dB)' if db else 'Magnitude')
    else:
        pp.plot( w, m, **plot_kws )
        pp.ylabel('Magnitude (dB)' if db else 'Magnitude')
    pp.xlabel('Frequency (Hz)')
    pp.title('Frequency response' + (' (filtfilt)' if filtfilt else ''))
    if phase:
        plot_kws['ls'] = '--'
        ax2 = pp.gca().twinx()
        ax2.plot( w, np.angle(f), **plot_kws )
        ax2.set_ylabel('radians')

def continuous_amplitude_linphase(ft_samps):
    """Given Fourier Transform samples of a linear phase system,
    return functions of amplitude and phase such that the amplitude
    function is continuous (ie, not a magnitude function), and that
    f(e) = a(e)exp(j*p(e))

    Parameters
    ----------

    ft_samps: ndarray
      N complex samples of the fourier transform

    Returns
    -------

    (a, p): ndarray
      (continuous) amplitude and phase functions
    """
    npts = len(ft_samps)
    p_jumps = np.unwrap(np.angle(ft_samps))
    p_diff = np.diff(p_jumps)
    # assume there is not a filter zero at point 0 or 1
    p_slope = p_diff[0]
    zeros = np.where(np.pi - (p_diff-p_slope) <= (np.pi-1e-5))[0] + 1
    zeros = np.where(np.abs(p_diff-p_slope) >= 1)[0] + 1
                     
    zeros = np.r_[zeros, npts]

    # now get magnitude from ft_samps
    # assumption: amplitude from 0 to first filter zero is positive 
    a = np.abs(ft_samps)
    psign = np.sign(p_slope)
    k=1
    for lower, upper in zip(zeros[:-1], zeros[1:]):
        a[lower:upper] *= np.power(-1, k)
        p_jumps[lower:upper] += k*psign*np.pi
        k += 1

    return a, p_jumps
