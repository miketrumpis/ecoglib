"""
Simple filter design wrappings
"""
import numpy as np
import scipy.signal.filter_design as fdesign
import scipy.signal as signal

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
    
def cheby2_bp(rs, lo=0, hi=0, Fs=2.0, ord=3):
    freqs, btype = _bandpass_params(lo, hi)
    return fdesign.cheby2(ord, rs, 2*freqs/Fs, btype=btype)
    
def notch(fcut, Fs=2.0, nwid=6.0, ftype='butter'):
    (bn, an) = fdesign.iirdesign(
        2*np.array([fcut-nwid/2, fcut+nwid/2], 'd')/Fs, 
        2*np.array([fcut-nwid/5, fcut+nwid/5], 'd')/Fs,
        0.1, 30, ftype=ftype
    )
    return (bn, an)

def plot_filt(
        b, a, Fs=2.0, n=2048, log=True, db=False, filtfilt=False, ax=None
        ):
    import matplotlib.pyplot as pp
    w, f = signal.freqz(b, a, worN=n)
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
    
    if log:
        pp.semilogy( w*Fs/2/np.pi, m )
    else:
        if db:
            m = 10*np.log(m)
        pp.plot( w*Fs/2/np.pi, m )
    pp.title('freq response')
