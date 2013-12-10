"""
Simple filter design wrappings
"""
import numpy as np
import scipy.signal as sig

def butter_bp(lo=0, hi=0, Fs=1, ord=6):

    # note: "lo" corresponds to highpass cutoff
    #       "hi" corresponds to lowpass cutoff
    
    if not (lo or hi):
        raise ValueError('no cutoff frequencies set')
    if lo and not hi:
        return sig.filter_design.butter(
            ord, 2 * lo / Fs, btype='highpass'
            )
    if hi and not lo:
        return sig.filter_design.butter(
            ord, 2 * hi / Fs, btype='lowpass'
            )
    
    return sig.filter_design.butter(
        ord, 2*np.array([lo, hi])/Fs, btype='bandpass'
        )
