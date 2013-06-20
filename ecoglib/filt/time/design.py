"""
Simple filter design wrappings
"""
import numpy as np
import scipy.signal as sig

def butter_bp(lo, hi, Fs=1, ord=6):

    return sig.filter_design.butter(
        ord, np.array([lo, hi])/Fs, btype='bandpass'
        )
