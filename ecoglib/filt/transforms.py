from .time import *
from .blocks import *

from scipy.signal import hilbert
from ecoglib.numutil import nextpow2

def lowpass_envelope(x, n, lowpass, design_kws=dict(), filt_kws=dict()):
    """
    Compute the lowpass envelope of the timeseries rows in array x.

    x : ndarray, n_chans x n_points

    n : int
        block size over which to compute Hilbert transform

    lowpass : float, 0 < lowpass < 1
        the corner frequency of the lowpass filter in unit frequency

    design_kws : dict
        any extra filter design arguments (sampling rate will be forced to 1)

    filt_kws : dict
        any extra filter processing arguments (e.g. "filtfilt" : {T/F} )
    
    """
    n = nextpow2(n)
    block_sig = BlockedSignal(x, n)
    for block in block_sig.fwd():
        block_a = hilbert(block, N=n)
        block[:] = np.abs(block_a[..., :block.shape[-1]])

    if lowpass == 1:
        return x
    
    filt_kws.setdefault('filtfilt', True)

    # hope for the best
    design_kws['Fs'] = 1.0
    design_kws['hi'] = lowpass
    design_kws.setdefault('ord', 5)
    
    
    return filter_array(
        x, inplace=False, design_kwargs=design_kws, filt_kwargs=filt_kws
        )
    
