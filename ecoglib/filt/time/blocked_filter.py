import numpy as np
import scipy.signal as signal

from ..blocks import BlockedSignal

def bfilter(b, a, x, bsize=0, axis=-1, zi=None, filtfilt=False):
    """
    Apply linear filter inplace over the (possibly blocked) axis of x.
    If implementing a blockwise filtering for extra large runs, take
    advantage of initial and final conditions for continuity between
    blocks.
    """
    if not bsize:
        bsize = x.shape[axis]
    x_blk = BlockedSignal(x, bsize, axis=axis)

    zii = signal.lfilter_zi(b, a)
    zi_sl = [np.newaxis] * x.ndim
    zi_sl[axis] = slice(None)
    xc_sl = [slice(None)] * x.ndim
    xc_sl[axis] = slice(0,1)

    for n, xc in enumerate(x_blk.fwd()):
        if n == 0:
            zi = zii[ tuple(zi_sl) ] * xc[ tuple(xc_sl) ]
        xcf, zi = signal.lfilter(b, a, xc, axis=axis, zi=zi)
        xc[:] = xcf

    if not filtfilt:
        return

    # loop through in reverse order, slicing out reverse-time blocks
    for n, xc in enumerate(x_blk.bwd()):
        if n == 0:
            zi = zii[ tuple(zi_sl) ] * xc[ tuple(xc_sl) ]
        xcf, zi = signal.lfilter(b, a, xc, axis=axis, zi=zi)
        xc[:] = xcf
    del xc
    del x_blk

def bdetrend(x, bsize=0, **kwargs):
    "Apply detrending over the (possibly blocked) axis of x."
    axis = kwargs.pop('axis', -1)
    if not bsize:
        bsize = x.shape[axis]
    x_blk = BlockedSignal(x, bsize, axis=axis)

    for xc in x_blk.fwd():
        xc[:] = signal.detrend(xc, axis=axis, **kwargs)
    del xc
    del x_blk

