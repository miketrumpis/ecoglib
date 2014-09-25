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

    bp = kwargs.pop('bp', ())
    bp_table = dict()
    if len(bp):
        # find out which block each break-point falls into, and
        # then set up a break-point table for each block
        bp = np.asarray(bp)
        bp_blocks = (bp/bsize).astype('i')
        new_bp = bp - bsize*bp_blocks
        bp_table.update( zip(bp_blocks, new_bp) )

    for n, xc in enumerate(x_blk.fwd()):
        blk_bp = bp_table.get(n, 0)
        xc[:] = signal.detrend(xc, axis=axis, bp=blk_bp, **kwargs)
    del xc
    del x_blk

def remove_modes(x, bsize=0, axis=-1, modetype='dense', n=1):

    # remove chronos modes discovered by SVD
    # the mode types may be:
    # * most "dense" (scored by l1-norm of corresponding topos mode)
    # * most "sparse" (scored by l1-norm of corresponding topos mode)
    # * most "powerful" (scored by singular value)
    
    x_blk = BlockedSignal(x, bsize, axis=axis)

    def _get_mode(blk, n):
        u, s, vt = np.linalg.svd(blk, 0)
        if modetype in ('sparse', 'dense'):
            support = np.abs(u).sum(0)
            ix = np.argsort(support)
            if modetype == 'sparse':
                m_ix = ix[:n]
            else:
                m_ix = ix[-n:]
            return u[:, m_ix].dot( vt[m_ix] * s[m_ix][:, None] )
        # else return most powerful modes
        return u[:, :n].dot( vt[:n] * s[:n][:, None] )

    for blk in x_blk.fwd():
        blk -= _get_mode(blk, n)
    return x
