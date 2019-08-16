import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, ifft
from scipy.linalg import LinAlgError

from tqdm import trange

from ecoglib.numutil import nextpow2
from ecoglib.util import input_as_2d

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

    if zi is not None:
        zii = zi.copy()
    else:
        try:
            zii = signal.lfilter_zi(b, a)
        except LinAlgError:
            # the integrating filter doesn't have valid zi
            zii = np.array([0.0])
        
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
        bp_table.update( list(zip(bp_blocks, new_bp)) )

    for n, xc in enumerate(x_blk.fwd()):
        blk_bp = bp_table.get(n, 0)
        xc[:] = signal.detrend(xc, axis=axis, bp=blk_bp, **kwargs)
    del xc
    del x_blk


@input_as_2d()
def overlap_add(x, w, progress=False):

    M = len(w)
    N = 0
    nfft = nextpow2(M) / 2
    while N < M:
        nfft *= 2
        # segment size > M and long enough not to cause circular convolution
        N = nfft - M + 1


    blocks = BlockedSignal(x, N, axis=-1)
    xf = np.zeros_like(x)
    #blocks_f = BlockedSignal(xf, nfft, overlap=1.0 - float(N)/nfft, axis=-1)
    blocks_f = BlockedSignal(xf, nfft, overlap=M-1, axis=-1)

    nb1 = blocks.nblock
    nb2 = blocks_f.nblock

    #print M, N, nfft, nb1, nb2

    # centered kernel (does not work!)
    ## pre_z = (nfft-M) // 2
    ## post_z = nfft - M - pre_z
    ## kern = fft( np.r_[ np.zeros(pre_z), w, np.zeros(post_z) ] )

    # not-centered kernel
    kern = fft(np.r_[w, np.zeros(nfft-M)])

    # centered padding (does not work!)
    ## pre_z = (nfft-N) // 2
    ## post_z = nfft - N - pre_z
    ## z1 = np.zeros( x.shape[:-1] + (pre_z,) )
    ## z2 = np.zeros( x.shape[:-1] + (post_z,) )

    # not-centered padding
    z = np.zeros( x.shape[:-1] + (nfft-N,) )

    if progress:
        itr = trange(nb1)
    else:
        itr = range(nb1)
    for n in itr:
        
        b = blocks.block(n)
        if b.shape[-1] == N:
            b = fft(np.concatenate( (b, z), axis=-1 ), axis=-1)
            #b = fft(np.concatenate( (z1, b, z2), axis=-1 ), axis=-1)
        else:
            # zero pad final segment
            z_ = np.zeros( b.shape[:-1] + (nfft-b.shape[-1],) )
            b = fft(np.concatenate( (b, z_), axis=-1 ), axis=-1)
        b = ifft(b * kern).real
        if n < nb2:
            bf = blocks_f.block(n)
            b_sl = [slice(None)] * bf.ndim
            b_sl[-1] = slice(0, bf.shape[-1])
        else:
            # keep the final filtered block and advance the overlap-index
            # within it
            b_sl = [slice(None)] * bf.ndim
            b_sl[-1] = slice(0, bf.shape[-1])
        bf[:] += b[b_sl]    
        #bf[:] += b[..., :bf.shape[-1]]
    #print b.shape, bf.shape, n

    if nb2 > nb1:
        print('more output blocks than input blocks??')
    
    return xf
        
        
        
    
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
