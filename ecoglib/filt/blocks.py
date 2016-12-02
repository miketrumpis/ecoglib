from __future__ import division
import numpy as np

from numpy.lib.stride_tricks import as_strided

__all__ = ['BlockedSignal']

class BlockedSignal(object):
    """A class that transforms an N-dimension signal into multiple
    blocks along a given axis. The resulting object can yield blocks
    in forward or reverse sequence.
    """

    def __init__(self, x, bsize, overlap=0, axis=-1, partial_block=True):
        """
        Split a (possibly quite large) array into blocks along one axis.

        Parameters
        ----------

        x : ndarray
          The signal to blockify.
        bsize : int
          The maximum blocksize for the given axis.
        overlap : float 0 <= overlap <= 1 or int 0 < bsize
          The proportion of overlap between adjacent blocks. The (k+1)th
          block will begin at an offset of (1-overlap)*bsize points into
          the kth block. If overlap is an integer, it will be used literally.
        axis : int (optional)
          The axis to split into blocks
        partial_block : bool
          If blocks don't divide the axis length exactly, allow a partial
          block at the end (default True).

        """
        # if x is not contiguous then I think we're out of luck
        if not x.flags.c_contiguous:
            raise RuntimeError('The data to be blocked must be C-contiguous')
        # first reshape x to have shape (..., nblock, bsize, ...),
        # where the (nblock, bsize) pair replaces the axis in kwargs
        shape = x.shape
        strides = x.strides
        bitdepth = x.dtype.itemsize
        while axis < 0:
            axis += len(shape)
        bsize = int(bsize)
        if isinstance(overlap, int) and overlap > 1:
            L = bsize - overlap
        else:
            L = int( round( (1-overlap) * bsize ) )
        nblock = (shape[axis] - bsize) // L
        if partial_block and (shape[axis] > L*nblock + bsize):
            nblock += 1
            self._last_block_sz = shape[axis] - L*nblock
        else:
            self._last_block_sz = bsize
        nblock += 1
        nshape = shape[:axis] + (nblock, bsize) + shape[axis+1:]
        # Assuming C-contiguous, strides were previously
        # (..., nx*ny, nx, 1) * bitdepth
        # Change the strides at axis to reflect new shape
        b_offset = int( np.prod(shape[axis+1:]) * bitdepth )
        nstrides = strides[:axis] + \
          (L*b_offset, b_offset) + \
          strides[axis+1:]
        self.nblock = nblock
        self._axis = axis
        self._x_blk = as_strided(x, shape=nshape, strides=nstrides)

    def fwd(self):
        "Yield the blocks one at a time in forward sequence"
        # this object will be repeatedly modified in the following loop(s)
        blk_slice = [slice(None)] * self._x_blk.ndim
        axis = self._axis
        for blk in xrange(self.nblock):
            blk_slice[self._axis] = blk
            if blk == self.nblock-1:
                # VERY important! don't go out of bounds in memory!
                blk_slice[self._axis+1] = slice(0, self._last_block_sz)
            else:
                blk_slice[self._axis+1] = slice(None)
            xc = self._x_blk[ tuple(blk_slice) ]
            yield xc

    def bwd(self):
        "Yield the blocks one at a time in reverse sequence"
        # loop through in reverse order, slicing out reverse-time blocks
        bsize = self._x_blk.shape[self._axis+1]
        # this object will be repeatedly modified in the following loop(s)
        blk_slice = [slice(None)] * self._x_blk.ndim
        for blk in xrange(self.nblock-1, -1, -1):
            blk_slice[self._axis] = blk
            if blk == self.nblock-1:
                # VERY important! don't go out of bounds in memory!
                # (XXX: since when does this not work??)
                # blk_slice[axis+1] = slice(last_block_sz-1, -1, -1)
                # confusing.. but want to count down from the *negative*
                # index of the last good point: -(bsize+1-last_block_sz)
                # down to the *negative* index of the
                # beginning of the block: -(bsize+1)
                blk_slice[self._axis+1] = slice(
                    -(bsize+1) + self._last_block_sz, -(bsize+1), -1
                    )
            else:
                blk_slice[self._axis+1] = slice(None, None, -1)
            xc = self._x_blk[ tuple(blk_slice) ]
            yield xc

    def block(self, b):
        "Yield the index b block"
        blk_slice = [slice(None)] * self._x_blk.ndim
        while b < 0:
            b += self.nblock
        if b >= self.nblock:
            raise IndexError
        blk_slice[self._axis] = b
        if b == self.nblock-1:
            blk_slice[self._axis+1] = slice(0, self._last_block_sz)
        else:
            blk_slice[self._axis+1] = slice(None)
        return self._x_blk[ tuple(blk_slice) ]

def block_reduce(rfn, array, bsize, f_axis=1, **kwargs):
    bsig = BlockedSignal(array, bsize, **kwargs)
    reduced = list()
    for blk in bsig.fwd():
        reduced.append( rfn(blk, axis=f_axis) )
    return np.array( reduced )

def block_apply(fn, bsize, args, block_arg=0, b_axis=-1, **kwargs):
    """
    Performs blockwise computation of an array operator 'fn'.

    Parameters
    ----------

    fn : operator method with one-to-one array input-output
    bsize : block size to operate over
    args : method arguments sequence
    block_arg : index of the operand in 'args' sequence
    b_axis : axis of array to block (currently must be last)
    kwargs : method keyword arguments

    Returns
    -------

    arr : output array of same shape and dtype as input array

    """
    array = args[block_arg]
    if not (b_axis == -1 or b_axis == array.ndim-1):
        raise ValueError('Currently only blocking on last axis')

    def _hotswap_block(blk):
        n_arg = len(args)
        a = [args[n] for n in xrange(n_arg) if n != block_arg]
        a.insert(block_arg, blk)
        return a
    
    b_sig = BlockedSignal(array, bsize, partial_block=True, axis=b_axis)
    a_proc = np.empty_like(array)
    b_proc = BlockedSignal(a_proc, bsize, partial_block=True, axis=b_axis)

    for b_in, b_out in zip( b_sig.fwd(), b_proc.fwd() ):
        a = _hotswap_block(b_in)
        b_out[:] = fn(*a, **kwargs)
    return a_proc
