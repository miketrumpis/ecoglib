import numpy as np
import scipy.ndimage as ndimage

from ..blocks import BlockedSignal

def pixel_corrections(
        arr, bad_lines, bad_pix,
        ksize=3, bw=np.inf, blocks=0
        ):
    """
    Parameters
    ----------
    arr: ndarray shaped (nsamp, ncol, nrow)

    bad_lines: sequence
      A pair of lists (bad_row, bad_col)

    bad_pix: sequence
      A sequence of array coordinates indexing bad pixels

    blocks: int (optional)
      If blocks > 0, then use a signal blocking object to avoid making
      a full memory copy of the array.
    
    """
    # all indices are transposed
    nsamp, ncol, nrow = arr.shape
    msk = np.ones((ncol, nrow), 'd')

    br, bc = bad_lines

    for col in bc:
        arr[:,col,:] = 0
        msk[col,:] = 0
    for row in br:
        arr[:,:,row] = 0
        msk[:,row] = 0

    for r, c in bad_pix:
        arr[:,c,r] = 0
        msk[c,r] = 0

    # Create a separable interpolation kernel with ksize points.
    # This kernel is gaussian with a given BW (uniform 1s if bw == infinity)
    if bw == np.inf:
        gf = np.ones(ksize)
    else:
        gf = signal.gaussian(ksize, bw, sym=ksize%2)

    weights = ndimage.convolve1d(msk, gf, axis=-1, mode='constant')
    weights = ndimage.convolve1d(weights, gf, axis=-2, mode='constant')
    weights = 1 / weights

    bsize = blocks if blocks > 0 else arr.shape[0]
    arr_blks = BlockedSignal(arr, bsize, axis=0)

    for blk in arr_blks.fwd():
        arr_i = ndimage.convolve1d(blk, gf, axis=-1, mode='constant')
        arr_i = ndimage.convolve1d(arr_i, gf, axis=-2, mode='constant')
        arr_i *= weights

        for col in bc:
            blk[:,col,:] = arr_i[:,col,:]
        for row in br:
            blk[:,:,row] = arr_i[:,:,row]
        for r, c in bad_pix:
            blk[:,c,r] = arr_i[:,c,r]

    return arr.squeeze()
