import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from ..blocks import BlockedSignal

def pixel_corrections(
        arr, bad_lines, bad_pix,
        ksize=3, bw=np.inf, blocks=0
        ):
    """
    Parameters
    ----------
    arr: ndarray shaped (nsamp, ncol, nrow) or (nrow, ncol, nsamp)

    bad_lines: sequence
      A pair of lists (bad_row, bad_col)

    bad_pix: sequence
      A sequence of array coordinates indexing bad pixels

    blocks: int (optional)
      If blocks > 0, then use a signal blocking object to avoid making
      a full memory copy of the array.
    
    """
    # automatically determine array layout
    shape = arr.shape
    if shape[2] == max(shape):
        nrow, ncol, nsamp = shape
        rdim = 0; cdim = 1; tdim = 2
        c_slice = lambda x: (slice(None), slice(x,x+1), slice(None))
        r_slice = lambda x: (slice(x,x+1), slice(None), slice(None))
        rc_slice = lambda rc: (slice(rc[0],rc[0]+1), slice(rc[1],rc[1]+1),
                               slice(None))
    else:
        # all indices are transposed
        nsamp, ncol, nrow = arr.shape
        rdim = 2; cdim = 1; tdim = 0
        c_slice = lambda x: (slice(None), slice(x,x+1), slice(None))
        r_slice = lambda x: (slice(None), slice(None), slice(x,x+1))
        rc_slice = lambda rc: (slice(None),
                               slice(rc[1],rc[1]+1), slice(rc[0],rc[0]+1))

    msk = np.ones((nrow, ncol), 'd')
    if not (bad_lines or bad_pix):
        print('Nothing to interpolate!')
        return arr
    
    if bad_lines:
        br, bc = bad_lines
        for col in bc:
            s = c_slice(col)
            arr[s] = 0
            msk[:,col] = 0
            ## arr[:,col,:] = 0
            ## msk[col,:] = 0
        for row in br:
            s = r_slice(row)
            arr[s] = 0
            msk[row,:] = 0
            ## arr[:,:,row] = 0
            ## msk[:,row] = 0
    else:
        br = ()
        bc = ()
    if bad_pix:
        for r, c in bad_pix:
            s = rc_slice( (r,c) )
            arr[s] = 0
            msk[r,c] = 0
            ## arr[:,c,r] = 0
            ## msk[c,r] = 0

    if rdim > 0:
        msk = msk.T
    
    # Create a separable interpolation kernel with ksize points.
    # This kernel is gaussian with a given BW (uniform 1s if bw == infinity)
    if bw == np.inf:
        gf = np.ones(ksize) / ksize
    else:
        gf = signal.gaussian(ksize, bw, sym=ksize%2)

    weights = ndimage.convolve1d(msk, gf, axis=-1, mode='constant')
    weights = ndimage.convolve1d(weights, gf, axis=-2, mode='constant')
    zm = (weights == 0)
    weights = 1 / weights
    weights[zm] = 0
    
    bsize = blocks if blocks > 0 else arr.shape[tdim]
    arr_blks = BlockedSignal(arr, bsize, axis=tdim)

    for blk in arr_blks.fwd():
        arr_i = ndimage.convolve1d(blk, gf, axis=rdim, mode='constant')
        arr_i = ndimage.convolve1d(arr_i, gf, axis=cdim, mode='constant')
        if tdim == 0:
            arr_i *= weights
        else:
            arr_i *= weights[:,:,None]
        
        for col in bc:
            s = c_slice(col)
            blk[s] = arr_i[s]
            ## blk[:,col,:] = arr_i[:,col,:]
        for row in br:
            s = r_slice(row)
            blk[s] = arr_i[s]
            ## blk[:,:,row] = arr_i[:,:,row]
        for r, c in bad_pix:
            s = rc_slice( (r,c) )
            blk[s] = arr_i[s]
            ## blk[:,c,r] = arr_i[:,c,r]

    return arr.squeeze()

class Scaler(object):
    def __init__(self, realval):
        self.min = realval.min()
        self.max = realval.max()
    def quantize(self, x, scale=255, round=True):
        q = (x - self.min) / (self.max - self.min) * scale
        return np.round(q) if round else q
    def rescale(self, x, mx, mn=0):
        return (x - mn) / float(mx - mn) * (self.max - self.min) + self.min

import cv2
def inpaint_pixels(img, mask=None, radius=3, method=cv2.INPAINT_TELEA):

    if isinstance(img, np.ma.MaskedArray):
        mask = img.mask
        img = img.data

    elif mask is None:
        return img

    scl = Scaler(img)
    img_fill = cv2.inpaint(
        scl.quantize(img).astype('B'), mask, radius, method
        )
    return scl.rescale(img_fill, 255.0)
    
