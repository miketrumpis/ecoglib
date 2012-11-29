import numpy as np
import scipy.io as sio
import tables
import os

# these segments are intended for snipping pre-processed data at load
_load_prune_db = dict()
_load_prune_db['cat1.2010-05-19_test_41_filtered'] = (
    (200, 17700), (23000, 92848)
    )
_load_prune_db['cat1.test_40_filtered2'] = (
    (0, 577083), (577362, 582313), (582592, 610094), (610371, 678146)
    )

# these segments are for snipping after filtering raw data
_post_prune_db = dict()
_post_prune_db['cat1.test_41_demux'] = (
    (1000, 17391), (23000, 92848)
    )
_post_prune_db['cat1.test_40_demux'] = (
    (22300, 311265), (311555, 327536), (328000, 372000),
    (416800, 573400), (603000, 650000), (653000, 656000),
    (657000, 659600), (660000, 664600), (666800, 705600),
    (709000, 785600)
    )


def _parse_path(dfile):
    pth, fl = os.path.split(dfile)
    p1, p2 = os.path.split(pth)
    fl, ext = os.path.splitext(fl)
    return p2 + '.' + fl

def get_load_snips(dfile):
    return _load_prune_db.get(_parse_path(dfile), ())

def get_post_snips(dfile):
    return _post_prune_db.get(_parse_path(dfile), ())

def load_arr(dfile, pruned_pts = (), auto_prune = True):
    try:
        m = sio.loadmat(dfile)
        d = m.pop('data')
        Fs = float(m['Fs'][0,0])
        nrow = int(m['numRow'][0,0])
        ncol = int(m['numCol'][0,0])
        del m
    except NotImplementedError:
        d, shape, Fs = load_hdf5_arr(dfile)
        nrow, ncol = shape

    t = d.shape[0] < d.shape[1]
    if not pruned_pts and auto_prune:
        pruned_pts = get_load_snips(dfile)

    tx = np.arange(max(d.shape)) / Fs
    segs = ()
    if pruned_pts:
        #d = d.T[pruned_pts, :nrow*ncol] if t else d[pruned_pts, :nrow*ncol]
        if t:
            d, _ = pruned_arr(d[:nrow*ncol, :].T, pruned_pts, axis=0)
        else:
            d, _ = pruned_arr(d[:, :nrow*ncol], pruned_pts, axis=0)
        tx, segs = pruned_arr(tx, pruned_pts)
    elif min(d.shape) != nrow*ncol:
        # explicitly make contiguous
        d = d.T[:, :nrow*ncol].copy() if t else d[:,:nrow*ncol].copy()
    return d, (nrow, ncol), Fs, tx, segs

def load_hdf5_arr(dfile):
    f = tables.openFile(dfile)
    d = f.root.data
    Fs = f.root.Fs[0,0]
    nrow = int(f.root.numRow[0,0])
    ncol = int(f.root.numCol[0,0])
    ## if pruned_pts:
    ##     d = d[pruned_pts, :nrow*ncol]
    ## else:
    ##     d = d[:, :nrow*ncol]
    d = d[:,:nrow*ncol]
    f.close()
    del f
    return d, (nrow, ncol), Fs

def pruned_arr(arr, snips, axis=-1):
    # snips is (start, stop) pairs
    seg_len = [0]
    for start, stop in snips:
        seg_len.append( stop - start )
    segs = np.cumsum(seg_len)
    new_len = segs[-1]
    new_shape = list(arr.shape)
    new_shape[axis] = new_len

    snip_arr = np.empty( tuple(new_shape), arr.dtype )

    arr_slice = [slice(None)] * arr.ndim
    snip_slice = [slice(None)] * arr.ndim
    for n, pair in enumerate(snips):
        start, stop = pair
        arr_slice[axis] = slice(start, stop)
        snip_slice[axis] = slice(segs[n], segs[n+1])
        snip_arr[snip_slice] = arr[arr_slice]
    return snip_arr, segs
