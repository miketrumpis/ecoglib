from functools import partial
import numpy as np

from ecogdata.parallel.split_methods import multi_taper_psd
from ecogdata.datasource import ElectrodeDataSource
from ecogdata.filt.blocks import BlockSignalBase
from ecogdata.filt.time import ar_whiten_blocks
from ecogdata.util import fenced_out, nextpow2
from ecogdata.parallel.mproc import multiprocessing as mp
from ecogdata.expconfig import load_params

from ecoglib.estimation.spatial_variance import ergodic_semivariogram


__all__ = ['band_power', 'bad_channel_mask', 'block_psds', 'logged_estimators', 'safe_corrcoef', 'safe_avg_power',
           'spatial_autocovariance']


def band_power(f, pf, fc=None, root_hz=True):
    """
    Sum of band power in a power spectral density estimate.
    Parameters
    ----------
    f
    pf
    fc
    root_hz

    Returns
    -------
    P: float
        Integral of spectrum up to cutoff frequency.

    """
    # freq axis must be last axis
    # and assuming real and one-sided
    f_mask = f < fc if fc else slice(None)
    p_slice = pf[..., f_mask]
    if root_hz:
        igrl = np.nansum(p_slice ** 2, axis=-1)
    else:
        igrl = np.nansum(p_slice, axis=-1)

    # apply df (assuming regular freq grid)
    df = f[1] - f[0]
    return igrl * df


def bad_channel_mask(pwr_est, iqr=4.0, **kwargs):
    """
    Create a channel mask based on hard coded heuristics and outlier detection. See comments for details.

    Parameters
    ----------
    pwr_est: ndarray
        Power (squared or RMS) estimates for each channel. Will be log-transformed if all values > 0.
    iqr: float
        Multiple of the interquartile range defining the inlier/outlier fence in ecogdata.util.fenced_out
    kwargs: dict
        Other arguments for fenced_out method.

    Returns
    -------
    mask: ndarray
        Binary channel mask such that mask[i] = False means channel i should be discarded.

    """
    # estimated power should be log-transformed
    if np.all(pwr_est >= 0):
        pwr_est = np.log(pwr_est)

    # first off, throw out anything pushing numerical precision
    m = pwr_est > np.log(1e-8)
    # automatically reject anything 2 orders of magnitude
    # lower than the median broadband power
    m[pwr_est < (np.median(pwr_est[m]) - np.log(1e2))] = False
    # now also apply a fairly wide outlier rejection
    kwargs.setdefault('quantiles', (25, 75))
    kwargs.setdefault('thresh', iqr)
    kwargs.setdefault('fences', 'both')
    msub = fenced_out(pwr_est[m], **kwargs)
    m[m] = msub
    return m


def make_block_generator(data, block_size, block_for_channels=True):
    """
    Prepare a block generator from data, which may be

    * ndarray that is pre-shaped as blocks or is a matrix
    * BlockSignalBase -- already is a block iterator
    * ElectrodeDataSource -- knows how to issue its own blocks

    Parameters
    ----------
    data: ndarray or BlockSignalBase or ElectrodeDataSource
        Data to block and yield
    block_size: int
        Size of blocks
    block_for_channels: bool
        If True, each block much be sized (nchan, block_size).
        Otherwise may yield >n_chan blocks at a time. In this case
        the block sequence will always be reshapable as (nblock, nchan)

    Returns
    -------
    block_itr: iterator
        block iterator
    nchan: int
        size of the channels dimension
    nblock: int
        number of blocks to expect
    block_size: int
        block size

    """
    if isinstance(data, BlockSignalBase):
        block_itr = data
        nblock = len(data)
        nchan = data.array_shape[0]
    elif isinstance(data, ElectrodeDataSource):
        # if data is a data source, then it can generate blocks internally
        block_itr = data.iter_blocks(block_length=block_size, not_strided=True)
        nchan = len(data)
        nblock = len(block_itr)
    else:
        # otherwise data is an array, which can can be shaped to iterate multiple blocks at a time
        if data.ndim > 2:
            nblock, nchan, block_size = data.shape
            if block_for_channels:
                blk_data = data
            else:
                blk_data = data.reshape(nblock * nchan, block_size)
        else:
            nchan, npt = data.shape
            nblock = npt // block_size
            blk_data = data[..., :block_size * nblock].reshape(-1, nblock, block_size)
            blk_data = blk_data.transpose(1, 0, 2).copy()
            if not block_for_channels:
                blk_data = blk_data.reshape(nblock * nchan, block_size)
        block_itr = blk_data
    return block_itr, nchan, nblock, block_size


def block_psds(data, btime, Fs, max_blocks=-1, **mtm_kw):
    """
    Parameters
    ----------
    data: 2D or 3D array
        Array with timeseries in the last axis. If 3D, then it
        has already been cut into blocks, indexed in the first dimension
        (or second dimension, if old_blocks==True).
    btime: float
        Length (in seconds) of the blocking.
    Fs: float
        Sampling frequency
    max_blocks: int
        Limits the number of blocks computed (-1 for no limit).
    mtm_kw: dict
        Keywords for multi_taper_psd, e.g. NW (reasonable defaults will be used).

    Returns
    -------
    freqs: array
        Frequency bins
    psds: ndarray
        Power spectral density of blocks.
    """

    if 'jackknife' not in mtm_kw:
        mtm_kw['jackknife'] = False
    if 'adaptive' not in mtm_kw:
        mtm_kw['adaptive'] = False
    if 'NW' not in mtm_kw:
        mtm_kw['NW'] = 2.5

    bsize = int(round(Fs * btime))
    block_itr, nchan, nblock, bsize = make_block_generator(data, bsize, block_for_channels=False)
    nfft = nextpow2(bsize)
    if not isinstance(data, ElectrodeDataSource):
        # If data is not a data source, then control how many blocks are processed simultaneously.
        # Try to keep computation chunks to modest sizes.. 6 GB
        # so (2*NW) * nfft * nchan * sizeof(complex128) * comp_blocks < 3 GB
        n_tapers = 2.0 * mtm_kw['NW']
        global_params = load_params()
        mem_limit = global_params.memory_limit / 2.0
        comp_blocks = mem_limit / n_tapers / nfft / (2 * data.dtype.itemsize)
        comp_blocks = max(1, int(comp_blocks))
        n_comp_blocks = nchan * nblock // comp_blocks + int((nchan * nblock) % comp_blocks > 0)
        block_itr = np.array_split(block_itr, n_comp_blocks, axis=0)

    psds = list()

    for n, blocks in enumerate(block_itr):
        freqs, psds_, _ = multi_taper_psd(
            blocks, NFFT=nfft, Fs=Fs, **mtm_kw
        )
        # unwrap into blocks x chans x pts
        psds.append(psds_)
        if max_blocks > 0 and n + 1 >= max_blocks:
            break
    psds = np.concatenate(psds, axis=0)
    return freqs, psds.reshape(-1, nchan, psds.shape[1])


def logged_estimators(psds, sem=True):
    """
    Compute center and dispersion statistics based on logged (normalizing transformation) PSDs returned by block_psd.

    Parameters
    ----------
    psds: ndarray
        Output of block_psd with shape (blocks, channels, freq)
    sem: bool
        If True, dispersion is standard error of the mean

    Returns
    -------
    avg_psds: ndarray
        Average per channel
    grand_avg: ndarray
        Average over channels
    upper: ndarray
        Upper range of dispersion, calculated as exp(grand_avg + dispersion)
    lower: ndarray
        Lower range of dispersion.

    """

    # *** used with block_psds above ***
    # returns (blockwise mean, channel mean, channel mean +/- stderr)
    lg_psds = np.mean(np.log(psds), axis=0)
    mn = np.mean(lg_psds, axis=0)
    err = np.std(lg_psds, axis=0)
    if sem:
        err = err / np.sqrt(lg_psds.shape[0])
    return list(map(np.exp, (lg_psds, mn, mn - err, mn + err)))


def safe_avg_power(data, bsize=2000, iqr_thresh=3.0, mean=True, mask_per_chan=False):
    """
    Calculate robust RMS power over blocks of data. Discard outlying blocks.

    Parameters
    ----------
    data: ndarray
        If 2D, then data will be blocked every bsize points. If 3D, the first dimension iterates over blocks.
        RMS power is calculated over the last axis
    bsize: int
        Block size in samples
    iqr_thresh:
        Multiple of the IQR defining inlier/outlier fence
    mean: bool
        If True, return the robust average, else return power for all blocks.
    mask_per_chan: bool
        If True, treat each channel as a separate population for outlier detection.

    Returns
    -------
    rms_vals: ndarray
        Robust average of (or all) RMS blocks.

    """
    iterator, nchan, nblock, bsize = make_block_generator(data, bsize)
    rms_vals = np.zeros((nblock, nchan))
    for n, blk in enumerate(iterator):
        rms_vals[n] = blk.std(axis=-1)
    # If mask_per_chan is True, then evaluate outliers relative to
    # each channel's samples. Otherwise eval outliers relative to the
    # whole sample.
    axis = 0 if mask_per_chan else None
    if not mean:
        return rms_vals
    omask = fenced_out(rms_vals, thresh=iqr_thresh, axis=axis)
    rms_vals[~omask] = np.nan
    return np.nanmean(rms_vals, axis=0)


def safe_corrcoef(data, bsize=2000, iqr_thresh=3.0, mean=True, normed=True, semivar=False, ar_whiten=False):
    """
    Calculate robust covariance matrix over blocks of data. Outlier detection is based on the Frobenius norm of the
    blockwise matrices.

    Parameters
    ----------
    data: ndarray
        If 2D, then data will be blocked every bsize points. If 3D, the first dimension iterates over blocks.
        Covariance is calculated over the last axis.
    bsize: int
        Block size in samples.
    iqr_thresh:
        Multiple of the IQR defining inlier/outlier fence
    mean: bool
        If True, return the robust average, else return power for all blocks.
    normed: bool
        If True, normalize the covariance/semivariance between channels (i, j) by sqrt(var(i)) * sqrt(var(j))
    semivar: bool
        If True, calculate a semivariance matrix
    ar_whiten: bool
        If True, reduce temporal autocorrelation by using an autoregression residual.

    Returns
    -------
    cxx: ndarray
        Covariance/correlation/semivariance matrix.

    """
    iterator, nchan, nblock, bsize = make_block_generator(data, bsize)
    cxx = np.zeros((nblock, nchan, nchan))
    results = []
    if not semivar:
        fn = np.corrcoef if normed else np.cov
    else:
        fn = partial(ergodic_semivariogram, normed=normed, mask_outliers=False)
        pool = mp.Pool(min(8, mp.cpu_count()))
    for n, blk in enumerate(iterator):
        if ar_whiten:
            blk = ar_whiten_blocks(blk)
        if semivar:
            results.append(pool.apply_async(fn, (blk,)))
        else:
            cxx[n] = fn(blk)
    if semivar:
        pool.close()
        pool.join()
        for n, r in enumerate(results):
            cxx[n] = r.get()

    cxx_norm = np.nansum(np.nansum(cxx ** 2, axis=-1), axis=-1)
    mask = fenced_out(cxx_norm, thresh=iqr_thresh)
    if not mean:
        return cxx
    return np.nanmean(cxx[mask], axis=0)


def spatial_autocovariance(covar_matrix, channel_map, mean=True):
    """
    Calculate a 2D spatial autocovariance map.

    Parameters
    ----------
    covar_matrix: ndarray
        May be a square covariance/semivariance matrix, or simply the upper triangular values of such a
        channel-channel correspondence matrix. If upper-triangular values are given, the center of each map will be
        zero, rather than the appropriate autovariance quantity. In either case, further dimensions are allowed
        (e.g. spectral coherence).
    channel_map: ChannelMap
        The channel map to locate data channels on an electrode array
    mean: bool
        If True, return the average covariance map, else return a stack of maps for n channels.

    Returns
    -------
    covar_map: ndarray
        Single or stack of autocovariance maps.

    """

    n = len(channel_map)
    if len(covar_matrix) == n * (n - 1) / 2:
        covar_matrix_square = np.zeros((n, n) + covar_matrix.shape[1:])
        covar_matrix_square[np.triu_indices(n, k=1)] = covar_matrix
        covar_matrix = covar_matrix_square + covar_matrix_square.conj().transpose()

    gr, gc = channel_map.geometry
    rows, cols = channel_map.to_mat()
    mi = rows.max()
    mj = cols.max()
    offset = np.array([mi, mj])

    covar_maps = np.zeros((n, 2 * mi + 1, 2 * mj + 1) + covar_matrix.shape[2:])
    covar_maps.fill(np.nan)
    for n, row in enumerate(covar_matrix):
        # position this channels map so that it's (i, j) location lies in the center of autocovar map
        cov = channel_map.embed(row)
        i = offset[0] - rows[n]
        j = offset[1] - cols[n]
        covar_maps[n, i:i + gr, j:j + gc] = cov
    if mean:
        return np.nanmean(covar_maps, axis=0)
    return covar_maps





