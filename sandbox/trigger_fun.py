import numpy as np

# define some trigger aggregating utilities

def fenced_out(samps, quantiles=(25,75), thresh=2.0, axis=None, low=True):

    oshape = samps.shape

    if axis is None:
        # do pooled distribution
        samps = samps.ravel()
    else:
        # roll axis of interest to the end
        samps = np.rollaxis(samps, axis, samps.ndim)

    quantiles = map(float, quantiles)
    qr = np.percentile(samps, quantiles, axis=-1)
    extended_range = thresh * (qr[1] - qr[0])
    high_cutoff = qr[1] + extended_range/2
    low_cutoff = qr[0] - extended_range/2
    if not low:
        out_mask = samps < high_cutoff[...,None]
    else:
        out_mask = (samps < high_cutoff[...,None]) & \
          (samps > low_cutoff[...,None])

    if axis is None:
        out_mask.shape = oshape
    else:
        out_mask = np.rollaxis(out_mask, samps.ndim-1, axis)
    return out_mask


def cond_trigger_avg(x, trig_code, pre=0, post=-1, sum_limit=-1, iqr_thresh=-1):
    """
    Average response to 1 or more experimental conditions

    Arguments
    ---------

    data: (nchan, npts)

    trig_code: array (2, stim)
      First row is the trigger indices, second
      row is a condition ID (integer). Condition
      ID -1 codes for a flagged trial to be skipped

    pre, post: ints
      Number of pre- and post-stim samples in interval. post + pre > 0
      default: 0 and stim-to-stim interval

    sum_limit: int
      Do partial sum up to this many terms

    iqr_thresh: float
      If set, do simple outlier detection on all groups of repeated
      conditions based on RMS power in the epoch interval. The iqr_thresh
      multiplies the width of the inter-quartile range to determine the
      "inlier" range of RMS power.


    Returns
    -------

    x: (nchan, ncond, epoch_length)

    n_avg: number of triggers found for each condition

    skipped: (nskip, nchan, epoch_length) epochs that were not averaged

    """


    pos_edge = trig_code[0]; conds = trig_code[1]
    epoch_len = int( np.round(np.median(np.diff(pos_edge))) )

    n_cond = conds.max()
    n_pt = x.shape[1]

    if post < 0:
        post = epoch_len

    epoch_len = post + pre

    # edit trigger list to exclude out-of-bounds epochs
    while pos_edge[0] - pre < 0:
        pos_edge = pos_edge[1:]
    while pos_edge[-1] + post >= n_pt:
        pos_edge = pos_edge[:-1]

    avg = np.zeros( (x.shape[0], n_cond, epoch_len), x.dtype )
    n_avg = np.zeros( (x.shape[0], n_cond), 'i' )

    for c in xrange(1,n_cond+1):
        trials = np.where(conds == c)[0]
        if not len(trials):
            continue
        epochs = extract_epochs(x, trig_code, trials, pre, post)
        if iqr_thresh > 0:
            pwr = np.sqrt(np.sum(epochs**2, axis=-1))
            # analyze outlier trials per channel
            out_mask = fenced_out(pwr, thresh=iqr_thresh, axis=1, low=False)
            epochs = epochs * out_mask[:,:,None]
            n_avg[:,c-1] = np.sum(out_mask, axis=1)
        else:
            n_avg[:,c-1] = len(trials)

        avg[:,c-1,:] = np.sum(epochs, axis=1) / n_avg[:,c-1][:,None]

    return avg, n_avg

def extract_epochs(x, trig_code, selected, pre=0, post=-1):
    """
    Extract an array of epochs pivoted at the specified triggers.

    Parameters
    ----------

    x : data (n_chan, n_pt)

    trig_code : array (2, n_stim)
      First row is the stim times, second is the condition labeling

    selected : sequencef
      Indices into trig_code for a subset of stims

    pre, post : ints
      Number of pre- and post-stim samples in interval. post + pre > 0
      default: 0 and stim-to-stim interval

    Returns
    -------

    epochs : array (n_chan, n_epoch, epoch_len)

    """

    pos_edge = trig_code[0]
    epoch_len = int( np.median(np.diff(pos_edge)) )

    if post < 0:
        post = epoch_len

    epoch_len = post + pre
    pos_edge = pos_edge[selected]

    epochs = np.empty( (x.shape[0], len(selected), epoch_len), x.dtype )
    epochs.fill(np.nan)

    for n, k in enumerate(pos_edge):
        idx = (slice(None), slice(k-pre, k+post))
        epochs[:,n,:] = x[idx]

    return epochs
