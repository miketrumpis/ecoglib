import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as nfft

import ecoglib.util as ut
import ecoglib.numutil as nut
from sandbox.expo import StimulatedExperiment
import sandbox.array_split as array_split

# define some trigger-locked aggregating utilities
def trigs_and_conds(trig_code):
    if isinstance(trig_code, np.ndarray) or \
      isinstance(trig_code, tuple) or \
      isinstance(trig_code, list):
        trigs, conds = trig_code
    elif isinstance(trig_code, StimulatedExperiment):
        trigs = trig_code.trig_times
        conds, _ = trig_code.enumerate_conditions()
    return trigs, conds

def ep_trigger_avg(
        x, trig_code, pre=0, post=0, 
        sum_limit=-1, iqr_thresh=-1,
        envelope=False
        ):
    """
    Average response to 1 or more experimental conditions

    Arguments
    ---------

    x: data (nchan, npts)

    trig_code: sequence-type (2, stim) or StimulatedExperiment
      First row is the trigger indices, second row is a condition 
      ID (integer). Condition ID -1 codes for a flagged trial to 
      be skipped. If a StimulatedExperiment, then triggers and
      conditions are available from this object.

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

    avg: (nchan, ncond, epoch_length)

    n_avg: number of triggers found for each condition

    skipped: (nskip, nchan, epoch_length) epochs that were not averaged

    """
    x.shape = (1,) + x.shape if x.ndim == 1 else x.shape
    #pos_edge = trig_code[0]; conds = trig_code[1]
    pos_edge, conds = trigs_and_conds(trig_code)
    epoch_len = int( np.round(np.median(np.diff(pos_edge))) )

    n_cond = conds.max()
    n_pt = x.shape[1]

    if not (post or pre):
        post = epoch_len

    (pre, post) = map(int, (pre, post))
    epoch_len = post + pre

    # edit trigger list to exclude out-of-bounds epochs
    while pos_edge[0] - pre < 0:
        pos_edge = pos_edge[1:]
        conds = conds[1:]
    while pos_edge[-1] + post >= n_pt:
        pos_edge = pos_edge[:-1]
        conds = conds[:-1]

    avg = np.zeros( (x.shape[0], n_cond, epoch_len), x.dtype )
    n_avg = np.zeros( (x.shape[0], n_cond), 'i' )

    for c in xrange(1,n_cond+1):
        trials = np.where(conds == c)[0]
        if not len(trials):
            continue
        epochs = extract_epochs(
            x, np.row_stack((pos_edge, conds)), trials, pre, post
            )
        if iqr_thresh > 0:
            pwr = np.sqrt(np.sum(epochs**2, axis=-1))
            # analyze outlier trials per channel
            out_mask = nut.fenced_out(
                pwr, thresh=iqr_thresh, axis=1, low=False
                )
            epochs = epochs * out_mask[:,:,None]
            n_avg[:,c-1] = np.sum(out_mask, axis=1)
        else:
            n_avg[:,c-1] = len(trials)

        avg[:,c-1,:] = np.sum(epochs, axis=1) / n_avg[:,c-1][:,None]

    x.shape = filter(lambda x: x > 1, x.shape)
    return avg, n_avg

def extract_epochs(x, trig_code, selected=(), pre=0, post=0):
    """
    Extract an array of epochs pivoted at the specified triggers.

    Parameters
    ----------

    x : data (n_chan, n_pt)

    trig_code : array (2, n_stim)
      First row is the stim times, second is the condition labeling

    selected : sequencef
      Indices into trig_code for a subset of stims. If empty, return *ALL*
      epochs (*a potentially very large array*)

    pre, post : ints
      Number of pre- and post-stim samples in interval. post + pre > 0
      default: 0 and stim-to-stim interval

    Returns
    -------

    epochs : array (n_chan, n_epoch, epoch_len)

    """
    x.shape = (1,) + x.shape if x.ndim == 1 else x.shape
    pos_edge, conds = trigs_and_conds(trig_code)
    epoch_len = int( np.median(np.diff(pos_edge)) )

    if not (post or pre):
        post = epoch_len
    
    (pre, post) = map(int, (pre, post))
    epoch_len = post + pre
    if len(selected):
        if hasattr(selected, 'dtype') and selected.dtype.char == '?':
            selected = np.where(selected)[0]
        pos_edge = np.take(pos_edge, selected)

    epochs = np.empty( (x.shape[0], len(pos_edge), epoch_len), x.dtype )
    epochs.fill(np.nan)

    for n, k in enumerate(pos_edge):
        idx = (slice(None), slice(k-pre, k+post))
        epochs[:,n,:] = x[idx]

    x.shape = filter(lambda x: x > 1, x.shape)
    return epochs

## from array_split_test import mtm_lite

def psd_trigger_avg(
        x, trig_code, plan, Fs,
        pre=0, post=0, ntaper=2, induced=1, stats=False, units='V^2'
        ):
    
    if abs(pre) < 10:
        pre = int( round(pre*Fs) )
    if abs(post) < 10:
        post = int( round(post*Fs) )
    ncond= plan.n_conds
    nvar = plan.n_var
    nchan = x.shape[0]
    nfft = nut.nextpow2(pre+post)
    spectra = ut.Bunch()
    for name in plan.var_names:
        spectra[name] = np.empty( (nchan, ncond, nfft/2+1), 'd' )
        
    tapers, eigs = ntalg.dpss_windows(pre+post, (ntaper+1)/2., ntaper)
    
    mx_epochs = max([max(map(len, p)) for p in plan.maps])
    #epochs = array_split.shared_ndarray((nchan, mx_epochs, 

    for c, cond_spec in enumerate(plan.walk_conditions()):
        maps = cond_spec.maps
        for n, var in enumerate(maps):
            # extract epochs corresponding to var
            epochs = extract_epochs(
                x, trig_code, selected=var, pre=pre, post=post
                )
            if induced:
                mn = np.mean(epochs, axis=1)
                epochs -= mn[:,None,:]
            # do mtm_psd on var
            # XXX: parallel tasks psd broken
            ## pf = mtm_lite(epochs, tapers, eigs, nfft)
            pf = mtm_wrap(
                epochs,
                NFFT=nfft, jackknife=False, adaptive=False,
                NW=(ntaper+1)/2., Fs=Fs
                )
            # take average over trials
            name = plan.var_names[n]
            pxx = spectra[name]
            pfm = np.mean(pf, axis=1)
            if units.lower()=='db':
                pxx[:,c,:] = 10*np.log10(pfm)
            else:
                pxx[:,c,:] = pfm
        if stats:
            # make statistical comparison of vars against baseline
            # (need to make decision about what baseline is..
            #  e.g. if baseline is an interval, then extract the
            #  correct intervals and to psd here)
            pass
        print 'proc cond', c
    
    spectra.fx = np.linspace(0, Fs/2, nfft/2+1)
    spectra.units = units
        
    return spectra


import nitime.algorithms as ntalg
#@array_split.splits
def mtm_wrap(x, **kwargs):
    r = ntalg.multi_taper_psd(x, **kwargs)
    fx, pxx = r[:2]
    # array splitting is restricted to single output currently
    return pxx

## XXX: keep this for notes, but it's not very fast
## def par_mtm_factory(view, x, **kwargs):
##     fn_str = """
## def mtm_call(x, **kwargs):
##     from nitime.algorithms import multi_taper_psd
##     r = multi_taper_psd(x, **kwargs)
##     return r
##     """
##     view.execute(fn_str, block=True)
##     view.scatter('x', x)
##     view.push(dict(kwargs=kwargs))
##     view.execute('r = mtm_call(x, **kwargs)', block=True)
##     res = view.gather('r', block=True)
##     ## blk = view.block
##     ## view.block = True
##     ## res = view.apply_sync(mtm_call, x, **kwargs)
##     ## res = view.map_sync(mtm_call, x, **kwargs)
##     ## view.block = blk
##     return res
