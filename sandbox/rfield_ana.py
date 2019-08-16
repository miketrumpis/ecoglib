
import numpy as np
from scikits.bootstrap import ci as boots_ci

import sandbox.trigger_fun as tfun
from ecoglib.util import Bunch

def rfield_snr(x, y=None):
    n = len(x)
    if y is None:
        n = n//2
        y = x[:n]
        x = x[n:2*n]
    return np.var(y, axis=0) / np.var(x, axis=0)

def divide_sample(x):
    n = len(x)
    n = n//2
    return np.mean(x[:n] / x[n:2*n])

def rfield_rms(x, idx=()):
    if idx:
        return np.sqrt( np.mean( x[...,idx]**2, axis=-1 ) )
    else:
        return np.sqrt( np.mean( x**2, axis=-1 ) )

def test_for_rf(
        dataset, exp_plan, epoch_itvl, alpha=0.01, rf_func=rfield_rms
        ):
    """
    Parameters
    ----------
    
    dataset : Bunch with trig_coding and data fields
    exp_plan : an ExperimentPlan that maps conditions and variations
    epoch_itvl : a tuple of (pre, post) stim points
    alpha : confidence interval bounds
    rf_func : function
      This function computes response fields from raw responses by
      performing some possible transformation and reduction on the 
      last axis of the response data. Examples are time-domain RMS
      power, or bandpass frequency domain power.


    """
    # could potentially add kwarg for doing PSDs and intregrating within band
    d_arr = dataset.data
    if d_arr.shape[0] > d_arr.shape[1]:
        d_arr = d_arr.T
    trigs = dataset.trig_coding
    n_sites = d_arr.shape[0]

    ratio_itvls = np.zeros((2, n_sites))
    
    pre, post = epoch_itvl
    n_cond = exp_plan.n_conds
    n_var = exp_plan.n_var
    n_pt = pre+post

    # xxx: this seems funky
    n_trial = max( [max(list(map(len, c_maps))) for c_maps in exp_plan.maps] )

    n_test = len(exp_plan.active)
    # these arrays are samples of the integrated field response
    field_var = np.zeros( (n_test, n_trial, n_sites) )
    baseline_var = np.zeros( (n_test, n_trial, n_sites) )
    # these arrays are trial-averages of the field responses
    field_mn = np.zeros( (n_test, n_sites, n_cond) )
    baseline_mn = np.zeros( (n_test, n_sites, n_cond) )

    
    # finally check to see if baseline is one of the vars,
    # or needs to be computed from pre-stim.
    # If prestim baseline is used, then the baseline samples are 
    # interleaved with the active condition samples
    if isinstance(exp_plan.baseline, str):
        prestim_baseline = True
        samps = np.zeros( (2*n_var, n_cond, n_trial, n_pt) )
        Fs = ml_data.Fs
        bpre, bpost = [int(round(x*Fs)) for x in exp_plan.b_itvl]
        bpre = -bpre
    else:
        prestim_baseline = False
        samps = np.zeros( (n_var, n_cond, n_trial, n_pt) )
   
    for n in range(n_sites):
        # gather samps
        for c, cond in enumerate(exp_plan.walk_conditions()):
            for v_idx in range(n_var):
                v_samps = tfun.extract_epochs(
                    d_arr[n], trigs, selected=cond.maps[v_idx], 
                    pre=pre, post=post
                    )
                if prestim_baseline:
                    b_samps = tfun.extract_epochs(
                        d_arr[n], trigs, selected=cond.maps[v_idx], 
                        pre=bpre, post=bpost
                        )
                    samps[2*v_idx, c] = v_samps.squeeze()
                    samps[2*v_idx+1, c] = b_samps.squeeze()
                else:
                    samps[v_idx, c] = v_samps.squeeze()
        
        if prestim_baseline:
            control_samps = samps[1::2]
        else:
            control_samps = samps[exp_plan.baseline]
        
        # reduce to the response field
        samp_pwr = rf_func(samps)

        if prestim_baseline:
            active_map = [ 2*a for a in exp_plan.active ]
            baseln_map = [ 2*a+1 for a in exp_plan.active ]
            active_pwr = samp_pwr[ active_map ]
            baseln_pwr = samp_pwr[ baseln_map ]
        else:
            active_pwr = samp_pwr[ list(exp_plan.active) ]
            baseln_pwr = samp_pwr[ exp_plan.baseline ]
            # make the dimensions consistent: if necessary, duplicate 
            # a single baseline set for each active set
            if baseln_pwr.ndim < active_pwr.ndim:
                baseln_pwr = np.tile(
                    baseln_pwr, (active_pwr.shape[0],) + (1,)*baseln_pwr.ndim
                    )

        # finally, integrate over the field of conditions
        field_var[:,:,n] = np.var(active_pwr, axis=1)
        baseline_var[:,:,n] = np.var(baseln_pwr, axis=1)
        # and save the trial-average field response
        field_mn[:,n,:] = np.mean(active_pwr, axis=2)
        baseline_mn[:,n,:] = np.mean(baseln_pwr, axis=2)
        
            
            
    # now we can perform the bootstrap ratio test in parallel over all sites
    res = Bunch()
    #ratio_fn = lambda x,y: np.mean(x/y, axis=0)
    ratio_fn = lambda x,y: np.mean(x, axis=0)/np.mean(y, axis=0)
    for t in range(n_test):
        active = field_var[t]
        baseline = baseline_var[t]
        ci = boots_ci( 
            (active, baseline), ratio_fn, alpha=alpha
            )
        name = exp_plan.var_names[exp_plan.active[t]]
        res[name+'_ci'] = ci
        res[name+'_snr'] = ratio_fn(active, baseline)
        res[name+'_field'] = field_mn[t]
        res[name+'_ctrl'] = baseline_mn[t]
    res['test_names'] = (exp_plan.var_names[a] for a in exp_plan.active)
    res['alpha'] = alpha
    return res


def plot_snr_results(rf_results):
    import matplotlib.pyplot as pp
    for test in rf_results.test_names:
        snr = rf_results[test+'_snr']
        ci = rf_results[test+'_ci']
        pp.figure(figsize=(8,4))
        x = np.arange(len(snr))

        above_1 = ci[0] > 1
        pp.scatter(x[above_1], snr[above_1], facecolor='r')
        pp.scatter(x[~above_1], snr[~above_1], facecolor='b')
        err = np.abs(ci - snr)
        pp.errorbar(x, snr, yerr=err, ecolor='g', elinewidth=1, mfc='g')
        pp.title('SNR results for %s receptive fields'%test)
        pp.xlabel('site #')
        cinv = (1 - 2*rf_results.alpha)*100
        pp.ylabel('SNR +/- %2.1f%% conf itvl'%cinv)
        pp.xlim(-10, len(snr)+10)
        pp.axhline(y=1, linestyle='--', color='k')
        
