import numpy as np
import scipy.signal as signal

from ecoglib.filt.time import blocked_filter

from ecoglib.ssc import admm

import sandbox.electrode_corrections as electrode_corrections
import sandbox.load_arr as load_arr

# Parameters
knn = 10; knn_scale = 8; scale = 0.8
self_connected = True
connectivity = False
mutual = False
normalize = True

tau = 0.1
overlap_factor = 2/3.


#dfile = '../../data/cat1/2010-05-19_test_41_filtered.mat'
dfile = '../../data/cat1/test_41_demux.mat'
#dfile = '../../data/cat1/test_40_demux.mat'

d, shape, Fs, tx, segs = load_arr.load_arr(dfile, auto_prune=True)
nrow, ncol = shape
N, M0 = d.shape
post_pruned = load_arr.get_post_snips(dfile)

if dfile.find('test_41') >= 0:
    [bn, an] = signal.iirdesign(
        2*np.array([117, 123], 'd')/Fs, 2*np.array([119., 121.])/Fs,
        1, 30, ftype='butter'
        )
    # simple butterworth seems to work out best! (after notch filter)
    [bb, ab] = signal.iirfilter(6, 2*np.array([1, 100])/Fs)
    d = signal.filtfilt(bn, an, d, axis=0)
    d = signal.filtfilt(bb, ab, d, axis=0)

elif dfile.find('test_40') >= 0:
    # do some block-proc
    [bn, an] = signal.iirdesign(
        2*np.array([57, 63], 'd')/Fs, 2*np.array([59., 61.])/Fs,
        1, 30, ftype='butter'
    )
    blocked_filter.bfilter(bn, an, d, bsize=20000, axis=0)

if post_pruned:
    d, segs = load_arr.pruned_arr(d, post_pruned, axis=0)
else:
    segs = ()

if dfile.find('test_40') >= 0:
    blocked_filter.bdetrend(
        d, int(np.round(Fs*10)), axis=0, type='linear', bp=segs
        )

d.shape = (-1, ncol, nrow)
d = electrode_corrections.correct_for_channels(d, dfile, blocks=20000, bw=2)
d.shape = (-1, ncol*nrow)

##### Reshape array to be lagged

bitdepth = d.dtype.itemsize
L = int(np.round( Fs*tau*(1-overlap_factor) ))
tau_fs = L * int(np.round( 1.0/(1-overlap_factor) ))

#tau_fs = 50
M1 = M0 * tau_fs
#N1 = int(np.floor( N/(1-overlap_factor)/tau_fs ))
N1 = int((N-tau_fs) / L)
N_end = N1*L + tau_fs

d_lag = np.lib.stride_tricks.as_strided(
    d[:N_end,:], shape=(N1, M1),
    strides=(bitdepth*L*M0, bitdepth)
    )
d_lag = d_lag.T.copy()

d_lag /= np.sqrt(np.sum(d_lag**2, axis=0))

d_lag_trn = d_lag[:,:4000]
d_lag_tst = d_lag[:,4000:]

YtY = d_lag_trn.T.dot(d_lag_trn)
YYt = d_lag_trn.dot(d_lag_trn.T)

muz, mue = admm.auto_mu(d_lag_trn, YtY=YtY)

C = admm.admm_one(d_lag_trn, 5/muz, 5/mue, 20.0, YYt=YYt, YtY=YtY, max_it=100)
