import numpy as np
import matplotlib.pyplot as pp
import scipy.io as sio
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.spatial.distance as dist
from time import time
import sys, os
from sklearn.cluster import k_means, mean_shift, estimate_bandwidth

import ecoglib.vis.ani as ani
from ecoglib.vis import data_scroll
from ecoglib.vis import scatter_scroller
from ecoglib.vis import combo_scroller
from ecoglib.vis import script_plotting as splot
from ecoglib.vis import single
from ecoglib.graph import cknn_graph
from ecoglib.graph import normalize as nrm
from ecoglib.filt.time import bfilter, butter_bp, notch

import sandbox.electrode_corrections as electrode_corrections
import ecoglib.data.jon_data as load_arr
import sandbox.spikes as spikes
import sandbox.tile_images as tile_images

# Parameters
eps = 1.0
knn = 20; knn_scale = 6; scale = 0.8
self_connected = True
connectivity = False
mutual = False
normalize = True
sparse_nn = False

spike_detection_window = 100e-3 #180e-3
spike_refractory = 145e-3 # 160 ms refractory period from NNeuro
spike_width = 160e-3 # 160 ms spike-length from NNeuro
pre_spike = 60e-3 # 60 ms pre-spike from NNeuro

#dfile = '/Users/mike/experiment_data/cat1/2010-05-19_test_41_filtered.mat'
dfile = '/Users/mike/experiment_data/cat1/test_41_demux.mat'
#dfile = '/Users/mike/experiment_data/test_40_demux.mat'
#dfile = '/Users/mike/experiment_data/cat1/test_40_filtered2.mat'
#dfile = '/Users/mike/experiment_data/cat1/test_40_demux_filtered.mat'

dset = load_arr.load_arr(dfile, auto_prune=True)
d = dset.data
nrow, ncol = dset.rowcol
Fs = dset.Fs
tx = dset.tx
segs = dset.segs

post_pruned = load_arr.get_post_snips(dfile)

if dfile.find('filter') < 0:
    (b, a) = butter_bp(lo=2, hi=50, Fs=Fs)
    bfilter(b, a, dset.data, axis=0, filtfilt=True)
    d = dset.data
    d.shape = (-1, ncol, nrow)
    d = electrode_corrections.correct_for_channels(d, dfile, blocks=20000)
    d.shape = (-1, ncol*nrow)
    
if post_pruned:
    d, segs = load_arr.pruned_arr(d, post_pruned, axis=0)
    tx, _ = load_arr.pruned_arr(tx, post_pruned)
else:
    segs = ()

dmean = d.mean(1)

## sp = spikes.find_spikes(
##     d, spike_detection_window, Fs, spikewindow=spike_width
##     )
spk_times = spikes.simple_spikes(d, 5e-4, int(spike_refractory * Fs), 1)
spk_times = [ (st - int(pre_spike*Fs), 
               st - int(pre_spike*Fs) + int(spike_width*Fs))
               for st in spk_times ]
spike_pts = spk_times[0][1] - spk_times[0][0]
sp_vecs = np.zeros((len(spk_times), spike_pts*nrow*ncol))
sp_vecs_nb = np.zeros((len(spk_times), spike_pts*nrow*ncol))

for n, zz in enumerate(spk_times):
    sp_vecs[n,:] = d[zz[0]:zz[1],:].ravel()

if normalize:
    # project onto a N-1 dimensional hypersphere
    sp_vecs_sv = sp_vecs.copy()
    sp_vecs_n = np.sqrt(np.sum( sp_vecs**2, axis=1 ))
    sp_vecs /= sp_vecs_n[:,None]

if sparse_nn:
    import pyflann
    fln = pyflann.FLANN()
    print 'finding nearest neighbors graph... ',
    sys.stdout.flush()
    t1 = time()
    p = fln.build_index(sp_vecs, algorithm='autotuned')
    nbs, dists = fln.nn_index(
        sp_vecs, num_neighbors=knn+1, checks=p['checks']
        )

    W = cknn_graph.knn_graph(
        nbs[:,:knn+1], np.sqrt(dists[:,:knn+1]), auto_scale=knn_scale
        )
    W.sort_indices()
    print time() - t1, 'sec'

else:
    dists = dist.pdist(sp_vecs, 'sqeuclidean')
    n = sp_vecs.shape[0]
    ui, uj = np.triu_indices(n, k=1)
    W = np.zeros((n,n))
    W[ui, uj] = dists
    W = W + W.T
    W = np.exp(-W / eps**2)
    W = np.matrix(W)    

print 'normalizing graph and finding spectrum... ',
sys.stdout.flush()
t1 = time()
W = nrm.anisotropic(W, alpha=1.0)
M = nrm.bimarkov(W)
# find largest eigs of symmetric matrix M
w, V = sp_la.eigsh(M, k=100, which='LM')
# these eigs are related to markov(K) by a normalization
Dh = nrm.degree_matrix(W, p=-1/2.)
V = Dh*V[:,:-1][:,::-1]
w = w[:-1][::-1]
print time() - t1, 'sec'

def fill_in(Vk, labels, n, spike_itvls):
    l_spikes = np.zeros(n)
    diff_coords = np.zeros( (n, Vk.shape[1]), 'd' )
    diff_coords.fill(np.nan)
    for m, zz in enumerate(spike_itvls):
        l_spikes[zz[0]:zz[1]] = labels[m]+1
        diff_coords[zz[0]:zz[1]] = Vk[m]
    return l_spikes, diff_coords

# find the number of dims by looking at the relative accuracy of 
# "diffusion" distance calculations in embedding space for a four-step
# diffusion process.
ndim = np.sum( w**4 > 0.01 * w[0]**4 )
feature_vecs = V[:,:ndim]

# with Mean Shift
ms_bw = estimate_bandwidth(feature_vecs, 0.1)    
centroids, labels = mean_shift(feature_vecs, ms_bw)

# with K-Means / K-Medians, check gap stat 
## import sandbox.gapstat as gapstat
## gk, sk = gapstat.gap_stat(feature_vecs, 28, p=2, nsurr=20)
## gap_closed = np.where(gk[:-1] > (gk[1:] - sk[1:]))[0][0]
## # find 1st gap closing that is > 0 (spurious result)
## gap_closed = gap_closed[gap_closed > 0][0]
## centroids, labels, r = k_means(feature_vecs, gap_closed+1, n_init=50)

l_spikes, dcoords = fill_in(feature_vecs, labels, d.shape[0], spk_times)
dscr = data_scroll.ClassCodedDataScroller(
    d, -dmean, l_spikes, rowcol=(nrow, ncol), Fs=Fs, ts_page_length=12
    )
dscr.configure_traits()

lims = pp.mlab.prctile(sp_vecs.ravel(), p=(1, 99))
lim = np.abs(lims).max()
lim = 0.003
norm = pp.normalize(-lim, lim)
tx = np.arange(spike_pts) / Fs

#ana_name = 'cat_sz_spike_ana_meanshift'
ana_name = 'cat_sz_spike_ana_temp'
try:
    os.mkdir(ana_name)
except:
    pass

with splot.ScriptPlotter('.', ana_name, dpi=200) as spt:
    f = single.subspace_scatters(
        feature_vecs, s=20, edgecolors='white', linewidths=0.5, 
        oneplot=True,
        )
    spt.savefig(f, 'spike_embedding_coords')
    f = single.subspace_scatters(
        feature_vecs, s=20, labels=labels, edgecolors='white', 
        linewidths=0.5, oneplot=True
        )
    spt.savefig(f, 'spike_embedding_coords_clustered')
    for n in xrange(labels.max()+1):
        spk_set = sp_vecs_sv[labels==n]
        repr_feature = np.argmin( 
            np.sum( (feature_vecs - centroids[n])**2, axis=1 )
            )
        spk = sp_vecs_nb[repr_feature]
        spk.shape = (spike_pts,ncol,nrow)
        mn_spk = spk.mean(axis=-1).mean(axis=-1)

        s_ani = ani.animate_frames_and_series(
            spk, mn_spk, tx=tx, title='Spike Exemplar %d'%(n+1,),
            interp=4,
            imshow_kw=dict(clim=(-lim, lim)), 
            line_props=dict(color='r', linewidth=3)
            )
        s_ani.repeat = False
        
        fig, func = ani.dynamic_frames_and_series(
            spk, mn_spk, tx=tx, title='Spike Exemplar %d'%(n+1,),
            interp=4,
            imshow_kw=dict(clim=(-lim, lim)), 
            line_props=dict(color='r', linewidth=3)
            )
        movie_name = os.path.join('.', ana_name)
        movie_name = os.path.join(movie_name, 'spike_waveform_%02d.mp4'%(n+1,))
        ani.write_anim(
            movie_name, fig, func, spk.shape[0],
            quicktime=True,
            title='Spike Waveform %02d'%(n+1,)
            )
        
        movie_name = os.path.join('.', ana_name)
        movie_name = os.path.join(
            movie_name, 'group_%02d_concatenated.mp4'%(n+1,)
            )
        ani.write_frames(
            spk_set.reshape(spk_set.shape[0]*spike_pts, ncol, nrow), 
            fname=movie_name, quicktime=True, fps=10, 
            clim=(spk_set.min(), spk_set.max())
            )

        lags = spikes.delay_map(spk_set, nrow*ncol)
        lag_map = 1000 * lags.mean(axis=0).reshape(ncol, nrow) / Fs
        f = pp.figure()
        pp.imshow(lag_map-lag_map.min(), cmap=pp.cm.jet); pp.axis('image'); 
        cbar = pp.colorbar()
        cbar.set_label('delay of spike peak (ms)')
        pp.title('Delay Map For Spike Group %02d'%(n+1,))
        spt.savefig(f, 'delay_map_%02d'%(n+1,))

        fig, axes = tile_images.quick_tiles(lags.shape[0], ncol=10)
        clim = (lags.min(), lags.max())
        spike_nums = pp.mlab.find(labels==n) + 1
        for num, ax, lag in zip(spike_nums, axes, lags):
            ax.imshow(lag.reshape(ncol, nrow), clim=clim, cmap=pp.cm.jet)
            ax.axis('image'); ax.axis('off')
            ax.set_title('spike %03d'%num, fontsize=8, va='center')
            m += 1
        spt.savefig(fig, 'all_delays_%02d'%(n+1,))
    
