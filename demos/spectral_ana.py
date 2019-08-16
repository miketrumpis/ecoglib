import numpy as np
import matplotlib.pyplot as pp
import scipy.io as sio
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la
from time import time
import sys
from sklearn.cluster import k_means

from ecoglib.vis import data_scroll
from ecoglib.vis import scatter_scroller
from ecoglib.graph import cknn_graph
from ecoglib.graph import normalize as nrm

from sandbox.load_arr import load_arr

# Parameters
knn = 30; knn_scale = 0; scale = 0.8
self_connected = True
connectivity = False
mutual = False
normalize = True

dfile = '../../data/cat1/2010-05-19_test_41_filtered.mat'
#dfile = '../../data/test_41_demux.mat'

# Load array data
d, shape, Fs, tx, segs = load_arr(dfile)
nrow, ncol = shape
npts = d.shape[0]

# save this for later before any normalization
dmean = d.mean(axis=1)

if normalize:
    # normalize samples
    dn = np.sqrt(np.sum(d**2, axis=1))
    d = d / dn[:,None]

import pyflann
fln = pyflann.FLANN()

print('finding nearest neighbors... ', end=' ')
sys.stdout.flush()
t1 = time()
# dists are squared--annoying
# XXX: must find out the relative error in approximate kNN distance
# versus actual kNN distance
nbs, dists = fln.nn(d, d, num_neighbors=knn+1, checks=2000)
print(time()-t1, 'sec')

print('constructing adjacency matrix... ', end=' ')
sys.stdout.flush()
t1 = time()
if connectivity:
    if self_connected:
        W = cknn_graph.knn_graph(nbs, mutual=mutual)
    else:
        W = cknn_graph.knn_graph(nbs[:,1:], mutual=mutual)
else:
    if self_connected:
        W = cknn_graph.knn_graph(
            nbs, np.sqrt(dists),
            scale=scale, auto_scale=knn_scale, mutual=mutual
            )
    else:
        W = cknn_graph.knn_graph(
            nbs[:,1:], np.sqrt(dists[:,1:]),
            scale=scale, auto_scale=knn_scale, mutual=mutual
            )
print(time() - t1, 'sec')

print('normalizing graph and finding spectrum... ', end=' ')
sys.stdout.flush()
t1 = time()
W.sort_indices()
K = nrm.anisotropic(W)
deg = K.sum(1)
Di = sparse.dia_matrix( (1/np.sqrt(deg.A.T), [0]), W.shape )
M = nrm.bimarkov(K)
# find largest eigs of symmetric matrix M
w, V = sp_la.eigsh(M, k=20, which='LM')
# these eigs are related to markov(K) by a normalization
V = Di*V[:,::-1]
print(time() - t1, 'sec')

## centroids, labels, r = k_means(V[:,1:], 6)
centroids, labels, r = k_means(V[:,1:10], 5)

# This launches array scrolling GUI
## scr = data_scroll.DataScroller(d, dmean, rowcol=shape, Fs=Fs, tx=tx)

# This launches scatter scrolling GUI
## scr = scatter_scroller.ScatterScroller(V[:,[1,2,3]], dmean, Fs=Fs)

# This launches class-coded scroller
## scr = scatter_scroller.ClassCodedScatterScroller(
##     V[:,[1,2,3]], dmean, labels, Fs=Fs
##     )

## scr.configure_traits()
