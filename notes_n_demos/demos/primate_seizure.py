## skip conversion
import numpy as np
import matplotlib.pyplot as pp
import tables

import ecoglib.util as ut
import ecoglib.data.preproc_data as ppd
import ecoglib.vis.data_scroll as data_scroll
import ecoglib.vis.ani as ani
import ecoglib.data.arangement as da
import ecoglib.filt.space.pix_corr as pc

ml_data = ppd.load_preproc('/Users/mike/work/ecog_sz/mlab/joined_data_33.h5')

(arr_data, null_sites) = da.array_geometry(
    ml_data.data, ml_data.egeo, ml_data.emap, axis=1
    )

(ni, nj) = ut.flat_to_mat(ml_data.egeo, null_sites)
pc.pixel_corrections(arr_data.reshape(ml_data.egeo + (-1,)), (), zip(ni, nj))
 
sz_traces = ml_data.data[:,2*29:3*29]
tser = np.mean(sz_traces, axis=1)
tx = np.arange(arr_data.shape[-1])/ml_data.Fs
dscr = data_scroll.DataScroller(
    arr_data, tser, rowcol=ml_data.egeo, Fs=ml_data.Fs, ts_page_length=15
    )


## Focus on interval
t_start = 230; t_stop = 240
ix = np.where((tx>t_start) & (tx<t_stop))[0]

sz_frames = arr_data[:,ix].reshape(16, 16, -1)
sz_frames = sz_frames.transpose(2, 0, 1)

clim = (-6e-4, 6e-4)

## ani.write_frames(
##     sz_frames, fname='pri_sz_test30_%d-%d_sec'%(t_start, t_stop),
##     title='Primate Seizure Test 30', fps=20, clim=clim
##     )


## a = ani.animate_frames_and_series(
##     sz_frames, sz_traces[ix], tx=tx[ix],
##     fps=30, imshow_kw=dict(clim=clim),
##     line_props=dict(color='b'),
##     title='Primate Sz Test 30'
##     )
