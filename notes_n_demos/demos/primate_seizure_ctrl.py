## skip example
import numpy as np
import matplotlib.pyplot as pp
import tables

import ecoglib.util as ut
import ecoglib.data.preproc_data as ppd
import ecoglib.vis.data_scroll as data_scroll
import ecoglib.vis.ani as ani
import ecoglib.data.arangement as da
import ecoglib.filt.space.pix_corr as pc

## An experiment to explore the visual impact of sampling delays
## between channels
##
## 1) Load references to the data
## 2) fake a dataset with some delay between the sampled channels

ml_data = ppd.load_preproc(
    '/Users/mike/work/ecog_sz/mlab/joined_data_33.h5', load=False
    )

# minus 1 to correct for MATLAB
emap = ml_data.emap.read().squeeze().astype('i') - 1
egeo = tuple(ml_data.egeo.read().squeeze().astype('i'))
emap = ut.flat_to_flat(egeo, emap)
Fs = ml_data.Fs.read()[0,0]

f = pp.figure()
for n in range(4):
    q_sites = emap[n*29:(n+1)*29]
    chart = np.zeros((16, 16), 'i')
    chart.flat[q_sites] = np.arange(1,29+1)
    f.add_subplot(2,4,n+1)
    pp.imshow(chart); pp.axis('image')
    pp.title('NI Conn %d'%(n+1,))
br_map = emap[116:]
for n in range(3):
    q_sites = br_map[n*32:(n+1)*32]
    chart = np.zeros((16,16), 'i')
    chart.flat[q_sites] = np.arange(1,32+1)
    f.add_subplot(2,4,n+5)
    pp.imshow(chart); pp.axis('image')
    pp.title('BR conn %d'%(n+1,))

fk_data = np.zeros( (len(emap), 10000), 'd' )
phs_by_chan = 2*np.pi*(np.arange(10000)[None,:] + np.linspace(0,1,29)[:,None])
fk_data[2*29:3*29, :] = np.sin( phs_by_chan )


(arr_data, null_sites) = da.array_geometry(
    fk_data, egeo, emap, axis=0, col_major=False
    )

(ni, nj) = ut.flat_to_mat(egeo, null_sites, col_major=False)
pc.pixel_corrections(arr_data.reshape(egeo + (-1,)), (), zip(ni, nj))
 
sz_traces = fk_data[2*29:3*29,:]
tser = np.mean(sz_traces, axis=0)
tx = np.arange(arr_data.shape[-1])/Fs
dscr = data_scroll.DataScroller(
    arr_data, tser, rowcol=egeo, Fs=Fs, ts_page_length=15
    )

0/0

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
