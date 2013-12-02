## skip example
"""

.. _vis_specgrams:

================================================================
 Demo for complex demodulates and specgram in sparse noise data
================================================================

"""

import sys
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.fftpack as ffts
import nitime.algorithms as nt_alg
import nitime.utils as nt_ut
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pp
import matplotlib.mlab as mlab
import sandbox.mtm_spectrogram as mtm_spec
import sandbox.trigger_fun as tfn
import sandbox.tile_images as tile_images
import multiprocessing as mp
import gc

try:
    import scipy.io as sio
    m = sio.loadmat('../../../mlab/test35_proc_unr_781Hz_3-300bp.mat')
    data = m['s']['data'][0,0]
    trig_coding = m['s']['trig_coding'][0,0].astype('i')
    Fs = float( m['s']['Fs'][0,0] )
    del m
except NotImplementedError:
    import tables
    f = tables.open_file('../../../mlab/test35_proc_unr_781Hz_3-300bp.mat')
    data = f.root.s.data[:].T
    trig_coding = f.root.s.trig_coding[:].T.astype('i')
    Fs = f.root.s.Fs[0,0]


n_locs = len(np.unique(trig_coding[1])) / 3
field_size = int( np.sqrt(n_locs) )

NW = 15
upsamp = 4
pre = int( round(Fs*0.5) ); post = int( round(Fs*1.0) )
n_pts = pre + post
nfft = 2 ** int( np.ceil( np.log2( n_pts ) ) )
tx_full = np.linspace(-0.5, 1.0, n_pts)
last_freq = int( (nfft/2+1) * 2 * 300 / Fs )

dpss, eigs = nt_alg.dpss_windows(n_pts, NW, 2*NW)
k = eigs > .99
dpss = dpss[k]; eigs = eigs[k]
spec_map = np.zeros( (field_size, field_size, 2*NW*upsamp, nfft/2 + 1), 'd' )
print 'array created: ', spec_map.shape, spec_map.size
def _smooth_z_norm(x, itvl):
    filt = np.abs(ffts.fft(dpss[0], n=nfft))
    bw = int( float(NW) * nfft / dpss.shape[1] + 0.5 )
    # normalize spectral smoothing so that the sum within the 2NW BW is unity
    filt /= (2*np.sum(filt[:bw]))
    # circular extension is logical for spectral smoothing
    xc = np.zeros( (nfft, x.shape[1]), 'd' )
    xc[:x.shape[0]] = x
    xc[x.shape[0]:] = x[1:-1][::-1]

    xf = ffts.fft(xc[:,itvl], axis=0)
    xf *= ffts.fft(filt)[:,None]
    xf = ffts.ifft(xf, axis=0).real
    # filter "averaged" in freq, now average in time
    xf = xf.mean(axis=1)
    xcent = xc - xf[:,None]
    # now do centered second moment
    xcent_sq = ffts.fft(xcent[:,itvl] ** 2, axis=0)
    xcent_sq *= ffts.fft(filt)[:,None]
    xcent_sq = ffts.ifft(xcent_sq, axis=0).real
    xcent_sq = np.sqrt(xcent_sq.mean(axis=1))
    xcent /= xcent_sq[:,None]
    return xcent[:nfft/2+1]

def _plot_proc(maps, ch, extent, clim, cmap):
    f = tile_images.tile_images(
        maps, extent=extent, cmap=cmap, clim=clim
        )
    t = f.text(
        .5, 1-.075, 'Chan %d Dark Contrast Z-Spectrogram'%(ch+1,),
        fontsize=20, ha='center'
        )
    f.savefig('ch_%d_dark_zspec.pdf'%(ch+1,))
    f.clf()
    pp.close(f)
    gc.collect()


for ch in xrange(9,data.shape[0]):
    for cond in xrange(field_size*field_size):

        cx = cond / field_size
        cy = cond % field_size

        d_cond = cond * 3 + 1; b_cond = cond * 3 + 3

        trig_labels_d = np.where(trig_coding[1] == d_cond)[0]
        trig_labels_b = np.where(trig_coding[1] == b_cond)[0]
        trig_labels = np.r_[trig_labels_d, trig_labels_b]

        trials = tfn.extract_epochs(
            data[ch], trig_coding, trig_labels,
            pre=pre, post=post
            )

        x_tf, ix, w = mtm_spec.mtm_complex_demodulate(
            trials.squeeze(), NW=NW, nfft=nfft, samp_factor=upsamp,
            adaptive=False, dpss=dpss, eigs=eigs
            )
        if cond < 1:
            tx_sub = np.interp(ix, np.arange(n_pts), tx_full)
            baseline = np.where( (tx_sub > -0.3) & (tx_sub < 0) )[0]
            plot_itvl = np.where( (tx_sub > -0.1) & (tx_sub < 0.5) )[0]
        pwr = np.sqrt( np.mean(np.abs(x_tf)**2, axis=0) )
        spec_map[cx,cy,:,:] = _smooth_z_norm(pwr, baseline).T
        print 'cond ', cond,
        sys.stdout.flush()
    # done cond
    p = mp.Process(
        target=_plot_proc,
        args=( spec_map[:,:,plot_itvl,:last_freq], ch,
               (-0.1, 0.5, 0, 300), (3, 20), 'spectral' )
              )
    p.start()
    ## f = tile_images.tile_images(
    ##     spec_map[:,:,plot_itvl,:last_freq],
    ##     extent=(-0.1, 0.5, 0, 300),
    ##     cmap='spectral', clim=(3, 20)
    ##     )
    ## t = f.text(
    ##     .5, 1-.075, 'Chan %d Dark Contrast Z-Spectrogram'%ch,
    ##     fontsize=20, ha='center'
    ##     )
    ## f.savefig('ch_%d_dark_zspec.pdf'%(ch+1,))
    ## f.clf()
    ## pp.close(f)
    ## gc.collect()

# done ch
