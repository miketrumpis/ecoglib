"""

.. _vis_specgrams:

================================================================
 Demo for complex demodulates and specgram in sparse noise data
================================================================

"""

import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.fftpack as ffts
import nitime.algorithms as nt_alg
import nitime.utils as nt_ut
import matplotlib.pyplot as pp
import matplotlib.mlab as mlab
import sandbox.mtm_spectrogram as mtm_spec
import sandbox.trigger_fun as tfn

import tables

f = tables.open_file('../../../mlab/fooaug6.mat')
data = f.root.s.data[:].T
trig_coding = f.root.s.trig_coding[:].T.astype('i')
Fs = f.root.s.Fs[0,0]

avg, navg = tfn.cond_trigger_avg(
    data, trig_coding, post=int(np.round(Fs*.2)), iqr_thresh=3
    )

"""
Quick and dirty RFs -- look for peak RMS power
"""

rms = np.sqrt(np.mean(avg**2, axis=-1))
rfs = tfn.fenced_out(rms, thresh=5, axis=1, low=False)
rfs = rfs[:,::3].reshape(-1, 16, 16) # just look at dark contrast
for rf in rfs:
    rf[:] = ndimage.morphology.binary_fill_holes(rf)
rfs = 1 - rfs

"""
Compute the full spectrogram for channel 16's recording
"""

## bsize = 512; pl = 0.75; NW = 4
## tx, fx, pmat = mtm_spec.mtm_spectrogram(
##     data[15], bsize, pl=pl, detrend='linear',
##     NW=NW, low_bias=True, adaptive=False
##     )
## pp.figure()
## pp.imshow(
##     np.log10(pmat), cmap='spectral', extent=(tx[0], tx[-1], 0, Fs/2),
##     interpolation='nearest', norm=pp.normalize(-12.5, -10.5)
##     )


"""
Find best stim for a given channel
"""

ch = 28 - 1
c_best = np.argmax(rms[ch])
c_labels = np.sort(np.where(trig_coding[1] == c_best + 1)[0])

pre = 0.5; post = 1.0
trials = tfn.extract_epochs(
    data, trig_coding, c_labels,
    pre=int(np.round(Fs*pre)), post=int(np.round(Fs*post))
    )
tx_full = np.arange(trials.shape[-1]) / Fs - pre

"""

Demo the components of the VEP power
====================================

"""
NW = 20
dpss, eigs = nt_alg.dpss_windows(trials.shape[-1], NW, 2*NW)
k = eigs > .99
dpss = dpss[k]
eigs = eigs[k]
nfft = 2 ** int( np.ceil(np.log2(len(tx_full))) )
x_tf, _, w = mtm_spec.mtm_complex_demodulate(
    trials[ch], NW=NW, nfft=nfft, samp_factor=0, adaptive=True,
    dpss=dpss, eigs=eigs
    )
fx2 = np.linspace(0, Fs/2, nfft/2 + 1)
b1c = 15; b2c = 30; b3c = 70; b4c = 200
bmin = np.argmin( np.abs(fx2 - 8) )
b1 = np.argmin( np.abs(fx2 - b1c) )
b2 = np.argmin( np.abs(fx2 - b2c) )
b3 = np.argmin( np.abs(fx2 - b3c) )
b4 = np.argmin( np.abs(fx2 - b4c) )

"""
Compute biased estimators of the VEP instantaneous power. The first is
simply the trial-by-trial average and the second is the trial
sum-of-squares of the complex demodulates.
"""

# (similarly) biased estimators of VEP power
x_pwr = np.sqrt( np.mean(trials[ch]**2, axis=0) )
band_pwr = np.sqrt(np.mean(np.abs(x_tf)**2, axis=0))
## x_pwr = np.mean(np.abs(trials[ch]), axis=0)
## band_pwr = np.mean(np.abs(x_tf), axis=0)
pp.figure()
pp.subplot(211)
pp.plot(tx_full, band_pwr[:b4].T, color=(.6, .6, .6))
pp.plot(tx_full, x_pwr, color='k', linewidth=3)

pp.subplot(212)
l1 = pp.plot(tx_full, band_pwr[bmin:b1].T, color='g', label='_nolegend_')
l2 = pp.plot(tx_full, band_pwr[b1:b2].T, color='y', label='_nolegend_')
l3 = pp.plot(tx_full, band_pwr[b2:b3].T, color='c', label='_nolegend_')
l4 = pp.plot(tx_full, band_pwr[b3:b4].T, color='m', label='_nolegend_')
pp.plot(tx_full, x_pwr, color='k', linewidth=3, label='_nolegend_')
pp.legend(
    (l1[0], l2[0], l3[0], l4[0]),
    ('(8, %d) Hz'%b1c, '(%d, %d) Hz'%(b1c, b2c),
     '(%d, %d) Hz'%(b2c, b3c), '(%d, %d) Hz'%(b3c, b4c)),
    loc='upper right'
    )
pp.gcf().tight_layout()

"""

.. image:: fig/vis_specgrams_01.png

We can see that the structure of the mean VEP is actually made up of
many different peaks in different bands, all with different
latencies. On the bottom is a more or less arbitrary partition of the
bands, showing some grouping of the latencies.

"""

def _z_norm(x, itvl):
    mn = x[:,itvl].mean()
    stdev = x[:,itvl].std()
    xz = (x - mn) / stdev
    return xz

"""
Define a quick and *dirty* Z normalization. Basically the individual
complex demodulates show quasi-regular bursts of power, which is to be
expected from the sparse noise sequence. Since we are aggregating
non-signed (positive) samples, these do not quite average out and
leave a wiggly baseline. What this normalization does is to first
average over frequencies (using the magnitude of the 1st Slepian as a
smoothing window), and then average over a pre-stim baseline. The
central 2nd moment is computed with a manipulation of the same method.
"""

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

baseline = np.where( (tx_full<0) & (tx_full>-0.3) )[0]
band_pwr_z = _smooth_z_norm(band_pwr, baseline)
## band_pwr_z = np.empty_like(band_pwr)
## band_pwr_z[bmin:b1] = _z_norm(band_pwr[bmin:b1], baseline)
## band_pwr_z[b1:b2] = _z_norm(band_pwr[b1:b2], baseline)
## band_pwr_z[b2:b3] = _z_norm(band_pwr[b2:b3], baseline)
## band_pwr_z[b3:b4] = _z_norm(band_pwr[b3:b4], baseline)
pp.figure()
l1 = pp.plot(tx_full, band_pwr_z[bmin:b1].T, color='g', label='_nolegend_')
l2 = pp.plot(tx_full, band_pwr_z[b1:b2].T, color='y', label='_nolegend_')
l3 = pp.plot(tx_full, band_pwr_z[b2:b3].T, color='c', label='_nolegend_')
l4 = pp.plot(tx_full, band_pwr_z[b3:b4].T, color='m', label='_nolegend_')
pp.legend(
    (l1[0], l2[0], l3[0], l4[0]),
    ('(8, %d) Hz'%b1c, '(%d, %d) Hz'%(b1c, b2c),
     '(%d, %d) Hz'%(b2c, b3c), '(%d, %d) Hz'%(b3c, b4c)),
    loc='upper right'
    )
pp.title('bandpass z scores')
pp.gcf().tight_layout()

"""

.. image:: fig/vis_specgrams_02.png

Here all the bands are on normalized footing, and the latencies stand
out better.

The same picture can be plotted in image form as a spectrogram-like
signal.

"""

band_pwr_norm = band_pwr - \
  np.mean(band_pwr[:, (tx_full<0)&(tx_full>-0.3)], axis=1)[:,None]
eps = 1e-10
band_pwr_norm[band_pwr_norm <= 0] = eps
pp.figure()
pp.imshow(
    np.log10(band_pwr_norm), extent=(tx_full[0], tx_full[-1], 0, Fs/2),
    clim=(-6, -4),
    interpolation='nearest'
    )
pp.colorbar()

"""

.. image:: fig/vis_specgrams_03.png

This is the baseline-subtracted pseudo-spectrogram.

"""

pp.figure()
pp.imshow(
    band_pwr_z, extent=(tx_full[0], tx_full[-1], 0, Fs/2),
    interpolation='nearest'
    )
pp.colorbar()
pp.show()

"""

.. image:: fig/vis_specgrams_04.png

This is the z-score pseudo-spetrogram. Notice the sweeps in time in
the various wave-band activations.

"""
