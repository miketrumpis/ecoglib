"""

.. _mtm_spectrogram_freqmodulation:

======================================
High resolution multitaper spectrogram
======================================

The complex demodulates illustrated in
:doc:`mtm_baseband_power` can also be combined to form a high
resolution spectrogram. Since there is a complex time series
corresponding to the equivalent lowpass signal centered at each
analyzed frequency, then it is relatively straightfoward to manipulate
this time-frequency matrix into a spectrogram. Operating on shifting
blocks is also possible, and the spectrogram can be averaged at
overlapping time-frequency bins.

The frequency and time resolution are reciprocal, as
expected. Frequency resolution is of course 2W, and time resolution is
1/2W. 

This demo illustrates the time-frequency image of the frequency
modulation of a sinusoid.

"""

import numpy as np
import scipy.signal as signal
import nitime.algorithms as nt_alg
import nitime.utils as nt_ut
import matplotlib.pyplot as pp
import matplotlib.mlab as mlab
import ecoglib.estimation.multitaper as mtm_spec

"""
We're going to set up a FM of a sinusoid varying slowly between +/-
0.1. This signal will hover around a center frequency at 0.15.
"""

N = 60000
awgn_sig = 0
tx = np.arange(N, dtype='d')
fm_freq = 0.1*np.cos(2*np.pi*tx * .0002)
fm_sig = np.cos(2*np.pi*(0.15*tx + np.cumsum(fm_freq)))
fm_sig += np.random.randn(N) * awgn_sig

"""
Now we set up an overlapping spectrogram with 75% overlap. The
effective temporal resolution is determined by the oscillatory width of the
Slepian sequences, and is reciprocal with the bandwidth of the Slepian
spectral functions. Therefore the time dimension is downsampled by
default. 
"""

bsize = 512
lag = 128
pl = float(bsize-lag)/bsize
NW = 6

tx, fx, pmat = mtm_spec.mtm_spectrogram(
    fm_sig, bsize, pl=pl, detrend='linear',
    NW=NW, low_bias=True, adaptive=True
    )

# power / Hz should be spread across 2NW spectral width
mx_power = 0.5 / fx[1] / (2*NW)

f = pp.figure(figsize=(10,5))
pp.subplot(211)
pp.plot(fm_freq + 0.15)
pp.ylim(0, 0.5)
ax = pp.subplot(212)
im = pp.imshow(
    pmat, interpolation='nearest', 
    extent=[0, N-1, 0, 0.5], cmap='hot', clim=(0, mx_power)
    )
pp.axis('auto')
f.tight_layout()
f.subplots_adjust(bottom=.25)
pos = ax.get_position()
cax = f.add_axes( [pos.x0, 0.125, pos.width, 0.05] )
cbar = pp.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label(r'spectral power ($Hz^{-1}$) clipped to theoretical max')

"""

.. image:: fig/demo_specgram_01.png

We can see the image of the modulated sinusoid varying about the
carrier frequency. This is fairly computationally intense, but we have
very good time and frequency resolution, adequate for the dynamics of
this signal. As always, there is a complementary tradeoff between time
and frequency resolution. The resulting bandwidth of the carrier
signal estimation is ~ 2NW, but the time resolutwion scales by the
reciprocal of 2W. The result is a consistent power estimation for
both region of fast and slow dynamics.

By comparison, we'll try the traditional overlapping windows method of
specral estimation. From this method, we'll have only a single spectral
estimate at each step of the shifting window.
"""

f = pp.figure(figsize=(10,5))
pp.subplot(211)
pp.plot(fm_freq + 0.15)
pp.ylim(0, 0.5)
ax = pp.subplot(212)
welsh_specgram, wf, wb = mlab.specgram(
    fm_sig, bsize, detrend=mlab.detrend_linear, noverlap=bsize-lag, Fs=1.0,
    scale_by_freq=True
    )
im = pp.imshow(
    welsh_specgram, interpolation='nearest',
    extent=(wb[0], wb[-1], 0, 0.5), cmap='hot',
    clim=(0, bsize/8.)
    )
pp.axis('auto')
f.tight_layout()
f.subplots_adjust(bottom=.25)
pos = ax.get_position()
cax = f.add_axes( [pos.x0, 0.125, pos.width, 0.05] )
cbar = pp.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label(r'spectral power ($Hz^{-1}$) clipped to %d'%(bsize/8.,))


"""

.. image:: fig/demo_specgram_02.png

Here we have a much coarser grain temporal resolution. By comparison,
the spectral concentration of signal power is better in the
slow-moving regions (near the flats of the sinusoid carrier). However,
the windowed periodogram estimate of power density suffers between the
peaks of the sinusoid, where the derivative in frequency with respect
to time is highest. This could probably be overcome with even shorter
block shifts. 

"""

pp.show()

