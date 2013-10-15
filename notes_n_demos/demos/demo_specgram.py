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
import sandbox.mtm_spectrogram as mtm_spec

"""
We're going to set up a FM of a sinusoid varying slowly between +/-
0.1. This signal will hover around a carrier frequency at 0.15.
"""

N = 60000
awgn_sig = 1e-2
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
NW = 4

tx, fx, pmat = mtm_spec.mtm_spectrogram(
    fm_sig, bsize, pl=pl, detrend='linear',
    NW=NW, low_bias=True, adaptive=True
    )

pp.figure(figsize=(10,4))
pp.subplot(211)
pp.plot(fm_freq + 0.15)
pp.ylim(0, 0.5)
pp.subplot(212)
pp.imshow(pmat, interpolation='nearest', extent=[0, N-1, 0, 0.5], cmap='hot')
pp.gcf().tight_layout()

"""

.. image:: fig/demo_specgram_01.png

We can see the image of the modulated sinusoid varying about the
carrier frequency. This is fairly computationally intense, but we have
very good time and frequency resolution, adequate for the dynamics of
this signal. 

By comparison, we'll try the traditional overlapping windows method of
specral estimation. From this method, we'll have only a single spectral
estimate at each step of the shifting window.
"""

pp.figure(figsize=(10,4))
pp.subplot(211)
pp.plot(fm_freq + 0.15)
pp.ylim(0, 0.5)
pp.subplot(212)
welsh_specgram, wf, wb = mlab.specgram(
    fm_sig, bsize, detrend=mlab.detrend_linear, noverlap=bsize-lag, Fs=1.0
    )
pp.imshow(
    welsh_specgram, interpolation='nearest',
    extent=(wb[0], wb[-1], 0, 0.5), cmap='hot'
    )
pp.gcf().tight_layout()

"""

.. image:: fig/demo_specgram_02.png

Here we have a much coarser grain temporal resolution. The estimated
power density suffers between the peaks of the sinusoid, where the
derivative in frequency with respect to time is highest. This could
probably be overcome with even shorter block shifts.

"""

pp.show()

