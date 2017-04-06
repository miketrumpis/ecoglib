import numpy as np
import matplotlib.pyplot as pp
import scipy.stats.distributions as dists
import ecoglib.estimation.multitaper as mtm_spec


Fs = 2000.
N = 100000
blip_sz = int( 0.05 * Fs )
f0 = 50

test_sig = np.random.randn(N) * .1
blip = np.hanning(blip_sz) * np.sin( 2 * np.pi * 50 * np.arange(blip_sz) / Fs )
blip_locs = np.cumsum(dists.expon.rvs(10000, size=5)).astype('i')
for loc in blip_locs:
    test_sig[loc:loc+len(blip)] += blip

f, axs = pp.subplots(2, 1, sharex=True)
ax = axs[0]
ax.plot(test_sig)
for loc in blip_locs:
    ax.axvline(loc, color='r', ls='--')

block_size = 256
lag = 32
pl = float(block_size - lag) / block_size
NW = 3
tx, fx, pmat = mtm_spec.mtm_spectrogram(
    test_sig, block_size, pl=pl, detrend='linear', NW=NW, adaptive=False
    )

ax = axs[1]
ax.imshow(pmat, extent=[0, N-1, 0, Fs/2.0], cmap='hot')
ax.axis('auto')
f.tight_layout()

import sandbox.trigger_fun as tfun
epochs = tfun.extract_epochs(
    test_sig[None, :], np.c_[blip_locs, np.ones_like(blip_locs)].T,
    pre=0.05*Fs, post=0.1*Fs
    ).squeeze()

pmats = list()
for trial in epochs:
    tx, fx, pmat = mtm_spec.mtm_spectrogram(
        trial, 64, pl=0.5, Fs=Fs, NW=NW, NFFT=128, adaptive=False
        )
    pmats.append(pmat)
    

pp.show()

