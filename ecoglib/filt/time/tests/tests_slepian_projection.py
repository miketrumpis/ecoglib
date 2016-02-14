from nose.tools import assert_true, assert_equal
 
from numpy.testing import assert_almost_equal
import numpy as np

from ecoglib.filt.time.slepian_projection import slepian_projection

def gen_sig(am=False, w0=80, bw=30, nfreqs=10):
    freqs = (np.random.rand(10) - 0.5) * bw
    if w0 and not am:
        freqs += w0
    amps = np.random.randn(10)
    phs = np.random.rand(10) * 2 * np.pi
    tx = np.arange(2000) * 2 * np.pi / 1000.
    narrowband = amps[:, None] * np.cos(freqs[:,None]*tx + phs[:,None])
    narrowband = narrowband.sum(0)
    if am:
        modulated = narrowband * np.cos(tx * w0)
        return modulated, narrowband
    return narrowband

def test_baseband_recon():
    modulated, baseband = gen_sig(am=True)
    recon = slepian_projection(modulated, 50, 1000.0, w0=80, baseband=True)
    # I guess < 0.1% error is good -- sometimes boundary effects, so
    # check within interior of signal
    err = recon[150:-150] - baseband[150:-150]
    rel_error = np.sum(err**2) / np.sum(baseband[150:-150]**2)
    assert_true( rel_error < 1e-3 )

def test_narrowband_recon():
    nb = gen_sig()
    nb_est = slepian_projection(nb, 50, 1000.0, w0=80)
    # I guess < 0.1% error is good -- sometimes boundary effects, so
    # check within interior of signal
    err = nb_est[150:-150] - nb[150:-150]
    rel_error = np.sum(err**2) / np.sum(nb[150:-150]**2)
    assert_true( rel_error < 1e-3 )

def test_bandpass_power():
    sg = np.random.randn(2000)
    # This bandpass window is from 150 to 250 (and -150 to -250),
    # Given the hypothetical bandwidth of 500 Hz, it should
    # take up ~ 20% of the signal power (which is pretty much unit-valued)
    sg_bp = slepian_projection(sg, 50, 1000, w0=200)
    assert_almost_equal(sg_bp.var(), 0.2, decimal=1)
    # This bandpass window is from -50 to 50, 
    # Given the hypothetical bandwidth of 500 Hz, it should
    # take up ~ 10% of the signal power (which is pretty much unit-valued)
    sg_bp = slepian_projection(sg, 50, 1000, w0=0)
    assert_almost_equal(sg_bp.var(), 0.1, decimal=1)

def test_shapes():
    sg_1d = np.random.randn(200)
    sg_2d = np.random.randn(3, 200)
    sg_3d = np.random.randn(2, 2, 200)

    assert_equal(slepian_projection(sg_1d, 20, 1000.0).shape, sg_1d.shape)
    assert_equal(slepian_projection(sg_2d, 20, 1000.0).shape, sg_2d.shape)
    assert_equal(slepian_projection(sg_3d, 20, 1000.0).shape, sg_3d.shape)
