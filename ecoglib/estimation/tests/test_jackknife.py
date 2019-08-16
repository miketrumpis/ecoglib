from nose.tools import assert_true, assert_equal

from scipy.special import comb
from numpy.testing import assert_almost_equal
import numpy as np

from ecoglib.estimation.jackknife import Jackknife

def test_sample_size():
    "Tests correct sample size"

    N = 10
    d = 4
    r = np.random.randn(N)
    
    assert_true( len(Jackknife(r).all_samples()) == N )
    assert_true( len(Jackknife(r, n_out=d).all_samples()) == comb(N, d) )
    assert_true( len(Jackknife(r, n_out=d, max_samps=10).all_samples()) == 10 )

def test_nd():
    "Tests correct sample size for ND"

    
    N1, N2, N3 = 10, 3, 8
    d = 4
    r = np.random.randn(N1, N2, N3)

    s = Jackknife(r, axis=0).all_samples()
    assert_true( len(s) == N1 )
    assert_true( s[0].shape == (N1-1, N2, N3) )
    
    s = Jackknife(r, axis=1).all_samples()
    assert_true( len(s) == N2 )
    assert_true( s[0].shape == (N1, N2-1, N3) )

    s = Jackknife(r, axis=2).all_samples()
    assert_true( len(s) == N3 )
    assert_true( s[0].shape == (N1, N2, N3-1) )

def test_nd_estimator():
    "Tests correct estimate size for ND"

    
    N1, N2, N3 = 10, 3, 8
    d = 4
    r = np.random.randn(N1, N2, N3)

    t = Jackknife(r, axis=0).estimate(np.mean)
    assert_true( t.shape == (N2, N3) )
    t = Jackknife(r, axis=0).estimate(np.mean, keepdims=True)
    assert_true( t.shape == (1, N2, N3) )
    
    t = Jackknife(r, axis=1).estimate(np.mean)
    assert_true( t.shape == (N1, N3) )
    t = Jackknife(r, axis=1).estimate(np.mean, keepdims=True)
    assert_true( t.shape == (N1, 1, N3) )

    t = Jackknife(r, axis=2).estimate(np.mean)
    assert_true( t.shape == (N1, N2) )
    t = Jackknife(r, axis=2).estimate(np.mean, keepdims=True)
    assert_true( t.shape == (N1, N2, 1) )

def test_sample_consistency():
    "Test sample-resample equality"

    N = 10
    d = 4
    r = np.random.randn(N)

    jn = Jackknife(r, ordered_samples=True)
    s1 = np.array( jn.all_samples() )
    s2 = np.array( jn.all_samples() )
    assert_true( (s1 == s2).all() )

    # check for random subset of N-choose-r jackknifes
    jn = Jackknife(r, n_out=d, max_samps=5, ordered_samples=True)
    s1 = np.array( jn.all_samples() )
    s2 = np.array( jn.all_samples() )
    print(list(map(np.shape, (s1, s2))))
    assert_true( (s1 == s2).all() )
    

        
def test_pseudoval_consistency():
    "Check PVs sum up to bias-corrected estimator"


    N = 10
    d = 4
    r = np.random.randn(N)

    jn = Jackknife(r)
    est = np.std(r)
    
    # compute bias another way:
    s = jn.all_samples(estimator=np.std)
    bias = float(N-1) * (np.mean(s, axis=0) - est)
    assert_almost_equal(bias, jn.bias(np.std))
    
    # compute bias another way:
    jn = Jackknife(r, n_out=d)
    s = jn.all_samples(estimator=np.std)
    bias = float(N-d) * (np.mean(s, axis=0) - est) / d
    assert_almost_equal(bias, jn.bias(np.std))
        
