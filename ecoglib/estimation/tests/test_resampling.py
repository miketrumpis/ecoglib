from scipy.special import comb
from numpy.testing import assert_almost_equal
import numpy as np

from ecogdata.parallel.tests import with_start_methods
from ecoglib.estimation.resampling import Jackknife, Bootstrap


def test_boot_sample_size():

    N = 10
    r = np.random.randn(N)

    assert np.array(Bootstrap(r, 4).all_samples()).shape == (4, 10)
    assert np.array(Bootstrap(r, 4, sample_size=8).all_samples()).shape == (4, 8)


def test_jn_sample_size():
    """Tests correct sample size"""

    N = 10
    d = 4
    r = np.random.randn(N)

    assert len(Jackknife(r).all_samples()) == N
    assert len(Jackknife(r, n_out=d).all_samples()) == comb(N, d)
    assert len(Jackknife(r, n_out=d, max_samps=10).all_samples()) == 10


def test_bs_nd():
    N1, N2, N3 = 10, 3, 8
    samps = 4
    r = np.random.randn(N1, N2, N3)

    s = Bootstrap(r, samps, axis=0).all_samples()
    assert len(s) == samps
    assert s[0].shape == (N1, N2, N3)

    s = Bootstrap(r, samps, axis=1).all_samples()
    assert len(s) == samps
    assert s[0].shape == (N1, N2, N3)

    s = Bootstrap(r, samps, axis=2).all_samples()
    assert len(s) == samps
    assert s[0].shape == (N1, N2, N3)


def test_jn_nd():
    """Tests correct sample size for ND"""

    N1, N2, N3 = 10, 3, 8
    r = np.random.randn(N1, N2, N3)

    s = Jackknife(r, axis=0).all_samples()
    assert len(s) == N1
    assert s[0].shape == (N1 - 1, N2, N3)

    s = Jackknife(r, axis=1).all_samples()
    assert len(s) == N2
    assert s[0].shape == (N1, N2 - 1, N3)

    s = Jackknife(r, axis=2).all_samples()
    assert len(s) == N3
    assert s[0].shape == (N1, N2, N3 - 1)


def test_jn_multiarray():
    """Tests correct sample size for multiple inputs"""

    # match 1st axis but let others be whatever
    N1, N2, N3 = np.random.randint(5, 10, size=3)
    M1 = N1
    M2, M3 = np.random.randint(5, 10, size=2)
    r1 = np.zeros((N1, N2, N3))
    r2 = np.zeros((M1, M2, M3))
    s = Jackknife([r1, r2], axis=0).all_samples()
    assert len(s) == N1
    assert len(s[0]) == 2
    assert s[0][0].shape == (N1 - 1, N2, N3)
    assert s[0][1].shape == (M1 - 1, M2, M3)

    N1, N2, N3 = np.random.randint(5, 10, size=3)
    M2 = N2
    M1, M3 = np.random.randint(5, 10, size=2)
    r1 = np.zeros((N1, N2, N3))
    r2 = np.zeros((M1, M2, M3))
    s = Jackknife([r1, r2], axis=1).all_samples()
    assert len(s) == N2
    assert len(s[0]) == 2
    assert s[0][0].shape == (N1, N2 - 1, N3)
    assert s[0][1].shape == (M1, M2 - 1, M3)

    N1, N2, N3 = np.random.randint(5, 10, size=3)
    M3 = N3
    M1, M2 = np.random.randint(5, 10, size=2)
    r1 = np.zeros((N1, N2, N3))
    r2 = np.zeros((M1, M2, M3))
    s = Jackknife([r1, r2], axis=2).all_samples()
    assert len(s) == N3
    assert len(s[0]) == 2
    assert s[0][0].shape == (N1, N2, N3 - 1)
    assert s[0][1].shape == (M1, M2, M3 - 1)


def test_jn_nd_estimator():
    """Tests correct estimate size for ND"""

    N1, N2, N3 = 10, 3, 8
    d = 4
    r = np.random.randn(N1, N2, N3)

    t, se = Jackknife(r, axis=0).estimate(np.mean)
    assert t.shape == (N2, N3)
    assert se.shape == (N2, N3)
    t, se = Jackknife(r, axis=0).estimate(np.mean, keepdims=True)
    assert t.shape == (1, N2, N3)
    assert se.shape == (1, N2, N3)

    t, se = Jackknife(r, axis=1).estimate(np.mean)
    assert t.shape == (N1, N3)
    assert se.shape == (N1, N3)
    t, se = Jackknife(r, axis=1).estimate(np.mean, keepdims=True)
    assert t.shape == (N1, 1, N3)
    assert se.shape == (N1, 1, N3)

    t, se = Jackknife(r, axis=2).estimate(np.mean)
    assert t.shape == (N1, N2)
    assert se.shape == (N1, N2)
    t, se = Jackknife(r, axis=2).estimate(np.mean, keepdims=True)
    assert t.shape == (N1, N2, 1)
    assert se.shape == (N1, N2, 1)


def test_jn_sample_consistency():
    """Test sample-resample equality"""

    N = 10
    d = 4
    r = np.random.randn(N)

    jn = Jackknife(r, ordered_samples=True)
    s1 = np.array(jn.all_samples())
    s2 = np.array(jn.all_samples())
    assert (s1 == s2).all()

    # check for N-choose-r jackknifes
    jn = Jackknife(r, n_out=d, ordered_samples=True)
    s1 = np.array(jn.all_samples())
    s2 = np.array(jn.all_samples())
    print(list(map(np.shape, (s1, s2))))
    assert (s1 == s2).all()


def test_jn_pseudoval_consistency():
    """Check PVs sum up to bias-corrected estimator"""

    N = 10
    d = 4
    r = np.random.randn(N)

    jn = Jackknife(r)
    est = np.std(r)

    # compute bias another way:
    s = jn.all_samples(estimator=np.std)
    bias = float(N - 1) * (np.mean(s, axis=0) - est)
    assert_almost_equal(bias, jn.bias(np.std))

    # compute bias another way:
    jn = Jackknife(r, n_out=d)
    s = jn.all_samples(estimator=np.std)
    bias = float(N - d) * (np.mean(s, axis=0) - est) / d
    assert_almost_equal(bias, jn.bias(np.std))


def test_jn_variance():
    """Jackknife variance should equal standard SEM calculation"""
    r = np.random.randn(10)

    sem = np.std(r) / np.sqrt(len(r) - 1)

    jn_se1 = Jackknife(r).estimate(np.mean, se=True, correct_bias=True)[1]
    assert_almost_equal(jn_se1, sem, err_msg='Jackknife SE with pseudovalues not almost equal')
    jn_se2 = Jackknife(r).estimate(np.mean, se=True, correct_bias=False)[1]
    assert_almost_equal(jn_se2, sem, err_msg='Standard Jackknife SE not almost equal')
    jn_se3 = Jackknife(r).variance(np.mean) ** 0.5
    assert_almost_equal(jn_se3, sem, err_msg='Standard(2) Jackknife SE not almost equal')


@with_start_methods
def test_bs_multi_jobs():
    # going to resample all ones 100 times and calculate the average (which will be one)
    a = np.ones(1000)
    bs = Bootstrap(a, 100, n_jobs=4)
    estimates = bs.all_samples(estimator=np.mean)
    assert all([e == 1 for e in estimates])

@with_start_methods
def test_jn_multi_jobs():
    # going to jack-knife resample all ones and calculate the average (which will be one)
    a = np.ones(25)
    jn = Jackknife(a, n_jobs=4)
    estimates = jn.all_samples(estimator=np.mean)
    assert all([e == 1 for e in estimates])