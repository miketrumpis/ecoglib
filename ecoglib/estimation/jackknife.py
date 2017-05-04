"""Iterators and tools for jackknife estimators"""

import random
import numpy as np
from scipy.special import comb
from sklearn.model_selection import KFold
from itertools import combinations, tee
from contextlib import closing
import ecoglib.mproc as mp
from sandbox.array_split import SharedmemManager

__all__ = ['random_combinations', 'Jackknife']

def random_combinations(iterable, r, n):
    "Pull n random selections from itertools.combinations(iterable, r)"
    if n > comb(len(iterable), r):
        raise ValueError('Not enough combinations for {0} samples'.format(n))
    pulls = set()
    pool = tuple(iterable)
    L = len(pool)
    while len(pulls) < n:
        indices = sorted(random.sample(xrange(L), r))
        pulls.add( tuple(pool[i] for i in indices) )
    return pulls

def _jackknife_sampler(index):
    """Light array sampler using shared memory.

    The multiprocess Pool is initialized so that these variables are in
    global memory:

    * shm_array
    * axis
    * estimator
    * e_args
    * e_kwargs
    
    """
        
    array = shm_array.get_ndarray()
    samps = np.take(array, index, axis=axis)
    if estimator is not None:
        e_kwargs['axis'] = axis
        return estimator(samps, *e_args, **e_kwargs)
    return samps

    
class Jackknife(object):
    """
    Jackknife generator with leave-L-out resampling.

    Parameters
    ----------
    array : ndarray
        Samples to jack-knife
    n_out : int
        Number of samples to leave out of each jack-knife
    axis : int
        Which axis to resample
    max_samps : int
        If N-choose-L is large, limit the number of jack-knife samples

    """

    def __init__(
            self, array, n_out=1, axis=-1, max_samps=-1,
            n_jobs=None, ordered_samples=False
            ):

        self._array = array
        self._axis = axis
        N = array.shape[axis]
        self.__choose = (xrange(N), N-n_out)
        self._resampler = None
        self._max_samps = max_samps
        self._n_jobs = n_jobs
        self._ordered_samples = ordered_samples

    def _init_sampler(self):
        if self._max_samps < 0:
            self._resampler = combinations(*self.__choose)
        elif self._resampler is None:
            iterable, r = self.__choose
            self._resampler = random_combinations(iterable, r, self._max_samps)
        # else random sampler is already set

    def __len__(self):
        if self._max_samps > 0:
            return self._max_samps
        r = self.__choose[1]
        return comb(self._array.shape[self._axis], r)
        
    def sample(
            self, estimator=None, e_args=(), **e_kwargs
            ):
        """
        Make min(N-choose-L, max_samps) jack-knife samples.
        Optionally return jack-knife samples of an estimator. The
        estimator must be a callable that accepts the "axis" keyword.
        """

        self._init_sampler()
        def _init_pool(pool_args):
            for kw in pool_args.keys():
                globals()[kw] = pool_args[kw]
        pool_args = dict(shm_array=SharedmemManager(self._array, use_lock=True),
                         axis=self._axis, estimator=estimator,
                         e_args=e_args, e_kwargs=e_kwargs)
        with closing(mp.Pool(processes=self._n_jobs,
                             initializer=_init_pool,
                             initargs=(pool_args,))) as p:
            if self._ordered_samples:
                for samp in p.imap(_jackknife_sampler, self._resampler):
                    yield samp
            else:
                for samp in p.imap_unordered(
                        _jackknife_sampler, self._resampler
                        ):
                    yield samp

    def all_samples(self, estimator=None, e_args=(), **e_kwargs):
        """Return all samples from the generator"""
        samps = list(self.sample(estimator=estimator, e_args=e_args, **e_kwargs))
        return np.array(samps)

    def pseudovals(self, estimator, jn_samples=(), e_args=(), **e_kwargs):
        """Return the bias-correcting pseudovalues of the estimator"""
        if not len(jn_samples):
            jn_samples = self.all_samples(
                estimator=estimator, e_args=e_args, **e_kwargs
                )
        e_kwargs['axis'] = self._axis
        theta = estimator(self._array, *e_args, **e_kwargs)
        N1 = float(self._array.shape[self._axis])
        d = N1 - self.__choose[1]
        return N1 / d * theta - (N1 - d) * jn_samples / d
        
    def estimate(
            self, estimator, correct_bias=True, se=False,
            e_args=(), **e_kwargs
            ):
        if correct_bias:
            vals = self.pseudovals(estimator, e_args=e_args, **e_kwargs)
        else:
            vals = self.all_samples(
                estimator=estimator, e_args=e_args, **e_kwargs
                )
        if se:
            se = np.std(vals, axis=0) / np.sqrt( len(vals) )
            return np.mean(vals, axis=0), se
        return np.mean(vals, axis=0)

    def bias(self, estimator, jn_samples=(), e_args=(), **e_kwargs):
        """
        Compute the jack-knife bias of an estimator.
        """

        e_kwargs['axis'] = self._axis
        theta = estimator(self._array, *e_args, **e_kwargs)
        pv = self.pseudovals(
            estimator, jn_samples=jn_samples, e_args=e_args, **e_kwargs
            )
        # mean of PV = theta - bias
        # bias = theta - avg{PV}
        return theta - np.mean(pv, axis=0)

    def variance(self, estimator, jn_samples=(), e_args=(), **e_kwargs):
        """
        Compute the jack-knife bias of an estimator.

        NOTE! The normalization is probably wrong for delete-d JN
        """

        pv = self.pseudovals(
            estimator, jn_samples=jn_samples, e_args=e_args, **e_kwargs
            )
        N1 = float(self._array.shape[self._axis])
        return np.var(pv, axis=0) / N1
        
