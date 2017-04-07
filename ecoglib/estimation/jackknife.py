"""Iterators and tools for jackknife estimators"""

import random
import numpy as np
from scipy.special import comb
from sklearn.model_selection import KFold
from itertools import combinations, tee

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

    def __init__(self, array, n_out=1, axis=-1, max_samps=-1):

        self._array = array
        self._axis = axis
        N = array.shape[axis]
        self.__choose = (xrange(N), N-n_out)
        self._resampler = None
        self._max_samps = max_samps

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
        
    def sample(self, estimator=None, *e_args, **e_kwargs):
        """
        Make min(N-choose-L, max_samps) jack-knife samples.
        Optionally return jack-knife samples of an estimator. The
        estimator must be a callable that accepts the "axis" keyword.
        """

        self._init_sampler()
        for comb in self._resampler:
            samps = np.take(self._array, comb, axis=self._axis)
            if estimator is not None:
                e_kwargs['axis'] = self._axis
                yield estimator(samps, *e_args, **e_kwargs)
            else:
                yield samps

    def all_samples(self, estimator=None, *e_args, **e_kwargs):
        """Return all samples from the generator"""
        samps = list(self.sample(estimator=estimator, *e_args, **e_kwargs))
        return np.array(samps)

    def pseudovals(self, estimator, jn_samples=(), *e_args, **e_kwargs):
        """Return the bias-correcting pseudovalues of the estimator"""
        if not len(jn_samples):
            jn_samples = self.all_samples(
                estimator=estimator, *e_args, **e_kwargs
                )
        e_kwargs['axis'] = self._axis
        theta = estimator(self._array, *e_args, **e_kwargs)
        N1 = float(self._array.shape[self._axis])
        d = N1 - self.__choose[1]
        return N1 / d * theta - (N1 - d) * jn_samples / d
        
    def estimate(self, estimator, correct_bias=True, *e_args, **e_kwargs):
        if correct_bias:
            pv = self.pseudovals(estimator, *e_args, **e_kwargs)
            return np.mean(pv, axis=0)

        jn = self.all_samples(estimator=estimator, *e_args, **e_kwargs)
        return np.mean(jn, axis=0)        

    def bias(self, estimator, jn_samples=(), *e_args, **e_kwargs):
        """
        Compute the jack-knife bias of an estimator.
        """

        e_kwargs['axis'] = self._axis
        theta = estimator(self._array, *e_args, **e_kwargs)
        pv = self.pseudovals(
            estimator, jn_samples=jn_samples, *e_args, **e_kwargs
            )
        # mean of PV = theta - bias
        # bias = theta - avg{PV}
        return theta - np.mean(pv, axis=0)

    def variance(self, estimator, jn_samples=(), *e_args, **e_kwargs):
        """
        Compute the jack-knife bias of an estimator.

        NOTE! The normalization is probably wrong for delete-d JN
        """

        pv = self.pseudo_vals(
            estimator, jn_samples=jn_samples, *e_args, **e_kwargs
            )
        N1 = float(self._array.shape[self._axis])
        return np.var(pv, axis=0) / N1
        
