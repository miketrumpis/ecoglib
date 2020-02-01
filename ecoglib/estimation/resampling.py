"""Iterators and tools for jackknife estimators"""
import sys
import random
import numpy as np
from scipy.special import comb
from itertools import combinations
from contextlib import closing, ExitStack
import ecogdata.parallel.mproc as mp
from ecogdata.util import get_default_args
from ecogdata.parallel.array_split import SharedmemManager


__all__ = ['random_combinations', 'Jackknife', 'Bootstrap']


def random_combinations(iterable, r, n):
    """Pull n random selections from itertools.combinations(iterable, r)"""
    if n > comb(len(iterable), r):
        raise ValueError('Not enough combinations for {0} samples'.format(n))
    pulls = set()
    pool = tuple(iterable)
    L = len(pool)
    while len(pulls) < n:
        indices = sorted(random.sample(range(L), r))
        pulls.add(tuple(pool[i] for i in indices))
    return pulls


def _init_pool(pool_args):
    """stick a dictionary of variable names / values into the global namespace of the process worker"""
    for kw in pool_args.keys():
        globals()[kw] = pool_args[kw]


def _resampler(index):
    """Light array sampler using shared memory.

    The multiprocess Pool is initialized so that these variables are in
    global memory:

    * shm_arrays: list of arrays to be resampled on the given axis
    * axis
    * estimator
    * e_args
    * e_kwargs

    """

    with ExitStack() as stack:
        arrays = [stack.enter_context(shm.get_ndarray()) for shm in shm_arrays]
    # with shm_array.get_ndarray() as array:
        samps = [np.take(array, index, axis=axis) for array in arrays]
    if estimator is not None:
        # if there is *not*  an axis, then hope for the best
        if 'axis' in get_default_args(estimator):
            e_kwargs['axis'] = axis
        return estimator(*samps, *e_args, **e_kwargs)
    if len(samps) == 1:
        samps = samps[0]
    return samps


class Bootstrap:
    """
    Bootstrap resampler
    """

    def __init__(self, arrays, num_samples, axis=-1, sample_size=None, n_jobs=None, ordered_samples=False):
        """
        Make a bootstrap resampler for an array.

        Parameters
        ----------
        array: Sequence or ndarray
            Multidimensional sample data (or list of multiple arrays to be resampled)
        num_samples: int
            Number of bootstrap samples to create
        axis: int
            Axis of the array to resample
        sample_size: int
            If given, then pull so many samples with replacement from the array. Normally equal to the original
            sample size.
        n_jobs: int
            Number of parallel jobs to make bootstrap samples. None uses cpu_count().
        ordered_samples: bool
            ???

        """

        if not isinstance(arrays, (tuple, list)):
            self._arrays = [arrays]
        else:
            self._arrays = arrays
        self._axis = axis
        self._num_samples = num_samples
        if sample_size is None:
            self._sample_size = self._arrays[0].shape[axis]
        else:
            self._sample_size = sample_size
        self._resampler = None
        if sys.platform == 'win32':
            self._n_jobs = 1
        else:
            self._n_jobs = n_jobs
        self._ordered_samples = ordered_samples

    def _init_sampler(self):
        max_n = self._arrays[0].shape[self._axis]
        samp_size = self._sample_size

        def anon_sampler():
            for i in range(self._num_samples):
                yield np.random.randint(0, max_n, samp_size)
        self._resampler = anon_sampler()

    def __len__(self):
        return self._num_samples

    def sample(self, estimator=None, e_args=(), **e_kwargs):
        """Generate bootstrap samples, or estimates from the samples. The estimator must be a
        callable that accepts the "axis" keyword.
        """

        self._init_sampler()
        parallel = self._n_jobs is None or self._n_jobs > 1
        if not parallel:
            for samp_idx in self._resampler:
                samps = [np.take(arr, samp_idx, axis=self._axis) for arr in self._arrays]
                if estimator is not None:
                    if 'axis' in get_default_args(estimator):
                        e_kwargs['axis'] = self._axis
                    yield estimator(*samps, *e_args, **e_kwargs)
                else:
                    if len(samps) == 1:
                        samps = samps[0]
                    yield samps
            return
        shm_arrays = [SharedmemManager(arr, use_lock=True) for arr in self._arrays]
        pool_args = dict(shm_arrays=shm_arrays,
                         axis=self._axis, estimator=estimator,
                         e_args=e_args, e_kwargs=e_kwargs)
        with closing(mp.Pool(processes=self._n_jobs,
                             initializer=_init_pool,
                             initargs=(pool_args,))) as p:
            if self._ordered_samples:
                for samp in p.imap(_resampler, self._resampler):
                    yield samp
            else:
                for samp in p.imap_unordered(_resampler, self._resampler):
                    yield samp

    def all_samples(self, estimator=None, e_args=(), **e_kwargs):
        """Return all samples from the generator"""
        samps = list(self.sample(estimator=estimator, e_args=e_args, **e_kwargs))
        # return np.array(samps)
        # I think it is on the user to make this an array in the appropriate form
        return samps

    def estimate(self, estimator, se=False, e_args=(), **e_kwargs):
        vals = self.all_samples(estimator=estimator, e_args=e_args, **e_kwargs)
        if se:
            se = np.std(vals, axis=0) / np.sqrt(len(vals))
            return np.mean(vals, axis=0), se
        return np.mean(vals, axis=0)

    @classmethod
    def bootstrap_estimate(cls, sample, num_resample, estimator, axis=-1, n_jobs=None, e_args=(), **e_kwargs):
        bootstrapper = Bootstrap(sample, num_resample, axis=axis, n_jobs=n_jobs)
        return bootstrapper.estimate(estimator, e_args=e_args, **e_kwargs)


class Jackknife(Bootstrap):
    """
    Jackknife generator with leave-L-out resampling.

    Parameters
    ----------
    arrays: Sequence or ndarray
        Samples to jack-knife
    n_out: int
        Number of samples to leave out of each jack-knife
    axis: int
        Which axis to resample
    max_samps: int
        If N-choose-L is large, limit the number of jack-knife samples
    ordered_samples: bool
        Return jackknife samples in the normal series order: X_{\not 1}, X_{\not 2}, ...

    """

    def __init__(self, arrays, n_out=1, axis=-1, max_samps=-1, n_jobs=None, ordered_samples=False):
        if isinstance(arrays, np.ndarray):
            N = arrays.shape[axis]
        else:
            N = arrays[0].shape[axis]
        if max_samps > 0:
            num_samples = max_samps
        else:
            r = N - n_out
            num_samples = comb(N, r)
        super(Jackknife, self).__init__(arrays, num_samples, axis=axis, n_jobs=n_jobs, ordered_samples=ordered_samples)
        self.__choose = (range(N), N - n_out)
        self._max_samps = max_samps

    def _init_sampler(self):
        if self._max_samps < 0:
            self._resampler = list(combinations(*self.__choose))[::-1]
        elif self._resampler is None:
            iterable, r = self.__choose
            self._resampler = random_combinations(iterable, r, self._max_samps)
        # else random sampler is already set

    def pseudovals(self, estimator, jn_samples=(), e_args=(), **e_kwargs):
        """Return the bias-correcting pseudovalues of the estimator"""
        if not len(jn_samples):
            jn_samples = self.all_samples(estimator=estimator, e_args=e_args, **e_kwargs)
        jn_samples = np.asarray(jn_samples)
        if 'axis' in get_default_args(estimator):
            e_kwargs['axis'] = self._axis
        theta = estimator(*self._arrays, *e_args, **e_kwargs)
        N1 = float(self._arrays[0].shape[self._axis])
        d = N1 - self.__choose[1]
        return N1 / d * theta - (N1 - d) * jn_samples / d

    def estimate(self, estimator, correct_bias=True, se=False, e_args=(), **e_kwargs):
        if correct_bias:
            vals = self.pseudovals(estimator, e_args=e_args, **e_kwargs)
        else:
            vals = self.all_samples(estimator=estimator, e_args=e_args, **e_kwargs)
        if se:
            if correct_bias:
                # The variance of pseudo values is bigger than the JN variance by factor of (n - 1)
                se = np.std(vals, axis=0, ddof=0) / np.sqrt(len(vals) - 1)
            else:
                # basic JN variance is (n - 1) * var(values)
                n = len(vals)
                se = np.sqrt(n - 1) * np.std(vals, axis=0, ddof=0)
            return np.mean(vals, axis=0), se
        return np.mean(vals, axis=0)

    def bias(self, estimator, jn_samples=(), e_args=(), **e_kwargs):
        """
        Compute the jack-knife bias of an estimator.
        """

        if 'axis' in get_default_args(estimator):
            e_kwargs['axis'] = self._axis
        theta = estimator(*self._arrays, *e_args, **e_kwargs)
        pv = self.pseudovals(estimator, jn_samples=jn_samples, e_args=e_args, **e_kwargs)
        # mean of PV = theta - bias
        # bias = theta - avg{PV}
        return theta - np.mean(pv, axis=0)

    def variance(self, estimator, jn_samples=(), e_args=(), **e_kwargs):
        """
        Compute the jack-knife variance of an estimator.

        NOTE! The normalization is probably wrong for delete-d JN
        """
        if not len(jn_samples):
            jn_samples = self.all_samples(estimator, e_args=e_args, **e_kwargs)
        N = float(self._arrays[0].shape[self._axis])
        return (N - 1) * np.var(jn_samples, axis=0)

    @classmethod
    def jackknife_estimate(cls, sample, estimator, correct_bias=True, se=False,
                           axis=-1, n_jobs=None, e_args=(), **e_kwargs):
        sampler = Jackknife(sample, axis=axis, n_jobs=n_jobs)
        return sampler.estimate(estimator, correct_bias=correct_bias, se=se, e_args=e_args, **e_kwargs)
