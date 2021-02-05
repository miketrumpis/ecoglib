"""Iterators and tools for jackknife estimators"""
import random
from warnings import warn
import numpy as np
from scipy.special import comb
from itertools import combinations
from contextlib import ExitStack
import ecogdata.parallel.mproc as mp
from ecogdata.util import get_default_args
from ecogdata.parallel.sharedmem import SharedmemManager
from ecogdata.parallel.mproc import timestamp


__all__ = ['random_combinations', 'Jackknife', 'Bootstrap']


def random_combinations(iterable, r, n):
    """Pull n random selections from itertools.combinations(iterable, r)"""
    if n > comb(len(iterable), r):
        raise ValueError('Not enough combinations for {0} samples'.format(n))
    pulls = set()
    pool = tuple(iterable)
    L = len(pool)
    # TODO: this is VERY SLOW
    while len(pulls) < n:
        indices = sorted(random.sample(range(L), r))
        pulls.add(tuple(pool[i] for i in indices))
    return pulls


def resample_with_replacement(max_n, sample_size, proba, num_samples=None):
    """

    Parameters
    ----------
    max_n: int
        maximum sample index to draw
    sample_size: int
        size of sample from range(0, max_n)
    proba: ndarray
        if given, the probability of drawing any index
    num_samples: int
        if given, generate this many permutations (otherwise generation is open-ended)

    Returns
    -------
    Yields bootstrap-style resampling with replacement

    """
    r = np.arange(max_n)
    i = 0
    while True:
        if proba is None:
            yield np.random.randint(0, max_n, sample_size)
        else:
            yield np.random.choice(r, sample_size, p=proba)
        i += 1
        if num_samples is not None and i == num_samples:
            return


def leave_n_out_resample(n, r, max_samples, ordered_samples):
    """

    Parameters
    ----------
    n: int
        original sample size
    r: int
        leave-n-out sample size
    max_samples: int or None
        limit leave-n-out combinations to a random draw of max_samples
    ordered_samples: bool
        order samples as (leave-0, leave-1, ...)

    Returns
    -------
    Yields leave-n-out permutations

    """
    if max_samples is None:
        resampler = combinations(range(n), r)
        if ordered_samples:
            resampler = list(resampler)[::-1]
    else:
        rng = range(n)
        resampler = random_combinations(rng, r, max_samples)
    for index in resampler:
        yield index


class BootstrapSampler(mp.Process):

    def __init__(self, resamples: mp.JoinableQueue, results: mp.Queue, shm_arrays, shm_output, axis, estimator,
                 e_args, e_kwargs, resample_args):
        mp.Process.__init__(self)
        # this is a JoinableQueue that the process will grab resampling permutations from
        self.resamples = resamples
        # this is a Queue that the the process will push results into
        self.results = results
        self.shm_arrays = shm_arrays
        self.shm_output = shm_output
        self.axis = axis
        self.estimator = estimator
        self.e_args = e_args
        self.e_kwargs = e_kwargs
        self.resample_args = resample_args

    def sample_generator(self):
        max_n, sample_size, proba = self.resample_args
        for index in resample_with_replacement(max_n, sample_size, proba):
            yield index

    def _seed_rng(self):
        time_parts = timestamp().split('-')
        # sum HH-MM-SS
        p1 = sum([int(p) for p in time_parts[:3]])
        # multiply by other part
        seed = p1 * int(time_parts[-1])
        np.random.seed(seed)

    def run(self):
        info = mp.get_logger().info
        shm_arrays = self.shm_arrays
        shm_output = self.shm_output
        axis = self.axis
        estimator = self.estimator
        e_args = self.e_args
        e_kwargs = self.e_kwargs
        sample_generator = self.sample_generator()
        if estimator is not None:
            # if there is *not*  an axis, then hope for the best
            if 'axis' in get_default_args(estimator):
                e_kwargs['axis'] = axis
            info('Estimator args: {}, {}'.format(e_args, e_kwargs))
        # be sure to start the RNG in a different state than peers
        self._seed_rng()
        info('Process running {}'.format(timestamp()))
        while True:
            job = self.resamples.get()
            if job is None:
                info('Caught exit signal: {}'.format(timestamp()))
                self.resamples.task_done()
                break
            # this state is important for deterministic sampling (i.e. jackknife)
            self.sample_num = job
            permutation = next(sample_generator)

            # This is the *safe* way to go: resample the arrays under a lock.
            # It should also prevent making copies in each subprocess
            if shm_output is None:
                with ExitStack() as stack:
                    arrays = [stack.enter_context(shm.get_ndarray()) for shm in shm_arrays]
                    info('Sampling perm {} from arrays {}'.format(self.sample_num, timestamp()))
                    samps = [np.take(array, permutation, axis=axis) for array in arrays]
            else:
                with ExitStack() as stack:
                    arrays = [stack.enter_context(shm.get_ndarray()) for shm in shm_arrays]
                    # de-reference the output memory that was allocated for this sample
                    out_managers = shm_output[self.sample_num]
                    out_arrays = [stack.enter_context(shm.get_ndarray()) for shm in out_managers]
                    samps = [np.take(array, permutation, axis=axis, out=out)
                             for (array, out) in zip(arrays, out_arrays)]
            info('Doing perm {} estimator method {}'.format(self.sample_num, timestamp()))
            if estimator is not None:
                r = estimator(*samps, *e_args, **e_kwargs)
                info('**** Estimator {} done {} ****'.format(self.sample_num, timestamp()))
                self.resamples.task_done()
                self.results.put(r)
            else:
                # if we just filled pre-allocated memory, return the list index that was filled
                self.resamples.task_done()
                if shm_output is not None:
                    self.results.put(self.sample_num)
                else:
                    # otherwise return the actual data
                    if len(samps) == 1:
                        samps = samps[0]
                    self.results.put(samps)
        info('Process ending {}'.format(timestamp()))


class JackknifeSampler(BootstrapSampler):

    def sample_generator(self):
        # since resampling is deterministic for the Jackknife, need
        # to create all permutations first and yield the one corresponding
        # to the present job
        n, n_out, max_samples, ordered_samples = self.resample_args
        all_permutations = list(leave_n_out_resample(n, n_out, max_samples, ordered_samples))
        while True:
            yield all_permutations[self.sample_num]


class Bootstrap:
    """
    Bootstrap resampler
    """

    def __init__(self, arrays, num_samples, axis=-1, proba=None, sample_size=None, n_jobs=1, ordered_samples=False,
                 subprocess_logging='error'):
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
        proba: ndarray
            Draw samples with this probability (uniform probability if proba is None).
        sample_size: int
            If given, then pull so many samples with replacement from the array. Normally equal to the original
            sample size.
        n_jobs: int
            Number of processes for bootstrap sampling (parallel is helpful if estimator cost is > O(n))
        ordered_samples: bool
            ???
        subprocess_logging: str
            Logging level (e.g. "error", "info", ...)

        """

        if not isinstance(arrays, (tuple, list)):
            self._arrays = [np.asanyarray(arrays)]
        else:
            self._arrays = [np.asanyarray(a) for a in arrays]
        if proba is not None:
            self._proba = np.asanyarray(proba)
            # make sure it is unity normalized
            self._proba = self._proba / self._proba.sum()
        else:
            self._proba = None
        self._axis = axis
        self._num_samples = num_samples
        if sample_size is None:
            self._sample_size = self._arrays[0].shape[axis]
        else:
            self._sample_size = sample_size
        self._resampler = None
        self._ordered_samples = ordered_samples
        self._n_jobs = n_jobs
        self._loglevel = subprocess_logging

    def _init_sampler(self):
        max_n = self._arrays[0].shape[self._axis]
        samp_size = self._sample_size
        return resample_with_replacement(max_n, samp_size, self._proba, num_samples=self._num_samples)

    def _alloc_output_memory(self):
        warn('Generating data resamples is generally SLOW with multiprocessing', RuntimeWarning)
        # each sample will have n arrays with these shapes
        output_shapes = [list(a.shape) for a in self._arrays]
        output_types = [a.dtype.char for a in self._arrays]
        for s in output_shapes:
            s[self._axis] = self._sample_size
        # create len(self) times these arrays in shared memory
        shm_output = []
        output_managers = []
        for _ in range(len(self)):
            arrays = [SharedmemManager.shared_ndarray(s, t) for (s, t) in zip(output_shapes, output_types)]
            managers = [SharedmemManager(a) for a in arrays]
            shm_output.append(arrays)
            output_managers.append(managers)
        return shm_output, output_managers

    def __len__(self):
        return self._num_samples

    def _sampling_processes(self, task_queue, results_queue, estimator, e_args, e_kwargs):
        """
        Create the correct resampling subprocess.

        Parameters
        ----------
        task_queue: mp.JoinableQueue
            Queue to push resample tasks to workers
        results_queue: mp.Queue
            Queue to receive resample estimators from workers
        estimator: callable
            Statistic to estimate
        e_args: tuple
            Additional arguments for the estimator
        e_kwargs: dict
            Additional keyword arguments for the estimator

        Returns
        -------
        samplers: list
            List of worker process objects
        output_arrays: list
            List of shared memory arrays (if workers are returning full resamples)

        """
        shm_arrays = [SharedmemManager(arr, use_lock=True) for arr in self._arrays]
        if estimator is None:
            # if generating samples avoid heavy data passing -- pre-allocate shared memory
            print('allocating output memory {}'.format(timestamp()), end='... ')
            output_arrays, output_managers = self._alloc_output_memory()
            print('done {}'.format(timestamp()))
        else:
            output_arrays = output_managers = None

        resample_args = (self._arrays[0].shape[self._axis], self._sample_size, self._proba)
        samplers = []
        for s in range(self._n_jobs):
            s = BootstrapSampler(task_queue, results_queue, shm_arrays, output_managers, self._axis,
                                 estimator, e_args, e_kwargs, resample_args)
            samplers.append(s)
        return samplers, output_arrays

    def sample(self, estimator=None, e_args=(), **e_kwargs):
        """
        Generator for bootstrap samples, or estimates from the samples. The estimator must be a
        callable.

        Parameters
        ----------
        estimator: callable
            Infer statistic distribution using this estimator
        e_args: tuple
            Additional arguments for the estimator
        e_kwargs:
            Additional keyword arguments for the estimator

        Yields
        ------
        r:
            Bootstrap estimate (or full bootstrap sample if estimator is None)

        """

        parallel = self._n_jobs > 1
        if not parallel:
            for samp_idx in self._init_sampler():
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
        # If parallel, build the workers, start the workers, push tasks, and collect results
        permutations = mp.JoinableQueue()
        resample_results = mp.Queue()
        samplers, output_arrays = self._sampling_processes(permutations, resample_results, estimator, e_args, e_kwargs)
        with mp.make_stderr_logger(self._loglevel):
            for s in samplers:
                s.start()
            # put jobs into the queue (just the sample number)
            for s_num in range(self._num_samples):
                permutations.put(s_num)
            # put the stop signal
            for n in range(len(samplers)):
                permutations.put(None)
            # wait for the results
            permutations.join()
            # yield results
            for n in range(self._num_samples):
                r = resample_results.get()
                if output_arrays is not None:
                    r = output_arrays[r]
                yield r

    def all_samples(self, estimator=None, e_args=(), **e_kwargs):
        """Return all samples from the generator"""
        samps = list(self.sample(estimator=estimator, e_args=e_args, **e_kwargs))
        # return np.array(samps)
        # I think it is on the user to make this an array in the appropriate form
        return samps

    def estimate(self, estimator, ci=0.95, e_args=(), **e_kwargs):
        """
        Make a bootstrapped estimate with optional confidence interval.

        Parameters
        ----------
        estimator: callable
            The estimator to bootstrap (e.g. "numpy.mean", trivially)
        ci: float, Str
            If a number < 1, then calculate the (1 - alpha) confidence interval based on percentiles.
            E.g. ci=0.95 returns the [0.025, 0.0975] quantile points.
            If ci == 'se', then return the standard deviation of the bootstrapped estimates.
        e_args: tuple
            Extra positional arguments for the estimator.
        e_kwargs:
            Extra keyword arguments for the estimator.

        Returns
        -------
        sample_mean:
            Estimator of the full sample
        mean_est:
            Mean of the bootstrapped estimates
        error:
            Confidence interval or SD of bootstrapped estimates

        """
        vals = self.all_samples(estimator=estimator, e_args=e_args, **e_kwargs)
        if isinstance(ci, str):
            iv = np.std(vals, axis=0)
        else:
            tol = 100 * (1 - ci) / 2
            iv = np.percentile(vals, [tol, 100 - tol], axis=0)
        sample_mean = estimator(*self._arrays, *e_args, **e_kwargs)
        return sample_mean, np.mean(vals, axis=0), iv

    @classmethod
    def bootstrap_estimate(cls, sample, num_resample, estimator, axis=-1, n_jobs=1, ci=0.95, e_args=(), **e_kwargs):
        """
        Make a bootstrapped estimate with optional confidence interval.

        Parameters
        ----------
        sample: ndarray
            Data array (or list of arrays) to resample
        num_resample: int
            Number of bootstrap resamples to use
        estimator: callable
            The estimator to bootstrap (e.g. "numpy.mean", trivially)
        axis: int
            If the data array is multivariate, the estimator works as estimator(data, ..., axis=axis)
        n_jobs: int
            Number of processes for bootstrap sampling (parallel is helpful if estimator cost is > O(n))
        ci: float, Str
            If a number < 1, then calculate the (1 - alpha) confidence interval based on percentiles.
            E.g. ci=0.95 returns the [0.025, 0.0975] quantile points.
            If ci == 'se', then return the standard deviation of the bootstrapped estimates.
        e_args: tuple
            Extra positional arguments for the estimator.
        e_kwargs:
            Extra keyword arguments for the estimator.

        Returns
        -------
        sample_mean:
            Estimator of the full sample
        mean_est:
            Mean of the bootstrapped estimates
        error:
            Confidence interval or SD of bootstrapped estimates

        """
        bootstrapper = Bootstrap(sample, num_resample, axis=axis, n_jobs=n_jobs)
        return bootstrapper.estimate(estimator, ci=ci, e_args=e_args, **e_kwargs)


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

    def __init__(self, arrays, n_out=1, axis=-1, max_samps=None, n_jobs=1, ordered_samples=False,
                 subprocess_logging='error'):
        if isinstance(arrays, np.ndarray):
            N = arrays.shape[axis]
        else:
            N = arrays[0].shape[axis]
        if max_samps is not None:
            num_samples = max_samps
        else:
            r = N - n_out
            num_samples = int(comb(N, r))
        super(Jackknife, self).__init__(arrays, num_samples, axis=axis, n_jobs=n_jobs,
                                        ordered_samples=ordered_samples, subprocess_logging=subprocess_logging)
        # self.__choose = (range(N), N - n_out)
        self._n_out = n_out
        self._max_samps = max_samps

    def _sampling_processes(self, task_queue, results_queue, estimator, e_args, e_kwargs):
        """
        Create the correct resampling subprocess.

        Parameters
        ----------
        task_queue: mp.JoinableQueue
            Queue to push resample tasks to workers
        results_queue: mp.Queue
            Queue to receive resample estimators from workers
        estimator: callable
            Statistic to estimate
        e_args: tuple
            Additional arguments for the estimator
        e_kwargs: dict
            Additional keyword arguments for the estimator

        Returns
        -------
        samplers: list
            List of worker process objects
        output_arrays: list
            List of shared memory arrays (if workers are returning full resamples)

        """
        shm_arrays = [SharedmemManager(arr, use_lock=True) for arr in self._arrays]
        if estimator is None:
            # if generating samples avoid heavy data passing -- pre-allocate shared memory
            print('allocating output memory {}'.format(timestamp()), end='... ')
            output_arrays, output_managers = self._alloc_output_memory()
            print('done {}'.format(timestamp()))
        else:
            output_arrays = output_managers = None

        n = self._arrays[0].shape[self._axis]
        r = n - self._n_out
        resample_args = (n, r, self._max_samps, self._ordered_samples)
        samplers = []
        for s in range(self._n_jobs):
            s = JackknifeSampler(task_queue, results_queue, shm_arrays, output_managers, self._axis,
                                 estimator, e_args, e_kwargs, resample_args)
            samplers.append(s)
        return samplers, output_arrays

    def _init_sampler(self):
        n = self._arrays[0].shape[self._axis]
        r = n - self._n_out
        return leave_n_out_resample(n, r, self._max_samps, self._ordered_samples)

    def pseudovals(self, estimator, jn_samples=(), e_args=(), **e_kwargs):
        """Return the bias-correcting pseudovalues of the estimator"""
        if not len(jn_samples):
            jn_samples = self.all_samples(estimator=estimator, e_args=e_args, **e_kwargs)
        jn_samples = np.asarray(jn_samples)
        if 'axis' in get_default_args(estimator):
            e_kwargs['axis'] = self._axis
        theta = estimator(*self._arrays, *e_args, **e_kwargs)
        N1 = float(self._arrays[0].shape[self._axis])
        d = self._n_out
        return N1 / d * theta - (N1 - d) * jn_samples / d

    def estimate(self, estimator, correct_bias=True, se=True, e_args=(), **e_kwargs):
        """
        Jackknife estimate of mean and standard error.

        Parameters
        ----------
        estimator: callable
            Infer distribution of this statistic.
        correct_bias: bool
            Jackknife is used to correct bias (but bias estimate itself might be highly variable)
        se: bool
            Jackknife is typically used to find the standard error of the estimator (True is recommended).
        e_args: tuple
            Additional arguments for the estimator
        e_kwargs: dict
            Additional keyword arguments for the estimator

        Returns
        -------
        mu:
            Jackknifed estimator mean
        se:
            Jackknifed estimator error (if se is True)

        """
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
    def jackknife_estimate(cls, sample, estimator, correct_bias=True, se=True,
                           axis=-1, max_samps=None, n_jobs=1, e_args=(), **e_kwargs):
        """
        Jackknife estimate of mean and standard error.

        Parameters
        ----------
        sample: ndarray
            Data array (or list of arrays) to jackknife-resample
        estimator: callable
            Infer distribution of this statistic.
        correct_bias: bool
            Jackknife is used to correct bias (but bias estimate itself might be highly variable)
        se: bool
            Jackknife is typically used to find the standard error of the estimator (True is recommended).
        axis: int
            For multivariate data, statistic estimator works as estimator(sample, ..., axis=axis)
        max_samps: int, None
            If the number of leave-n-out permutations is very large, limit it to max_samps
        n_jobs: int
            Number of processes for bootstrap sampling (parallel is helpful if estimator cost is > O(n))
        e_args: tuple
            Additional arguments for the estimator
        e_kwargs: dict
            Additional keyword arguments for the estimator

        Returns
        -------
        mu:
            Jackknifed estimator mean
        se:
            Jackknifed estimator error (if se is True)

        """

        sampler = Jackknife(sample, axis=axis, max_samps=max_samps, n_jobs=n_jobs)
        return sampler.estimate(estimator, correct_bias=correct_bias, se=se, e_args=e_args, **e_kwargs)


# TODO:
#  interesting doc: https://arxiv.org/pdf/1411.5279.pdf
#  * done: bootstrap estimate: should really emphasize CIs rather than bootstrap mean (which can be biased)
#  * bootstrap-t interval (for each resample, form t_i = (avg(x*) - avg(x)) / se(x*) -- probably requires estimator
#    function to return SE per sample
#  * bootstrap regression (optional residual resampling)
#  * bootstrap t-test