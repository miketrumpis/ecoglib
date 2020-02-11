import numpy as np
import random

#### Utility for quick range finding XXX: way too liberal of a bound!
def stochastic_limits(x, n_samps=100, conf=98.0):
    """
    Use Markov's inequality to estimate a bound on the
    absolute values in the array x.
    """
    n = len(x)
    r_pts = random.sample(xrange(n), n_samps)
    r_samps = np.take(x, r_pts)
    # unbiased estimator??
    e_abs = np.abs(r_samps).mean()
    # Pr{ |X| > t } <= E{|X|}/t = 1 - conf/100
    # so find the threshold for which there is only a
    # (100-conf)% chance that |X| is greater
    return e_abs/(1.0-conf/100.0)

#### Utility for safe sub-slicing
def safe_slice(x, start, num, fill=np.nan):
    """
    Slice array x contiguously (along 1st dimension) for num pts,
    starting from start. If all or part of the range lies outside
    of the actual bounds of x, then fill with NaN
    """
    lx = x.shape[0]
    sub_shape = (num,) + x.shape[1:]
    if start < 0 or start + num > lx:
        sx = np.empty(sub_shape, dtype=x.dtype)
        if start <= -num or start >= lx:
            sx.fill(fill)
            # range is entirely outside
            return sx
        if start < 0 and start + num > lx:
            # valid data is in the middle of the range
            if fill == 'extend':
                # only makes sense for regularly spaced pts
                dx = x[1] - x[0]
                bwd = np.arange(1, -start+1)
                sx[:-start, ...] = x[0] - bwd[::-1]*dx
                fwd = np.arange(1, num-lx+start+1)                
                sx[-start+lx:num, ...] = x[-1] + fwd*dx
            else:
                sx.fill(fill)
            sx[-start:-start+lx] = x
            return sx
        if start < 0:
            if fill == 'extend':
                # extend time back
                sx[:-start, ...] = x[0] - x[1:-start+1][::-1]
            else:
                # fill beginning ( where i < 0 ) with NaN
                sx[:-start, ...] = fill

            # fill the rest with x
            sx[-start:, ...] = x[:(num + start), ...]
        else:
            sx[:(lx-start), ...] = x[start:, ...]
            if fill == 'extend':
                # extend range with this many points
                n_fill = num - (lx - start)
                x_template = x[1:n_fill+1] - x[0]
                sx[(lx-start):, ...] = x[-1] + x_template
            else:
                sx[(lx-start):, ...] = fill
    else:
        sx = x[start:start+num, ...]
    return sx