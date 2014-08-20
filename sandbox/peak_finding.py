"""Graveyard module for peak-finding excursion"""
from ecoglib.numutils import *

def mini_mean_shift(f, ix, m):
    #fdens = density_normalize(f[ix-m:ix+m+1])
    fdens = f[ix-m:ix+m+1]
    fdens = fdens - fdens.min()
    return (fdens * np.arange(ix-m, ix+m+1)).sum() / fdens.sum()

def peak_to_peak(x, m, p=4, xwin=(), msiter=4, points=False):
    # m is approimately the characteristic width of a peak
    # x is 2d, n_array x n_pts
    oshape = x.shape
    x = x.reshape(-1, oshape[-1])
    
    # force m to be odd and get 1st derivatives
    m = 2 * (m//2) + 1
    dx = savitzky_golay(x, m, p, deriv=1, axis=-1)

    if not xwin:
        xwin = (0, dx.shape[axis])
    xwin = map(int, xwin)

    pk_dx = np.argmax(dx[:, xwin[0]:xwin[1]], axis=-1) + xwin[0]
    # these are our starting points for peak finding

    pos_pks = np.zeros(x.shape[0], 'i')
    neg_pks = np.zeros(x.shape[0], 'i')
    bw = m//2
    for n in xrange(x.shape[0]):
        trace = x[n].copy()
        # must be all positive
        #trace = trace - trace.min()
        ix0 = pk_dx[n]
        k = 0
        while k < msiter:
            ix = round(mini_mean_shift(trace, ix0, bw))
            if not ix-ix0:
                break
            ix0 = ix
            k += 1
        pos_pks[n] = (ix-bw) + np.argmax(trace[ix-bw:ix+bw+1])

        # grab a slice rewinding a little bit from here (in order
        # to keep window slicing in-bounds)
        trace = -x[n, pos_pks[n]-m:] + x[n, pos_pks[n]]
        ix0 = m + bw
        skip = 1
        k = 0
        # allow more mean-shift iterations, since the initial point
        # is a very rough guess
        while k < 2*msiter:
            if ix0 > len(trace) - bw:
                # we have no idea where we are!
                # return global minimum
                ix = np.argmin(trace)
                break
            ix = round(mini_mean_shift(trace, ix0, bw))
            if not ix-ix0:
                break
            if ix < ix0:
                # going backwards.. reset with bigger skip ahead
                ix0 = m + skip*bw
                skip += 1
                k = 0
                continue
            ix0 = ix
            k += 1

        #neg_pks[n] = round(ix) + pos_pks[n] - m
        neg_pks[n] = (ix-bw) + np.argmax(trace[ix-bw:ix+bw+1]) + \
          (pos_pks[n]-m)

    cx = np.arange(x.shape[0])
    p2p = x[ (cx, pos_pks) ] - x[ (cx, neg_pks) ]

    if points:
        return map(lambda x: x.reshape(oshape[:-1]), (p2p, neg_pks, pos_pks))
    else:
        return p2p.reshape(oshape[:-1])

    
def peak_to_peak2(x, m, p=4, xwin=(), msiter=4, points=False):
    # m is approimately the characteristic width of a peak
    # x is 2d, n_array x n_pts
    oshape = x.shape
    x = x.reshape(-1, oshape[-1])
    
    # force m to be odd and get 1st derivatives
    m = 2 * (m//2) + 1
    if m < 2*p:
        # m *must* satisify m > p+1
        # but pad to 2*p - 1 for stability
        m = 2*p - 1
    
    dx = savitzky_golay(x, m, p, deriv=1, axis=-1)

    if not xwin:
        xwin = (0, dx.shape[axis])
    xwin = map(int, xwin)

    pk_dx = np.argmax(dx[:, xwin[0]:xwin[1]], axis=-1) + xwin[0]
    # these are our starting points for peak finding

    pos_pks = np.zeros(x.shape[0], 'i')
    neg_pks = np.zeros(x.shape[0], 'i')
    bw = m//2
    for n in xrange(x.shape[0]):
        trace = x[n].copy()
        # must be all positive
        #trace = trace - trace.min()
        ix = pk_dx[n]
        pos_pks[n] = (ix-m) + np.argmax(trace[ix-m:ix+m+1])

        # grab a slice rewinding a little bit from here (in order
        # to keep window slicing in-bounds)
        trace = -x[n, pos_pks[n]-m:] + x[n, pos_pks[n]]
        ix0 = m + bw
        skip = 1
        k = 0
        # allow more mean-shift iterations, since the initial point
        # is a very rough guess
        while k < msiter:
            if ix0 >= len(trace) - bw:
                # we have no idea where we are!
                # return global minimum
                ix = np.argmin(trace)
                break
            ix = round(mini_mean_shift(trace, ix0, bw))
            if not ix-ix0:
                break
            if ix < ix0:
                # going backwards.. reset with bigger skip ahead
                ix0 = m + skip*bw
                skip += 1
                k = 0
                continue
            ix0 = ix
            k += 1

        #neg_pks[n] = round(ix) + pos_pks[n] - m
        neg_pks[n] = (ix-bw) + np.argmax(trace[ix-bw:ix+bw+1]) + \
          (pos_pks[n]-m)

    cx = np.arange(x.shape[0])
    p2p = x[ (cx, pos_pks) ] - x[ (cx, neg_pks) ]

    if points:
        return map(lambda x: x.reshape(oshape[:-1]), (p2p, neg_pks, pos_pks))
    else:
        return p2p.reshape(oshape[:-1])

def peak_to_peak3(x, m, p=4, xwin=(), msiter=4, points=False):
    # m is approimately the characteristic width of a peak
    # x is 2d, n_array x n_pts
    oshape = x.shape
    x = x.reshape(-1, oshape[-1])
    
    # force m to be odd and get 1st derivatives
    m = 2 * (m//2) + 1
    if m < 2*p:
        # m *must* satisify m > p+1
        # but pad to 2*p - 1 for stability
        m = 2*p - 1
    
    sx, dx = savitzky_golay(x, m, p, deriv=(0,1), axis=-1)

    if not xwin:
        xwin = (0, dx.shape[axis])
    xwin = map(int, xwin)

    pk_dx = np.argmax(dx[:, xwin[0]:xwin[1]], axis=-1) + xwin[0]
    # these are our starting points for peak finding

    pos_pks = np.zeros(x.shape[0], 'i')
    neg_pks = np.zeros(x.shape[0], 'i')
    bw = m//2
    for n in xrange(x.shape[0]):
        trace = x[n].copy()
        # must be all positive
        #trace = trace - trace.min()
        ix = pk_dx[n]
        rewind = max(0, int(ix-0.5*m))
        pos_pks[n] = rewind + np.argmax(trace[rewind:int(ix+0.5*m)])

        #trace = sx[n, pos_pks[n]:pos_pks[n]+5*m]
        trace = x[n, pos_pks[n]:pos_pks[n]+3*m]
        neg_pks[n] = np.argmin(trace) + pos_pks[n]

    cx = np.arange(x.shape[0])
    p2p = x[ (cx, pos_pks) ] - x[ (cx, neg_pks) ]

    if points:
        return map(lambda x: x.reshape(oshape[:-1]), (p2p, neg_pks, pos_pks))
    else:
        return p2p.reshape(oshape[:-1])

    
