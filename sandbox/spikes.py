
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
import matplotlib.pyplot as pp

## def find_spikes(mn_trace, winsize, spikewidth, measure='energy'):

##     if measure.lower()=='energy':
##         raw = mn_trace**2
##     elif measure.lower() == 'length':
##         raw = np.abs(mn_trace)

##     win_meas = np.convolve(raw, np.ones(winsize), mode='same')

##     sm_meas = savitzky_golay(win_meas, 2*spikewidth+1, order=2, deriv=1)
##     # want to look for pos-to-neg crossings
##     neg_deriv = (sm_meas < 0).astype('i')
##     peaks = np.where(np.diff(neg_deriv) > 0)[0] + 1

##     sm_meas = savitzky_golay(win_meas, 2*spikewidth+1, order=2)
##     peak_vals = sm_meas[peaks]
##     bc, bv = np.histogram(peak_vals, bins=200)
##     bc_sg_dx = savitzky_golay(bc.astype('d'), 11, order=2, deriv=1)
##     ndx = np.where(bc_sg_dx<0)[0][0]
##     bc_sg_dx = bc_sg_dx[ndx:]
##     pdx = np.where(bc_sg_dx>=0)[0][0]

##     bv_thresh = bv[ndx+pdx]
##     bv_thresh = 0
##     print len(peaks)
##     peaks = peaks[ peak_vals >= bv_thresh ]

##     spikes = list()
##     pre_spike = spikewidth//2
##     post_spike = spikewidth - pre_spike
##     spikes = [ (p - pre_spike, p + post_spike) for p in peaks ]

##     return spikes, sm_meas #sm_meas[peaks]

    ## gsize = 2*(winsize//2) + 1
    ## sm_meas = np.convolve(
    ##     win_meas, signal.gaussian(gsize, winsize/2, sym=True), mode='same'
    ##     )

    ## if measure.lower() == 'energy':
    ##     sx = (win_meas > 2e-6).astype('i')

    ## dsx = np.r_[0, np.diff(sx)]

    ## rising = np.where(dsx>0)[0]
    ## falling = np.where(dsx<0)[0]

    ## for r, f in zip(rising, falling):
    ##     mx_pt = np.argmax( sm_energy[r:f] ) + r
    ##     spikes.append( (mx_pt - pre_spike, mx_pt + post_spike) )

    ## return spikes, sm_energy

def find_spikes(d, spikewidth, Fs, spikewindow=None, fudge=0.8):
    spike_samps = int( spikewidth * Fs + 0.5 )
    if not spikewindow:
        spikewindow = spike_samps
    else:
        spikewindow = int( spikewindow * Fs + 0.5 )

    dl = np.abs(d).mean(axis=1)

    # don't want to lowpass below 2.5x spike width resolution--heuristically
    win_sz = int(np.floor(2.5 / spikewidth))
    win_sz += (win_sz+1) % 2
    print(win_sz)

    idl = np.convolve( dl, np.ones(win_sz)/win_sz, mode='same' )

    bc, bv = np.histogram(idl, bins=500)
    bc_sm_dx = savitzky_golay(bc, 51, order=4, deriv=1)
    ndx = (bc_sm_dx < 0).astype('i')
    # choose second peak -- or second pos-to-neg crossing in d/dx
    ix = np.where(np.diff(ndx) > 0)[0][1]

    tau = bv[ix] * fudge
    print(tau)
    survivors = (idl > tau).astype('i')
    ds = np.diff(survivors)
    spk_start = np.where(ds > 0)[0] + 1
    spk_stop = np.where(ds < 0)[0] + 1

    spikes = list()
    pre_spike = spikewindow//2
    post_spike = spikewindow - pre_spike
    for start, stop in zip(spk_start, spk_stop):
        # check for a super-threshold width of > (1 + n*0.8) * spike_samps
        # with n > 0
        if (stop-start) > 1.8 * spike_samps:
            n_find = int( (float(stop-start)/spike_samps - 1) / .8 ) + 1
            # need to find a few zero crossings of a very smooth derivative
            strip = idl[start:stop]
            dx = savitzky_golay(
                strip, win_sz + 10, order=4, deriv=1
                )
            ndx = (dx < 0).astype('i')
            zc = np.where(np.diff(ndx) > 0)[0]
            n_find = min(len(zc), n_find)
            # find which zero crossings are the top n amplitude peaks
            pks = np.argsort(strip[zc])[-n_find:]
            idc = zc[pks] + start
        else:
            idc = [np.argmax(idl[start:stop]) + start]
        for ix in idc:
            spikes.append( (ix - pre_spike, ix + post_spike) )
    return spikes



def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute
        (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat(
        [[k**i for i in order_range]
         for k in range(-half_window, half_window+1)]
        )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def mark_spikes(mn_trace, spikes):
    import matplotlib.pyplot as pp
    import matplotlib.collections as collections
    f = pp.figure()
    ax = f.add_subplot(111)
    t = np.arange(len(mn_trace))
    ax.plot(t, mn_trace)

    xr = [ (x[0], x[1]-x[0]) for x in spikes ]
    yr = ax.get_ylim()
    yr = (yr[0], yr[1] - yr[0])
    cx = collections.BrokenBarHCollection(
        xr, yr, facecolor='red', alpha=0.3
        )
    ax.add_collection(cx)
    pp.show()


def simple_spikes(d, thresh, t_refractory, t_min):
    #dmx = np.abs(d).max(1)
    dmx = np.abs(np.mean(d, axis=1))
    super_thresh = (dmx > thresh).astype('i')
    rising_edge = np.where(np.diff(super_thresh) > 0)[0] + 1
    falling_edge = np.where(np.diff(super_thresh) < 0)[0] + 1

    spikes = list()
    last_spike = -t_refractory
    print(len(rising_edge))
    for start, stop in zip(rising_edge, falling_edge):
        if stop - start < t_min:
            continue
        if start < last_spike + t_refractory:
            continue
        else:
            last_spike = start
            spikes.append(start)
    return spikes


def delay_map(spike_vecs, arr_dims, interp=4):
    n_spikes = spike_vecs.shape[0]
    sp_frames = spike_vecs.reshape( n_spikes, -1, arr_dims )
    n_pts = sp_frames.shape[1]
    tx = np.arange(n_pts)
    tx_plot = np.linspace(tx[0], tx[-1], (n_pts-1)*interp)
    ifun = interp1d(tx, sp_frames, kind='cubic', axis=1)
    sp_frames = ifun(tx_plot)
    mn_spikes = np.mean(sp_frames, axis=-1)
    lag_maps = np.zeros((n_spikes, arr_dims))
    n = 0
    for mn_spk, frames in zip(mn_spikes, sp_frames):
        cross_corr = ndimage.convolve1d(
            frames, mn_spk[::-1], axis=0, mode='constant'
            )
        lag_map = np.argmax(cross_corr, axis=0)
        centered_time = np.argmax(np.abs(mn_spk))
        #lag_map -= centered_time
        lag_maps[n] = lag_map.astype('d') / interp
        n = n + 1
    return lag_maps


def plot_maps(maps, max_plot=30):
    n_maps = maps.shape[0]
    if n_maps > max_plot:
        # choose random subset
        r = np.arange(n_maps)
        np.random.shuffle(r)
        return plot_maps(maps[r[:max_plot]])

    P1 = int( np.ceil(np.sqrt(float(n_maps))) )
    P2 = int( np.ceil(n_maps / float(P1)) )
    f = pp.figure()
    norm = pp.normalize(maps.min(), maps.max())
    for p, mp in enumerate(maps):
        pp.subplot(P1, P2, p+1)
        pp.imshow(mp, interpolation='nearest', norm=norm)
        pp.gca().xaxis.set_visible(False)
        pp.gca().yaxis.set_visible(False)
        pp.gca().axis('image')
    f.tight_layout()
    return f
