import numpy as np
from nitime.algorithms import dpss_windows
from sandbox.array_split import split_at

# This still crashes on Mac due to VecLib not playing nicely with forks
# @split_at() 
def slepian_projection(data, BW, Fs, Kmax=None, w0=0, baseband=False):
    shp = data.shape
    if len(shp) < 2:
        data = np.atleast_2d(data)
    if len(shp) > 2:
        data = data.reshape( shp[0] * shp[1], -1 )
    nchan, npts = data.shape
    # find NW which is also TW
    T = npts / Fs
    # round to the nearest multiple of 1/2
    TW = round( 2 * T * BW ) / 2.0
    K = 2 * TW
    if K < 1:
        min_bw = 0.5 / T
        err = 'BW is too small for the window size: ' \
          'minimum BW={0}'.format(min_bw)
        raise ValueError(err)
    if Kmax is not None:
        K = min(Kmax, K)
    dpss, _ = dpss_windows(npts, TW, K)
    if w0 == 0:
        # shortcut for lowpass only
        w = data.dot( dpss.T )
        # w is (nchan x K)
        # expand Slepians as (nchan x npts)
        bp =  w.dot( dpss )
    else:
        t = np.arange(npts)
        dpss_pf = np.exp(2j * np.pi * w0 * t / Fs) * dpss
        dpss_nf = np.exp(-2j * np.pi * w0 * t / Fs) * dpss
        nrm = np.sqrt( ( dpss_nf * dpss_nf.conj() ).sum(-1) )
        dpss_nf /= nrm[:, None]
        wp = data.dot( dpss_pf.conj().T )
        wn = data.dot( dpss_nf.conj().T )
        if baseband:
            bp = ( wp.dot( dpss ) + wn.dot( dpss ) ).real
        else:
            bp = ( wp.dot( dpss_pf ) + wn.dot( dpss_nf ) ).real
    return bp.reshape(shp)

