"""
One-stop shopping for digital filtering of arrays
"""
from __future__ import division
import numpy as np
from .design import butter_bp, cheby1_bp, cheby2_bp, notch
from ecoglib.util import get_default_args
from sandbox.array_split import shared_ndarray
import scipy.signal as signal

__all__ = [ 'filter_array', 'notch_all' ]

def _get_poles_zeros(destype, **filt_args):
    if destype.lower().startswith('butter'):
        return butter_bp(**filt_args)

    des_lookup = dict(cheby1=cheby1_bp, cheby2=cheby2_bp, notch=notch)
    desfun = des_lookup[destype]    
    def_args = get_default_args(desfun)
    extra_arg = [k for k in filt_args.keys() if k not in def_args.keys()]
    # should only be one extra key
    if len(extra_arg) > 1:
        raise ValueError('too many arguments for filter type '+destype)
    extra_arg = filt_args.pop( extra_arg.pop() )

    return desfun(extra_arg, **filt_args)

def filter_array(
        arr, ftype='butterworth', inplace=True,
        design_kwargs=dict(), filt_kwargs=dict()
        ):

        
    b, a = _get_poles_zeros(ftype, **design_kwargs)
    from sandbox.split_methods import bfilter
    def_args = get_default_args(bfilter)
    # reset these
    def_args['bsize'] = 10000
    def_args['filtfilt'] = True
    def_args.update(filt_kwargs)
    if inplace:
        bfilter(b, a, arr, **def_args)
        return arr
    else:
        # still use bfilter for memory efficiency
        arr_f = shared_ndarray(arr.shape)
        arr_f[:] = arr
        bfilter(b, a, arr_f, **def_args)
        return arr_f

def notch_all(
        arr, Fs, lines=60.0, nzo=3,
        nwid=3.0, inplace=True, nmax=-1, **filt_kwargs
        ):
    if not inplace:
        arr_f = shared_ndarray(arr.shape)
        arr_f[:] = arr
    else:
        arr_f = arr

    if isinstance(lines, float):
        # repeat lines until nmax
        nf = lines
        lines = [ nf*i for i in xrange(1, int(nmax//nf) + 1) ]

    notch_defs = get_default_args(notch)
    notch_defs['nwid'] = nwid
    notch_defs['nzo'] = nzo
    notch_defs['Fs'] = Fs
    for nf in lines:
        notch_defs['fcut'] = nf
        filter_array(
            arr_f, 'notch', inplace=True,
            design_kwargs=notch_defs, filt_kwargs=filt_kwargs
            )
    return arr_f

def downsample(x, fs, appx_fs=None, r=None, axis=-1):
    if appx_fs is None and r is None:
        return x
    if appx_fs is not None and r is not None:
        raise ValueError('only specify new fs or resample factor, not both')

    if appx_fs is not None:
        # new sampling interval must be a multiple of old sample interval,
        # so find the closest match that is >= appx_fs
        r = int( np.ceil(fs / appx_fs) )

    
    num_pts = x.shape[axis] // r
    num_pts += int( ( x.shape[axis] - num_pts*r ) > 0 )

    new_fs = fs / r

    fdesign = dict(ripple=0.5, hi=0.4*new_fs, Fs=fs)
    x_lp = filter_array(
        x, ftype='cheby1', inplace=False, 
        design_kwargs=fdesign, filt_kwargs=dict(axis=axis)
        )
    sl = [ slice(None) ] * len(x.shape)
    sl[axis] = slice(0, x.shape[axis], r)
    
    x_ds = x[ sl ].copy()
    return x_ds, new_fs
        
