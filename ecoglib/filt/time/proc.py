"""
One-stop shopping for digital filtering of arrays
"""

import numpy as np
from .design import butter_bp, cheby1_bp, cheby2_bp, notch
from ecoglib.util import get_default_args
from sandbox.split_methods import bfilter
from sandbox.array_split import shared_ndarray

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
        arr, Fs, lines=60.0,
        nwid=3.0, inplace=True, nmax=-1
        ):
    if not inplace:
        arr_f = shared_ndarray(arr.shape)
        arr_f[:] = arr
    else:
        arr_f = arr

    if isinstance(lines, float):
        # repeat lines until nmax
        nf = lines
        lines = [ nf*i for i in xrange(1, nmax//nf + 1) ]
        
    for nf in lines:
        filter_array(
            arr_f, 'notch', inplace=True,
            design_kwargs=dict(fcut=nf, nwid=nwid, Fs=Fs)
            )
    return arr_f
