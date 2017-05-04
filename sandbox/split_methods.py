import numpy as np
import scipy.ndimage as ndimage
import ecoglib.filt.time.blocked_filter as bf
import ecoglib.util as ut

import sandbox.array_split as array_split

import nitime.algorithms as ntalg

### Parallelized re-definitions
bfilter = array_split.split_at(split_arg=2)(bf.bfilter)
overlap_add = array_split.split_at()(bf.overlap_add)

multi_taper_psd = array_split.split_at(splice_at=(1,2))(
    ntalg.multi_taper_psd
    )

convolve1d = array_split.split_at(split_arg=0)(ndimage.convolve1d)

### Convenience wrappers
def filtfilt(arr, b, a, bsize=10000):
    """
    Docstring
    """
    # needs to be axis=-1 for contiguity
    bfilter(b, a, arr, bsize=bsize, axis=-1, filtfilt=True)
