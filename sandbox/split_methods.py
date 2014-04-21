import numpy as np

from ecoglib.filt.time import *
import ecoglib.util as ut

import multiprocessing as mp
import logging
logger = mp.log_to_stderr()
logger.setLevel(logging.CRITICAL)

import sandbox.array_split as array_split

import nitime.algorithms as ntalg

### Parallelized re-definitions
bfilter = array_split.split_at(split_arg=2)(bfilter)
multi_taper_psd = array_split.split_at(splice_at=(1,2))(ntalg.multi_taper_psd)


### Convenience wrappers
@array_split.split_at()
def filtfilt(arr, b, a, bsize=10000):
    """
    Docstring
    """
    # needs to be axis=-1 for contiguity
    bfilter(b, a, arr, bsize=bsize, axis=-1, filtfilt=True)
