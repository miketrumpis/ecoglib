import numpy as np

from ecoglib.filt.time import *
import ecoglib.util as ut

import multiprocessing as mp
import logging
logger = mp.log_to_stderr()
logger.setLevel(logging.CRITICAL)

import sandbox.array_split as array_split

@array_split.splits
def filtfilt(arr, b, a, bsize=10000):
    # needs to be axis=-1 for contiguity
    bfilter(b, a, arr, bsize=bsize, axis=-1)
