# module to load pre-processed data from matlab in Mike's data struct form.

import numpy as np
import tables
import os
from contextlib import closing

from sandbox.array_split import shared_ndarray
from ecoglib.util import ChannelMap
from .h5utils import traverse_table

def load_preproc(f, load=True, sharedmem=True):
    shared_paths = ('/data',) if sharedmem else ()
    if load:
        with closing(tables.open_file(f)) as h5:
            pre = traverse_table(h5, load=True, shared_paths=shared_paths)
        # convert a few arrays
        for key in ('trig_coding', 'emap', 'egeo', 'orig_condition'):
            if key in pre and pre[key] is not None:
                arr = pre[key]
                pre[key] = arr.astype('i')
                if key == 'egeo':
                    pre[key] = tuple(pre[key])
        # transpose
        if pre.trig_coding is not None:
            pre.trig_coding = pre.trig_coding.T
            # convert indexing
            pre.trig_coding[0] -= 1
        if pre.emap is not None:
            pre.emap -= 1
            pre.chan_map = ChannelMap(pre.emap, pre.egeo, col_major=True)
    else:
        # this keeps the h5 file open?
        h5 = tables.open_file(f)
        pre = traverse_table(h5, load=False)
    return pre


        
            
