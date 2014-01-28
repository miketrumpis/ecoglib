# module to load pre-processed data from matlab in Mike's data struct form.

import numpy as np
import tables
import os

from ecoglib.util import Bunch

def load_preproc(f, load=True):
    if load:
        with tables.open_file(f) as h5:
            pre = traverse_table(h5, load=True)
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
    else:
        # this keeps the h5 file open?
        h5 = tables.open_file(f)
        pre = traverse_table(h5, load=False)
    return pre

def traverse_table(f, path='/', load=True):
    # Walk nodes and stuff arrays into the bunch.
    # If we encouter a group, then loop back into this method
    if isinstance(f, str):
        f = tables.open_file(f)
    gbunch = Bunch()
    (p, g) = os.path.split(path)
    if g=='':
        g = p
    nlist = f.list_nodes(path)
    #for n in f.walk_nodes(where=path):
    for n in nlist:
        if isinstance(n, tables.Array):
            if load:
                arr = n.read()
                if arr.shape == (1,1):
                    arr = arr[0,0]
                    if arr==0:
                        arr = None
                else:
                    arr = arr.squeeze()
            else:
                arr = n
            gbunch[n.name] = arr
        elif isinstance(n, tables.Group):
            gname = n._v_name
            # walk_nodes() includes the current group:
            # don't try to descend into this node!
            if gname==g:
                continue
            subbunch = traverse_table(
                f, path=os.path.join(path, gname)
                )
            ## # get any attributes if they're around
            ## for attr in n._v_attrs._f_list():
            ##     subbunch[attr] = n._v_attrs[attr]
                
            gbunch[gname] = subbunch
            
        else:
            gbunch[n.name] = 'Not Loaded!'

    this_node = f.get_node(path)
    for attr in this_node._v_attrs._f_list():
        gbunch[attr] = this_node._v_attrs[attr]
            
    return gbunch
            


