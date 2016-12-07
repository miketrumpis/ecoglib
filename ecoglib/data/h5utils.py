"""A module for file I/O using HDF5 and "Bunch" collections.
"""

import numpy as np
import tables
from tables import NoSuchNodeError
import os
from contextlib import closing

from ecoglib.util import Bunch
from sandbox.array_split import shared_ndarray

_h5_seq_types = (str, list, int, float, complex, bool)

def save_bunch(f, path, b, mode='a', overwrite_paths=False, compress_arrays=0):
    """
    Save a Bunch type to an HDF5 group in a new or existing table.

    Parameters
    ---------

    f : path or open tables file
    path : path within the tables file, where the Bunch will be saved
    b : the Bunch itself
    mode : file mode (see tables.open_file)
    compress_arrays : if >0, then ndarrays will be compressed at this level

    Arrays, strings, lists, and various scalar types are saved as
    naturally supported array types. Sub-Bunches are written
    recursively in sub-paths. The remaining Bunch elements are
    pickled, preserving their object classification.

    """
    
    # * create a new group
    # * save any array-like type natively (esp ndarrays)
    # * save everything else as the pickled ObjectAtom 
    # * if there are any sub-bunches, then re-enter method with subgroup
    
    if not isinstance(f, tables.file.File):
        with closing(tables.open_file(f, mode)) as f:
            return save_bunch(
                f, path, b, 
                overwrite_paths=overwrite_paths,
                compress_arrays=compress_arrays
                )

    # If we want to overwrite a node, check to see that it exists.
    # If we want an exception when trying to overwrite, that will
    # be caught on f.create_group()
    if overwrite_paths:
        try:
            n = f.get_node(path)
            n._f_remove(recursive=True, force=True)
        except NoSuchNodeError:
            pass
    p, node = os.path.split(path)
    if node:
        f.create_group(p, node, createparents=True)

    sub_bunches = list()
    items = b.iteritems()
    pickle_bunch = Bunch()

    # 1) create arrays for suitable types
    for key, val in items:
        if isinstance(val, np.ndarray) and len(val):
            atom = tables.Atom.from_dtype(val.dtype)
            if compress_arrays:
                filters = tables.Filters(
                    complevel=compress_arrays, complib='zlib'
                    )
            else:
                filters = None
            ca = f.create_carray(
                path, key, atom=atom, shape=val.shape, filters=filters
                )
            ca[:] = val
            

        elif type(val) in _h5_seq_types:
            try:
                f.create_array(path, key, val)
            except TypeError, ValueError:
                pickle_bunch[key] = val

        elif isinstance(val, Bunch):
            sub_bunches.append( (key, val) )
        else:
            pickle_bunch[key] = val

    # 2) pickle the remaining items (that are not bunches)
    p_arr = f.create_vlarray(path, 'b_pickle', atom=tables.ObjectAtom())
    p_arr.append(pickle_bunch)

    # 3) repeat these steps for any bunch elements that are also bunches
    for n, b in sub_bunches:
        #print 'saving', n, b
        save_bunch(
            f, '/'.join([path, n]), b, compress_arrays=compress_arrays
            )
    return

def load_bunch(f, path, shared_arrays=(), load=True):
    """
    Load a saved bunch, or an arbitrary collection of arrays into a
    new Bunch object.
    
    Parameters
    ---------

    f : path or open tables file
    path : path within the tables file to load
    
    """

    shared_arrays = map(lambda a: '/'.join([path, a]), shared_arrays)
    return traverse_table(f, path=path, shared_paths=shared_arrays, load=load)

def traverse_table(f, path='/', load=True, shared_paths=()):
    # Walk nodes and stuff arrays into the bunch.
    # If we encouter a group, then loop back into this method
    if not isinstance(f, tables.file.File):
        with closing(tables.open_file(f)) as f:
            return traverse_table(
                f, path=path, load=load, shared_paths=shared_paths
                )
    gbunch = Bunch()
    (p, g) = os.path.split(path)
    if g=='':
        g = p
    nlist = f.list_nodes(path)
    #for n in f.walk_nodes(where=path):
    for n in nlist:
        if isinstance(n, tables.Array):
            if load:
                if '/'.join([path, n.name]) in shared_paths:
                    arr = shared_ndarray(n.shape)
                    arr[:] = n.read()
                else:
                    arr = n.read()
                if isinstance(arr, np.ndarray) and n.shape:
                    if arr.shape == (1,1):
                        arr = arr[0,0]
                        if arr==0:
                            arr = None
                    else:
                        arr = arr.squeeze()
            else:
                arr = n
            gbunch[n.name] = arr
        elif isinstance(n, tables.VLArray):
            if load:
                obj = n.read()[0]
                # if it's a generic Bunch Pickle, then update the bunch
                if n.name == 'b_pickle':
                    gbunch.update(obj)
                else:
                    gbunch[n.name] = obj
            else:
                # ignore the empty pickle
                if n.name == 'b_pickle' and n.size_in_memory > 32L:
                    gbunch[n.name] = 'unloaded pickle'
        elif isinstance(n, tables.Group):
            gname = n._v_name
            # walk_nodes() includes the current group:
            # don't try to descend into this node!
            if gname==g:
                continue
            subbunch = traverse_table(
                f, path='/'.join([path, gname]), load=load
                )
            gbunch[gname] = subbunch
            
        else:
            gbunch[n.name] = 'Not Loaded!'

    this_node = f.get_node(path)
    for attr in this_node._v_attrs._f_list():
        gbunch[attr] = this_node._v_attrs[attr]
            
    return gbunch


