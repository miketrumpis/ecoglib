import numpy as np
import ecoglib.util as ut

def array_geometry(data, geo, map, axis=-1, col_major=True):
    # re-arange the 2D matrix of timeseries data such that the
    # channel axis is ordered by the array geometry
    
    
    ## i, j = ut.flat_to_mat(geo, map, col_major=True)
    ## row_maj_conv = ut.mat_to_flat(geo, i, j, col_major=False)
    if col_major:
        map = ut.flat_to_flat(geo, map, col_major=True)
    
    
    dims = list(data.shape)
    while axis < 0:
        axis += len(dims)
    chan_dim = dims.pop(axis)
    ts_dim = dims[0]
    
    new_data = np.zeros( geo + (ts_dim,), data.dtype )
    new_data.shape = (-1, ts_dim)
    new_data[map,:] = data if axis==0 else data.T

    if col_major:
        map = ut.flat_to_flat(geo, map, col_major=False)
    null_sites = set(range(geo[0]*geo[1]))
    null_sites.difference_update(map)
    
    return new_data, np.array(list(null_sites))
    
    
    
