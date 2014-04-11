import numpy as np
import ecoglib.util as ut

def array_geometry(data, map, axis=-1):
    # re-arange the 2D matrix of timeseries data such that the
    # channel axis is ordered by the array geometry
        
    map = map.as_row_major()
    geo = map.geometry
    
    dims = list(data.shape)
    while axis < 0:
        axis += len(dims)
    chan_dim = dims.pop(axis)
    ts_dim = dims[0]
    
    new_data = np.zeros( geo + (ts_dim,), data.dtype )
    new_data.shape = (-1, ts_dim)
    new_data[map,:] = data if axis==0 else data.T

    null_sites = set(range(geo[0]*geo[1]))
    null_sites.difference_update(map)
    
    return new_data, ut.ChannelMap(list(null_sites), geo, col_major=False)
    
    
    
