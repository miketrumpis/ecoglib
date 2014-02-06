import numpy as np
import ecoglib.util as ut

def _rev(n, coords):
    return [ n - c - 1 for c in coords ]

## PSV 244 Array
psv_244_daq1 = {
    'geometry' : (16, 16),
    'rows1.1' : [0, 1, 1, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 5, 
                 3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 6, 3, 1, 2, 0, 1],
    'cols1.1' : [13, 13, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 
                 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 7, 3, 3, 2, 2, 1],
    'rows1.2' : [-1, -1, 0, 2, 0, 2, 1, 3, 6, 4, 0, 2, 1, 3, 5, 4, 0, 2, 
                 1, 3, 5, 4, 0, 2, 1, 3, 7, 6, 0, 2, 1, -1],
    'cols1.2' : [-1, -1, 12, 12, 11, 11, 10, 10, 8, 9, 9, 9, 8, 8, 8, 
                 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 7, 6, 3, 3, 2, -1],
    ## Quadrant 2
    'rows2.1' : [2, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 
                 9, 10, 10, 10, 11, 11, 11, 8, 12, 12, 13, 13, 14],
    'cols2.1' : [0, 1, 1, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 5, 3, 1, 2, 
                 0, 4, 5, 3, 1, 2, 0, 4, 6, 3, 1, 2, 0, 1],
    'rows2.2' : [-1, -1, 3, 3, 4, 4, 5, 5, 7, 6, 6, 6, 7, 7, 7, 8, 8, 
                 8, 9, 9, 9, 10, 10, 10, 11, 11, 8, 9, 12, 12, 13, -1],
    'cols2.2' : [-1, -1, 0, 2, 0, 2, 1, 3, 6, 4, 0, 2, 1, 3, 5, 4, 0, 
                 2, 1, 3, 5, 4, 0, 2, 1, 3, 7, 6, 0, 2, 1, -1],
    ## Quadrant 3
    'rows3.1' : [15, 14, 14, 12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11, 10, 
                 12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11, 9, 12, 14, 13, 
                 15, 14],
    'cols3.1' : [2, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 
                 9, 10, 10, 10, 11, 11, 11, 8, 12, 12, 13, 13, 14],
    'rows3.2' : [-1, -1, 15, 13, 15, 13, 14, 12, 9, 11, 15, 13, 14, 12, 
                 10, 11, 15, 13, 14, 12, 10, 11, 15, 13, 14, 12, 8, 9, 15,
                 13, 14, -1],
    'cols3.2' : [-1, -1, 3, 3, 4, 4, 5, 5, 7, 6, 6, 6, 7, 7, 7, 8, 8, 
                 8, 9, 9, 9, 10, 10, 10, 11, 11, 8, 9, 12, 12, 13, -1],
    ## Quadrant 4
    'rows4.1' : [13, 13, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 
                 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 7, 3, 3, 2, 2, 1],
    'cols4.1' : [15, 14, 14, 12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11, 
                 10, 12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11, 9, 12, 14, 
                 13, 15, 14],
    'rows4.2' : [-1, -1, 12, 12, 11, 11, 10, 10, 8, 9, 9, 9, 8, 8, 8, 7, 
                 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 7, 6, 3, 3, 2, -1],
    'cols4.2' : [-1, -1, 15, 13, 15, 13, 14, 12, 9, 11, 15, 13, 14, 12, 
                 10, 11, 15, 13, 14, 12, 10, 11, 15, 13, 14, 12, 8, 9, 
                 15, 13, 14, -1]
    }

psv_32 = dict(
    geometry = (6, 6),
    cols = _rev(6, [5, 4, 5, 4, 1, 0, 1, 0, 3, 1, 4, 2, 2, 3, 2, 3, 5, 
                    4, 5, 4, 1, 0, 1, 0, 3, 1, 4, 2, 2, 3, 2, 3]),
    rows = [4, 2, 2, 0, 3, 5, 1, 3, 1, 4, 5, 0, 4, 3, 2, 5, 5, 
            3, 3, 1, 2, 4, 0, 2, 0, 5, 4, 1, 5, 2, 3, 4]
    )


electrode_maps = dict(psv_244_daq1=psv_244_daq1, psv_32=psv_32)

def get_electrode_map(name, connectors=()):
    if connectors:
        row_spec = ['rows'+cnx for cnx in connectors]
        col_spec = ['cols'+cnx for cnx in connectors]
    else:
        row_spec = ('rows',)
        col_spec = ('cols',)

    pinouts = electrode_maps[name]
        
    rows = list(); cols = list()
    for rkey, ckey in zip(row_spec, col_spec):
        rows.extend(pinouts[rkey])
        cols.extend(pinouts[ckey])

    rows = np.array(rows)
    cols = np.array(cols)
    connected_chans = rows >= 0
    disconnected = np.where(~connected_chans)[0]
    rows = rows[connected_chans]
    cols = cols[connected_chans]

    geometry = pinouts['geometry']
    flat_idx = ut.mat_to_flat(geometry, rows, cols, col_major=False)
    chan_map = ut.ChannelMap(flat_idx, geometry, col_major=False)
    return chan_map, disconnected

    
