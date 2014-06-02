# ye olde utilities module
import numpy as np
import itertools
import scipy.misc as spmisc

# ye olde Bunch object
class Bunch(dict):
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self

    def __repr__(self):
        k_rep = self.keys()
        if not len(k_rep):
            return 'an empty Bunch'
        v_rep = [str(type(self[k])) for k in k_rep]
        mx_c1 = max([len(s) for s in k_rep])
        mx_c2 = max([len(s) for s in v_rep])
        
        table = [ '{0:<{col1}} : {1:<{col2}}\n'.format(
            k, v, col1=mx_c1, col2=mx_c2
            ) for (k, v) in zip(k_rep, v_rep) ]
        
        table = reduce(lambda x,y: x+y, table)
        return table.strip()

class ChannelMap(list):
    def __init__(self, chan_map, geo, col_major=True):
        list.__init__(self)
        self[:] = chan_map
        self.col_major = col_major
        self.geometry = geo

    def as_row_major(self):
        if self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:]),
                self.geometry, col_major=False
                )
        return self

    def as_col_major(self):
        if not self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:], col_major=False),
                self.geometry, col_major=True
                )
        return self

    def to_mat(self):
        return flat_to_mat(self.geometry, self, col_major=self.col_major)

    def lookup(self, i, j):
        flat_idx = mat_to_flat(self.geometry, i, j, col_major=self.col_major)
        if np.iterable(flat_idx):
            return np.array([self.index(fi) for fi in flat_idx])
        return self.index(flat_idx)

    def rlookup(self, c):
        return flat_to_mat(self.geometry, self[c], col_major=self.col_major)

    def subset(self, sub):
        return ChannelMap(
            [self[i] for i in sub],
            self.geometry, col_major=self.col_major
            )

    def __getslice__(self, i, j):
        return ChannelMap(
            super(ChannelMap, self).__getslice__(i,j),
            self.geometry, col_major=self.col_major
            )

    def embed(self, data, axis=0, fill=np.nan):
        """
        Embed the data in electrode array geometry, mapping channels
        on the given axis
        """
        shape = list(data.shape)
        if shape[axis] != len(self):
            raise ValueError(
                'Data array does not have the correct number of channels'
                )
        shape.pop(axis)
        shape.insert(axis, self.geometry[0]*self.geometry[1])
        array = np.empty(shape)
        array.fill(fill)
        slicing = [slice(None)] * len(shape)
        slicing[axis] = self.as_row_major()[:]
        array[slicing] = data
        shape.pop(axis)
        shape.insert(axis, self.geometry[1])
        shape.insert(axis, self.geometry[0])
        array.shape = shape
        return array
        

        
        
        

def flat_to_mat(mn, idx, col_major=True):
    idx = np.asarray(idx)
    # convert a flat matrix index into (i,j) style
    (m, n) = mn if col_major else mn[::-1]

    j = idx // m
    i = (idx - j*m)
    return (i, j) if col_major else (j, i)

def mat_to_flat(mn, i, j, col_major=True):
    (i, j) = map(np.asarray, (i, j))
    # covert matrix indexing to a flat (linear) indexing
    (fast, slow) = (i, j) if col_major else (j, i)
    block = mn[0] if col_major else mn[1]
    
    idx = slow*block + fast
    return idx

def flat_to_flat(mn, idx, col_major=True):
    # convert flat indexing from one convention to another
    i, j = flat_to_mat(mn, idx, col_major=col_major)
    return mat_to_flat(mn, i, j, col_major=not col_major)
    
def channel_combinations(chan_map, scale=1.0):
    combs = itertools.combinations(np.arange(len(chan_map)), 2)
    chan_combs = Bunch()
    npair = spmisc.comb(len(chan_map),2,exact=1)
    chan_combs.p1 = np.empty(npair, 'i')
    chan_combs.p2 = np.empty(npair, 'i')
    chan_combs.idx1 = np.empty((npair,2), 'i')
    chan_combs.idx2 = np.empty((npair,2), 'i')
    chan_combs.dist = np.empty(npair)
    ii, jj = chan_map.to_mat()
    for n, c in enumerate(combs):
        c0, c1 = c
        chan_combs.p1[n] = c0
        chan_combs.p2[n] = c1
        idx1 = np.array( [ii[c0], jj[c0]] )
        idx2 = np.array( [ii[c1], jj[c1]] )
        chan_combs.dist[n] = np.linalg.norm(idx1-idx2)*scale
        chan_combs.idx1[n] = idx1
        chan_combs.idx2[n] = idx2
    return chan_combs
