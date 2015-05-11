# ye olde utilities module
import os
import copy
import errno
import numpy as np
import inspect
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

    def __copy__(self):
        d = dict([ (k, copy.copy(v)) for k, v in self.items() ])
        return Bunch(**d)
    
    def copy(self):
        return copy.copy(self)

    def __deepcopy__(self, memo):
        d = dict([ (k, copy.deepcopy(v)) for k, v in self.items() ])
        return Bunch(**d)
    
    def deepcopy(self):
        return copy.deepcopy(self)

def map_intersection(maps):
    geometries = set([m.geometry for m in maps])
    if len(geometries) > 1:
        raise ValueError('cannot intersect maps with different geometries')
    bin_map = maps[0].embed(np.ones(len(maps[0])), fill=0)
    for m in maps[1:]:
        bin_map *= m.embed(np.ones(len(m)), fill=0)
    return bin_map.astype('?')
    
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
        if isinstance(sub, type(self)):
            # check that it's a submap
            submap = map_intersection([self, sub])
            if submap.sum() < len(sub):
                raise ValueError(
                    'The given channel map is not a subset of this map'
                    )
            # get the channels/indices of the subset of sites
            sub = self.lookup(*submap.nonzero())
        elif isinstance(sub, np.ndarray) and sub.ndim==2:
            # get the channels/indices of the subset of sites
            sub = self.lookup(*sub.nonzero())            
            
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
        array = np.empty(shape, dtype=data.dtype)
        if not isinstance(fill, str):
            array.fill(fill)
        slicing = [slice(None)] * len(shape)
        slicing[axis] = self.as_row_major()[:]
        array[slicing] = data
        shape.pop(axis)
        shape.insert(axis, self.geometry[1])
        shape.insert(axis, self.geometry[0])
        array.shape = shape
        if isinstance(fill, str):
            return self.interpolated(array, axis=axis)
        return array

    def interpolated(self, image, axis=0, method='median'):
        # acts in-place
        mask = self.embed(np.zeros(len(self), dtype='?'), fill=1)
        missing = np.where( mask )
        g = self.geometry
        def _slice(i, j, w):
            before = [slice(None)] * axis
            after = [slice(None)] * (image.ndim - axis - 2)
            if w:
                isl = slice( max(0, i-w), min(g[0], i+w+1) )
                jsl = slice( max(0, j-w), min(g[1], j+w+1) )
            else:
                isl = i; jsl = j
            before.extend( [isl, jsl] )
            before.extend( after )
            return tuple(before)

        # first pass, tag all missing sites with nan
        for i, j in zip(*missing):
            image[ _slice(i, j, 0) ] = np.nan
        for i, j in zip(*missing):
            # do a +/- 2 neighborhoods (8 neighbors)
            patch = image[ _slice(i, j, 1) ].copy()
            s = list( patch.shape )
            s = s[:axis] + [ s[axis]*s[axis+1] ] + s[axis+2:]
            patch.shape = s
            fill = np.nanmedian( patch, axis=axis )
            image[ _slice(i, j, 0) ] = fill
        return image
            
            

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

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))


### From SO:
### http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def equalize_groups(x, group_sizes, axis=0, fill=np.nan, reshape=True):
    
    mx_size = max(group_sizes)
    n_groups = len(group_sizes)
    steps = np.r_[0, np.cumsum(group_sizes)]
    if x.shape[axis] != steps[-1]:
        raise ValueError('axis {0} in x has wrong size'.format(axis))
    new_shape = list(x.shape)
    new_shape[axis] = n_groups * mx_size
    if all( [g==mx_size for g in group_sizes] ):
        if reshape:
            new_shape[axis] = n_groups
            new_shape.insert(axis+1, mx_size)
            x = x.reshape(new_shape)
        return x
    y = np.empty(new_shape, dtype=x.dtype)
    new_shape[axis] = n_groups
    new_shape.insert(axis+1, mx_size)
    y = y.reshape(new_shape)
    y.fill(fill)
    y_slice = [slice(None)] * len(new_shape)
    x_slice = [slice(None)] * len(x.shape)
    for g in xrange(n_groups):
        y_slice[axis] = g
        y_slice[axis+1] = slice(0, group_sizes[g])
        x_slice[axis] = slice(steps[g], steps[g+1])
        y[y_slice] = x[x_slice]
    if not reshape:
        new_shape[axis] *= mx_size
        new_shape.pop(axis+1)
        y = y.reshape(new_shape)
    return y
