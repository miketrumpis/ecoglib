# ye olde utilities module
import os
import copy
import errno
from glob import glob
import numpy as np
import inspect
import importlib
from weakref import WeakKeyDictionary

# ye olde Bunch object
class Bunch(dict):
    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)
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

### Matrix-indexing manipulations
def flat_to_mat(mn, idx, col_major=True):
    idx = np.asarray(idx)
    # convert a flat matrix index into (i,j) style
    (m, n) = mn if col_major else mn[::-1]
    if (idx < 0).any() or (idx >= m*n).any():
        raise ValueError(
            'The flat index does not lie inside the matrix: '+str(mn)
            )
    j = idx // m
    i = (idx - j*m)
    return (i, j) if col_major else (j, i)

def mat_to_flat(mn, i, j, col_major=True):
    i, j = map(np.asarray, (i, j))
    if (i < 0).any() or (i >= mn[0]).any() \
      or (j < 0).any() or (j >= mn[1]).any():
        raise ValueError('The matrix index does not fit the geometry: '+str(mn))
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

### Introspection
def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))

### Path trick from SO:
### http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

### Reshape into categories with possible padding
def equalize_groups(x, group_sizes, axis=0, fill=np.nan, reshape=True):
    
    mx_size = max(group_sizes)
    n_groups = len(group_sizes)
    steps = np.r_[0, np.cumsum(group_sizes)]
    new_shape = list(x.shape)
    new_shape[axis] = n_groups * mx_size
    if np.prod(x.shape) == np.prod(new_shape):
        # already has consistent size for equalized groups
        if reshape:
            new_shape[axis] = n_groups
            new_shape.insert(axis+1, mx_size)
            return x.reshape(new_shape)
        return x
    if x.shape[axis] != steps[-1]:
        raise ValueError('axis {0} in x has wrong size'.format(axis))
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

def search_results(path, filter=''):
    from ecoglib.data.h5utils import load_bunch
    filter = filter + '*.h5'
    existing = sorted(glob(os.path.join(path, filter)))
    if existing:
        print 'Precomputed results exist:'
        for n, path in enumerate(existing):
            print '\t(%d)\t%s'%(n,path)
        mode = raw_input(
            'Enter a choice to load existing work,'\
            'or (c) to compute new results: '
            )
        try:
            return load_bunch(existing[int(mode)], '/')
        except ValueError:
            return Bunch()

from decorator import decorator
def input_as_2d(in_arr=0, out_arr=-1):
    """
    A decorator to reshape input to be 2D and then bring output back 
    to original size (possibly with loss of last dimension). 
    Vectors will also be reshaped to (1, N) on input.

    Parameters
    ----------
    in_arr : int (sequence)
        position of argument(s) (input) to be reshaped
    out_arr : int
        Non-negative position of output to be reshaped. 
        If None, then no output is reshaped. If the method's
        return type is not a tuple, this argument has no effect.

    """

    if not np.iterable(in_arr):
        in_arr = (in_arr,)
    
    @decorator
    def _wrap(fn, *args, **kwargs):
        args = list(args)
        shapes = list()
        for p in in_arr:
            x = args[p]
            shp = x.shape
            x = x.reshape(-1, shp[-1])
            args[p] = x
            shapes.append(shp)
        r = fn(*args, **kwargs)
        if out_arr is None:
            return r
        if isinstance(r, tuple) and out_arr >= 0:
            x = r[out_arr]
        else:
            x = r
        # now relying on the 1st encountered shape to
        # represent the output shape
        shp = shapes[0]
        n_out = len(x.shape)
        # check to see if the function ate the last dimension
        if n_out < 2:
            shp = shp[:-1]
        elif shp[-1] != x.shape[-1]:
            shp = shp[:-1] + (x.shape[-1],)
        x = x.reshape(shp)
        if isinstance(r, tuple) and out_arr >= 0:
            return r[:out_arr] + (x,) + r[out_arr+1:]
        else:
            return x
    return _wrap

class RelocatedImport(object):
    def __init__(self, thing, new_module):
        self.new_module = new_module
        self.thing = thing
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        print str(self.thing) + ' should now be imported from ' \
          + self.new_module
        pkg = importlib.import_module(self.new_module)
        try:
            obj = getattr(pkg, self.thing)
        except AttributeError:
            return None
        return self.data.get(instance, obj)

class _MovedImports(object):
    #from .channel_map import ChannelMap, mat_to_flat, flat_to_mat, flat_to_flat
    ChannelMap = RelocatedImport('ChannelMap', 'ecoglib.channel_map')
    map_intersection = RelocatedImport(
        'map_intersection', 'ecoglib.channel_map')
    channel_combinations = RelocatedImport(
        'channel_combinations', 'ecoglib.channel_map')

_MV = _MovedImports()    
ChannelMap = _MV.ChannelMap
map_intersection = _MV.map_intersection
channel_combinations = _MV.channel_combinations
