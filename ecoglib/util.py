# ye olde utilities module
import numpy as np

# ye olde Bunch object
class Bunch(dict):
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self

    def __repr__(self):
        k_rep = self.keys()
        v_rep = [str(type(self[k])) for k in k_rep]
        mx_c1 = max([len(s) for s in k_rep])
        mx_c2 = max([len(s) for s in v_rep])
        
        table = [ '{0:<{col1}} : {1:<{col2}}\n'.format(
            k, v, col1=mx_c1, col2=mx_c2
            ) for (k, v) in zip(k_rep, v_rep) ]
        
        table = reduce(lambda x,y: x+y, table)
        return table.strip()

def flat_to_mat(mn, idx, col_major=True):
    # convert a flat matrix index into (i,j) style
    (m, n) = mn if col_major else mn[::-1]

    j = idx // m
    i = (idx - j*m)
    return (i, j) if col_major else (j, i)

def mat_to_flat(mn, i, j, col_major=True):
    # covert matrix indexing to a flat (linear) indexing
    (fast, slow) = (i, j) if col_major else (j, i)
    block = mn[0] if col_major else mn[1]
    
    idx = slow*block + fast
    return idx

def flat_to_flat(mn, idx, col_major=True):
    # convert flat indexing from one convention to another
    i, j = flat_to_mat(mn, idx, col_major=col_major)
    return mat_to_flat(mn, i, j, col_major=not col_major)
    
def nextpow2(n):
    pow = int( np.floor( np.log2(n) ) + 1 )
    return 2**pow

def ndim_prctile(x, p, axis=0):
    xs = np.sort(x, axis=axis)
    dim = xs.shape[axis]
    idx = np.round( float(dim) * np.asarray(p) / 100 ).astype('i')
    slicer = [slice(None)] * x.ndim
    slicer[axis] = idx
    return xs[slicer]
        
