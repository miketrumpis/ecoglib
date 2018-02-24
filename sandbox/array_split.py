import platform
import ecoglib.mproc as mp
import ctypes
from contextlib import closing, contextmanager
import warnings
import gc
from decorator import decorator
import numpy as np

class ParaState(object):

    def __init__(self, init_state=True):
        self.__never_para = platform.system().lower().find('windows') >= 0
        self.state = init_state and not self.__never_para
        
    def __enable(self):
        if self.__never_para:
            raise RuntimeError('Object cannot be set to parallel status')
        self.state = True
        
    def __disable(self):
        self.state = False

    @contextmanager
    def context(self, status):
        prev_status = self.state
        self.state = status and not self.__never_para
        try:
            yield
        except:
            raise
        finally:
            self.state = prev_status

parallel_controller = ParaState()
        
# from the "array" module docstring
"""
This module defines an object type which can efficiently represent
an array of basic values: characters, integers, floating point
numbers.  Arrays are sequence types and behave very much like lists,
except that the type of objects stored in them is constrained.  The
type is specified at object creation time by using a type code, which
is a single character.  The following type codes are defined:

    Type code   C Type             Minimum size in bytes 
    'c'         character          1 
    'b'         signed integer     1 
    'B'         unsigned integer   1 
    'u'         Unicode character  2 
    'h'         signed integer     2 
    'H'         unsigned integer   2 
    'i'         signed integer     2 
    'I'         unsigned integer   2 
    'l'         signed integer     4 
    'L'         unsigned integer   4 
    'f'         floating point     4 
    'd'         floating point     8 

"""

# can only think of booleans
dtype_maps_to = dict([ ('?', 'b') ])

dtype_ctype = dict( (('F', 'f'), ('D', 'd'), ('G', 'g')) )
ctype_dtype = dict( ( (v, k) for k, v in dtype_ctype.items() ) )

def shared_ndarray(shape, typecode='d'):
    if not parallel_controller.state:
        return np.empty( shape, dtype=typecode )
    N = reduce(np.multiply, shape)
    if typecode in dtype_ctype:
        N *= 2
        ctypecode = dtype_ctype[typecode]
    else:
        ctypecode = typecode
    shm = mp.Array(ctypecode, int(N))
    return SharedmemManager.tonumpyarray(shm, shape=shape, dtype=typecode)

def shared_copy(x):
    # don't create wasteful copy if not doing parallel
    if not parallel_controller.state:
        return x
    typecode = dtype_maps_to.get(x.dtype.char, x.dtype.char)
    y = shared_ndarray(x.shape, typecode=typecode)
    y[:] = x.astype(typecode)
    return y

class SharedmemManager(object):

    def __init__(self, np_array, use_lock=False):
        self.dtype = np_array.dtype.char
        self.shape = np_array.shape
        if self.dtype in dtype_ctype:
            ctype_view = dtype_ctype[self.dtype]
            self.shm = mp.sharedctypes.synchronized(
                np.ctypeslib.as_ctypes(np_array.view(ctype_view))
                )
        else:
            self.shm = mp.sharedctypes.synchronized(
                np.ctypeslib.as_ctypes(np_array)
                )
        self.use_lock = use_lock

    def get_ndarray(self):
        if self.use_lock:
            with self.shm.get_lock():
                return SharedmemManager.tonumpyarray(
                    self.shm, dtype=self.dtype, shape=self.shape
                    )
        else:
            return SharedmemManager.tonumpyarray(
                self.shm, dtype=self.dtype, shape=self.shape
                )

    @staticmethod
    def tonumpyarray(mp_arr, dtype=float, shape=None):
        if shape is None:
            #global shape_
            shape = shape_

        ## info = mp.get_logger().info
        ## info('reshaping %s'%repr(shape))
        return np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)

def split_at(
        split_arg=(0,), splice_at=(0,), 
        shared_args=(), n_jobs=-1, concurrent=False
        ):
    # short circuit if the platform is Windows-based (look into doing
    # real multiproc later)
    if n_jobs==0 or platform.system().lower().find('windows') >= 0:
        @decorator
        def inner_split_method(method, *args, **kwargs):
            return method(*args, **kwargs)
        return inner_split_method
    
    # normalize inputs
    if not np.iterable(splice_at):
        splice_at = (splice_at,)
    if not np.iterable(split_arg):
        split_arg = (split_arg,)
    if n_jobs < 0:
        n_jobs = mp.cpu_count()

    splice_at = tuple(splice_at)
    split_arg = tuple(split_arg)
    shared_args = tuple(shared_args)
    @decorator
    def inner_split_method(method, *args, **kwargs):
        # make available short-cut to not use subprocesses:
        if not parallel_controller.state:
            return method(*args, **kwargs)
        pop_args = sorted( split_arg + shared_args )
        sh_args = list()
        n = 0
        args = list(args)
        shm = list()
        split_x = list()
        for pos in pop_args:
            pos = pos - n
            a = args.pop(pos)
            x = SharedmemManager( a, use_lock=concurrent )
            if pos+n in split_arg:
                shm.append( x )
                split_x.append( a )
            else:
                sh_args.append( x )
            n += 1
        static_args = tuple(args)
        sh_args = tuple(sh_args)

        split_lens = set([ len(sx) for sx in split_x ])
        if len(split_lens) > 1:
            raise ValueError(
                'all of the arrays to split must have the same length '
                'on the first axis'
                )
            
        # create a pool and map the shared memory array over the method
        init_args = (split_arg, shm, 
                     [getattr(x, 'shape') for x in split_x],
                     shared_args, sh_args,
                     method, static_args, kwargs)
        mp.freeze_support()
        with closing(mp.Pool(
                processes=n_jobs, initializer=_init_globals,
                initargs=init_args
                )) as p:
            n_div = estimate_chunks(split_x[0].size, len(p._pool))
            dim_size = split_x[0].shape[0]

            # if there are less or equal dims as procs, then split it up 1 per
            # otherwise, balance it with some procs having 
            # N=ceil(dims / procs) dims, and the rest having N-1

            max_dims = int(np.ceil(float(dim_size) / n_div))
            job_dims = [max_dims] * n_div
            n = -1
            # step back and subtract job size until sum matches total size
            while np.sum(job_dims) > dim_size:
                m = job_dims[n]
                job_dims[n] = m - 1
                n -= 1
            # filter out any proc with zero size
            job_dims = filter(None, job_dims)
            n = 0
            job_slices = list()
            # now form the data slicing to map out to the jobs
            for dims in job_dims:
                job_slices.extend( [slice(n, n+dims)] )
                n += dims
            # map the jobs
            res = p.map_async( _global_method_wrap, job_slices )

        p.join()
        if res.successful():
            res = splice_results(res.get(), splice_at)
            #res = res.get()
        else:
            # raises exception ?
            res.get()
        #gc.collect()
        return res

    return inner_split_method
    
def estimate_chunks(arr_size, nproc):
    # do nothing now
    return nproc

def splice_results(map_list, splice_at):
    if filter(lambda x: x is None, map_list):
        return
    if isinstance(map_list[0], np.ndarray):
        res = np.concatenate(map_list, axis=0)
        return res
    splice_at = sorted(splice_at)

    res = tuple()
    pres = 0
    res = tuple()
    for sres in splice_at:
        res = res + map_list[0][pres:sres]
        arr_list = [m[sres] for m in map_list]
        res = res + (np.concatenate(arr_list, axis=0),)
        pres = sres + 1
    res = res + map_list[0][pres:]
        
    return res

# --- the following are initialized in the global state of the subprocesses

class shared_readonly(object):
    def __init__(self, mem_mgr):
        self.mem_mgr = mem_mgr

    def __getitem__(self, idx):
        ## with self.mem_mgr.shm.get_lock():
        shm_ndarray = self.mem_mgr.get_ndarray() 
        return shm_ndarray[idx].copy()

def _init_globals(
        split_arg, shm, shm_shape,
        shared_args, sh_arg_mem,
        method, args, kwdict
        ):
    """
    split_arg : sequence of positions of argument that are split over
    shm : the shared memory managers for these split args
    shm_shape : the shapes of the underlying ndarrays of the split args
    shared_args : positions for readonly, non-split shared memory args
    sh_arg_mem : shared mem managers for any readonly shared memory arguments
    method : the method
    kwdict : any keyword args
    """
    
    # globals for primary shared array
    global shared_arr_
    shared_arr_ = shm
    global shape_
    shape_ = shm_shape
    global split_arg_
    split_arg_ = split_arg

    # globals for secondary shared memory arguments
    global shared_args_
    shared_args_ = shared_args
    global shared_args_mem_
    shared_args_mem_ = tuple( [ shared_readonly(mm) for mm in sh_arg_mem ] )

    # globals for pickled method and other arguments
    global method_
    method_ = method
    global args_
    args_ = args
    ## info = mp.get_logger().info
    ## info(repr(map(type, args)))

    global kwdict_
    kwdict_ = kwdict

    ## info = mp.get_logger().info
    ## info('applied global variables')
    
def _global_method_wrap(aslice):
    arrs = [arr_.get_ndarray() for arr_ in shared_arr_]
    
    ## info = mp.get_logger().info
    
    spliced_in = zip( 
        split_arg_+shared_args_, 
        [arr_[aslice] for arr_ in arrs] + list(shared_args_mem_)
        )
    spliced_in = sorted(spliced_in, key=lambda x: x[0])
    # assemble argument order correctly
    args = list()
    n = 0
    l_args = list(args_)
    while l_args:
        if spliced_in and spliced_in[0][0] == n:
            args.append(spliced_in[0][1])
            spliced_in.pop(0)
        else:
            args.append(l_args.pop(0))
        n += 1
    args.extend( [spl[1] for spl in spliced_in] )
    args = tuple(args)
    #info(repr(map(type, args)))

    ## info(
    ##     'applying method {0} to slice {1} at position {2}'.format(
    ##         method_, aslice, split_arg_
    ##         )
    ##     )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = method_(*args, **kwdict_)
    return r

