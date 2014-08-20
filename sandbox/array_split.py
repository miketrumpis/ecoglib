import multiprocessing as mp
import multiprocessing.sharedctypes
import ctypes
from contextlib import closing
import warnings
import gc
from decorator import decorator
import numpy as np

def shared_ndarray(shape):
    N = reduce(np.multiply, shape)
    shm = mp.Array(ctypes.c_double, N)
    return tonumpyarray(shm, shape=shape)

dtype_ctype = dict( (('F', 'f'), ('D', 'd'), ('G', 'g')) )
ctype_dtype = dict( ( (v, k) for k, v in dtype_ctype.items() ) )

class SharedmemManager(object):

    def __init__(self, np_array):
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
    
    def get_ndarray(self):
        return tonumpyarray(
            self.shm, dtype=self.dtype, shape=self.shape
            )

def split_at(split_arg=0, splice_at=(0,), shared_args=(), n_jobs=-1, concurrent=False):
    if not np.iterable(splice_at):
        splice_at = (splice_at,)
    if n_jobs < 0:
        n_jobs = mp.cpu_count()
    @decorator
    def inner_split_method(method, *args, **kwargs):
        pop_args = sorted( (split_arg,) + shared_args )
        sh_args = list()
        n = 0
        args = list(args)
        for pos in pop_args:
            pos = pos - n
            a = args.pop(pos)
            x = SharedmemManager( a )
            if pos+n == split_arg:
                shm = x
                split_x = a
            else:
                sh_args.append( x )
            n += 1
        static_args = tuple(args)
        sh_args = tuple(sh_args)
            
        # create a pool and map the shared memory array over the method
        init_args = (split_arg, shm, split_x.shape,
                     shared_args, sh_args,
                     method, static_args, kwargs)
        mp.freeze_support()
        with closing(mp.Pool(
                processes=n_jobs, initializer=_init_globals,
                initargs=init_args
                )) as p:
            n_div = estimate_chunks(split_x.size, len(p._pool))
            dim_size = split_x.shape[0]

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
            if concurrent:
                res = p.map_async( _global_method_acquire, job_slices )
            else:
                res = p.map_async( _global_method_wrap, job_slices )

        p.join()
        if res.successful():
            res = splice_results(res.get(), splice_at)
            #res = res.get()
        else:
            # raises exception ?
            res.get()
        gc.collect()
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

def tonumpyarray(mp_arr, dtype=float, shape=None):
    if shape is None:
        #global shape_
        shape = shape_
    
    info = mp.get_logger().info
    info('reshaping %s'%repr(shape))
    return np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)

def _init_globals(
        split_arg, shm, shm_shape,
        shared_args, sh_arg_mem,
        method, args, kwdict
        ):
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
    info = mp.get_logger().info
    info(repr(map(type, args)))

    global kwdict_
    kwdict_ = kwdict

    info = mp.get_logger().info
    info('applied global variables')

def _global_method_acquire(aslice):
    with shared_arr_.shm.get_lock():
        return _global_method_wrap(aslice)
    
def _global_method_wrap(aslice):
    arr = shared_arr_.get_ndarray()
    
    info = mp.get_logger().info
    
    spliced_in = zip( 
        (split_arg_,)+shared_args_, (arr[aslice],) + shared_args_mem_
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

    info('applying method %s to slice %s at position %d'%(method_, aslice, split_arg_))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = method_(*args, **kwdict_)
    return r

