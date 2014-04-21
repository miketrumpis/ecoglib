import multiprocessing as mp
import multiprocessing.sharedctypes
import ctypes
from contextlib import closing
import gc
from decorator import decorator
import numpy as np

def shared_ndarray(shape):
    N = reduce(np.multiply, shape)
    shm = mp.Array(ctypes.c_double, N)
    return tonumpyarray(shm, shape)

def split_at(split_arg=0, splice_at=(0,)):
    if not np.iterable(splice_at):
        splice_at = (splice_at,)
    @decorator
    def inner_split_method(method, *args, **kwargs):
        x = args[split_arg]
        kwargs['_split_arg_'] = split_arg
        static_args = args[:split_arg] + args[split_arg+1:]
        shm = mp.sharedctypes.synchronized(
            np.ctypeslib.as_ctypes(x)
            )
        # create a pool and map the shared memory array over the method
        init_args = (shm, x.shape, method, static_args, kwargs)
        with closing(mp.Pool(
                processes=mp.cpu_count(), initializer=_init_globals,
                initargs=init_args
                )) as p:
            n_div = estimate_chunks(x.size, len(p._pool))
            dim_size = x.shape[0]

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
            print job_slices
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

def tonumpyarray(mp_arr, shape=None):
    if shape is None:
        #global shape_
        shape = shape_
    
    info = mp.get_logger().info
    info('reshaping %s'%repr(shape))
    return np.frombuffer(mp_arr.get_obj()).reshape(shape)

def _init_globals(shm, shm_shape, method, args, kwdict):
    global shared_arr_
    shared_arr_ = shm
    global shape_
    shape_ = shm_shape
    global method_
    method_ = method
    global args_
    args_ = args
    global kwdict_
    kwdict_ = kwdict
    info = mp.get_logger().info
    info('applied global variables')
    
def _global_method_wrap(aslice):
    arr = tonumpyarray(shared_arr_)
    split_arg = kwdict_.pop('_split_arg_')
    info = mp.get_logger().info
    info('applying method %s to slice %s at position %d'%(method_, aslice, split_arg))
    # assemble argument order correctly
    args = args_[:split_arg] + (arr[aslice],) + args_[split_arg:]
    info(repr(map(type, args)))
    r = method_(*args, **kwdict_)
    return r

