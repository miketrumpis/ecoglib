import multiprocessing as mp
import multiprocessing.sharedctypes
import ctypes
from contextlib import closing
import numpy as np

SPLIT_DISABLED = False

def shared_ndarray(shape):
    N = reduce(np.multiply, shape)
    shm = mp.Array(ctypes.c_double, N)
    return tonumpyarray(shm, shape)

def splits(method, split=0, off=False):
    # split not yet impelemented
    if off or SPLIT_DISABLED:
        return method
    def split_method(x, *args, **kwargs):
        shm = mp.sharedctypes.synchronized(
            np.ctypeslib.as_ctypes(x)
            )
        # create a pool and map the shared memory array over the method
        init_args = (shm, x.shape, method, args, kwargs)
        with closing(mp.Pool(
                processes=8, initializer=_init_globals,
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
            res = splice_results(res.get())
        else:
            # raises exception ?
            res.get()
        return res

    return split_method

def pow2(x):
    return x*x
    
def estimate_chunks(arr_size, nproc):
    # do nothing now
    return nproc

def splice_results(map_list):
    if filter(lambda x: x is None, map_list):
        return
    # for now only supporting single return in ndarray form
    return np.concatenate(map_list, axis=0)

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
    info = mp.get_logger().info
    info('applying method %s to slice %s'%(method_, aslice))
    #info('in norm: %f'%np.linalg.norm(np.ravel(arr[aslice])))
    #return method_(arr[aslice], *args_, **kwdict_)
    r = method_(arr[aslice], *args_, **kwdict_)
    #info('out norm 1: %f'%np.linalg.norm(np.ravel(arr[aslice])))
    #arr = tonumpyarray(shared_arr_)
    #info('out norm 2: %f'%np.linalg.norm(np.ravel(arr[aslice])))    
    return r

