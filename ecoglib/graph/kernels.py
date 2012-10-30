import numpy as np

def gauss_affinity(d_sq, sig_sq):
    """
    The Gaussian affinity function is defined as

    a(i,j) = exp{-d(i,j)**2/(2*sig**2)}

    Parameters
    ----------

    d_sq: ndarray
      A flat array of all d(i,j)**2 values to compute
    sig_sq: float or array
      The characteristic distance scale, or possibly an adaptive scale
      for to each (i,j) pair, listed in the same order as d_sq
    """
    return np.exp(-d_sq/(2*sig_sq))
