
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage

def constq_wavelet(f_start, Fs, Q=1, fpo=2, eps=1e-3):
    # the freq domain bandwidth should be
    # BW = f / Q
    # a(w) = (2*np.pi*BW/2)**2 / log(2) / 4
    #
    # 

    # do semi-dyadically spaced center freqs, default of 2 filters per octave
    f_stop = int( np.floor(np.log2(Fs / 4.) * fpo) )
    f_start = int( np.floor(np.log2(f_start) * fpo) )
    w0 = np.power(2.0, np.arange(f_start, f_stop+1, dtype='d')/fpo)

    w0 /= Fs
    
    BW = w0 / Q

    sigma = (2*np.pi*BW/2)**2 / 4. / np.log(2)
    # roll sampling rate into this term
    #sigma /= Fs**2
    #print sigma
    # find maximum sequence length to ensure that w(t;f) > eps
    N = np.floor(np.sqrt( np.log(1/eps) / sigma[0] ))//2
    atoms = np.zeros( (len(sigma), 2*N+1), dtype='D' )

    xx = np.linspace(-N, N, 2*N+1)
    for a, s, w in zip(atoms, sigma, w0):
        a[:] = np.exp(1j * 2*np.pi * w * xx) * np.exp(-s * xx**2)

    bias = np.exp(-0.5 * w0**2)
    #atoms -= bias[:,None]
    return atoms, w0 * Fs
        
def filterbank(x, *cq_args, **cq_kwargs):
    atoms, freqs = constq_wavelet(*cq_args, **cq_kwargs)

    y = np.empty( (x.shape[0], atoms.shape[0], x.shape[1]), atoms.dtype )
    for n, a in enumerate(atoms):
        ndimage.convolve1d(
            x, a.real, output=y[:,n,:].real, axis=-1, mode='reflect'
            )
        ndimage.convolve1d(
            x, a.imag, output=y[:,n,:].imag, axis=-1, mode='reflect'
            )

    return y, freqs

def freq_boundaries(freqs):
    # return the bin edges for freqs in log-constant steps
    lfreqs = np.log2(freqs)
    df = lfreqs[1] - lfreqs[0]

    bfreqs = np.r_[ lfreqs - df/2.0, lfreqs[-1] + df/2.0 ]
    return np.power(2.0, bfreqs)
