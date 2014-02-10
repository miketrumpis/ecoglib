from glob import glob
import os.path as p
import sys
import tables
import numpy as np
import scipy.io as sio
import scipy.signal as signal

import sandbox.array_split as array_split

import sandbox.expo as expo
import sandbox.electrode_pinouts as epins
from ecoglib.filt.time import *
import ecoglib.util as ut

import multiprocessing as mp
import logging
logger = mp.log_to_stderr()
logger.setLevel(logging.CRITICAL)

@array_split.splits
def filtfilt(arr, b, a, bsize=10000):
    # needs to be axis=-1 for contiguity
    bfilter(b, a, arr, bsize=bsize, axis=-1)

root_pth = '/Users/mike/experiment_data/AnimalExperiment-2013-11-01/blackrock'
bkp_pth = '/Users/mike/experiment_data/AnimalExperiment-2013-11-01/blackrock_offline'
root_exp = 'm645r4#'
xml_pth = '/Users/mike/experiment_data/AnimalExperiment-2013-11-01/expo_xml'
xml_exp = 'm645r3#'

def quick_load_br(test, notches=(), downsamp=15, page_size=10):

    ## Get array-to-channel pinouts

    if test > 2 and test < 13:
        connections = ('1.1', '1.2', '4.1')
    elif test > 12 and test < 25:
        connections = ('3.1', '3.2', '4.2')
    else:
        connections = ('3.1', '1.1', '4.1')

    chan_map, disconnected = epins.get_electrode_map(
        'psv_244_daq1', connectors=connections
        )

    ## Default Lowpass Filter
    (ord, wn) = signal.cheb2ord(2*700/3e4, 2*1000/3e4, 0.1, 60)
    (b, a) = cheby2_bp(60, hi=wn, Fs=2, ord=ord)

    try:
        test_file = p.join(root_pth, root_exp) + '%03d.h5'%test
        h5f = tables.open_file(test_file)
    except IOError:
        test_file = p.join(bkp_pth, root_exp) + '%03d.h5'%test
        h5f = tables.open_file(test_file)

    dlen, nchan = h5f.root.data.shape
    sublen = dlen / downsamp
    if dlen - sublen*downsamp > 0:
        sublen += 1

    #subdata = np.empty((len(chan_map), sublen), 'd')
    subdata = array_split.shared_ndarray((len(chan_map), sublen))
    if len(chan_map) < nchan:
        gndchan = np.empty((len(disconnected), sublen), 'd')
    else:
        gndchan = None
    peel = array_split.shared_ndarray( (page_size, h5f.root.data.shape[0]) )
    n = 0
    dchan_corr = 0
    while n < nchan:
        start = n
        stop = min(nchan, n+page_size)
        print 'processing BR channels %03d - %03d'%(start, stop-1)
        peel[0:stop-n] = h5f.root.data[:,start:stop].T.astype('d', order='C')
        peel *= 8e-3 / 2**15
        print 'parfilt',
        sys.stdout.flush()
        filtfilt(peel[0:stop-n], b, a)
        print 'done'
        sys.stdout.flush()
        data_chans = np.setdiff1d(np.arange(start,stop), disconnected)
        if len(data_chans):
            dstart = data_chans[0] - dchan_corr
            dstop = dstart + len(data_chans)
        if len(data_chans) == (stop-start):
            # if all data channels, peel off in a straightforward way
            subdata[dstart:dstop,:] = peel[0:stop-n,::downsamp]
        else:
            if len(data_chans):
                # get data channels first
                raw_data = peel[data_chans-n, :]
                subdata[dstart:dstop, :] = raw_data[:, ::downsamp]
            # Now filter for ground channels within this set of channels:
            gnd_chans = filter(
                lambda x: x[0]>=start and x[0]<stop, 
                zip(disconnected, range(len(disconnected)))
                )
            for g in gnd_chans:
                gndchan[g[1], :] = peel[g[0]-n, ::downsamp]
            dchan_corr += len(gnd_chans)
        n += page_size

    Fs = h5f.root.Fs.read()[0,0] / downsamp
    trigs = h5f.root.trig_idx.read().squeeze()
    if not trigs.shape:
        trigs = None
    else:
        trigs = np.round( trigs / downsamp ).astype('i')
    h5f.close()

    ## Set up Expo Experiment    
    xml = glob(p.join(xml_pth, xml_exp) + '%d*'%test)
    if not xml:
        print 'Did not find EXPO XML file, generating generic experiment'
        exp = expo.StimulatedExperiment(trigs)
    else:
        exp = expo.get_expo_experiment(xml[0], trigs)

    # seems to be more numerically stable to do highpass and 
    # notch filtering after downsampling
    (b, a) = butter_bp(lo=2, Fs=Fs, ord=4)
    filtfilt(subdata, b, a)
    for nfreq in notches:
        (b, a) = notch(nfreq, Fs=Fs, ftype='cheby2')
        filtfilt(subdata, b, a)
    
    dset = ut.Bunch(
        data=subdata, ground_chans=gndchan, 
        exptab=exp, Fs=Fs, notches=notches,
        chan_map=chan_map
        )
    return dset
    
