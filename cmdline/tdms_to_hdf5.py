#!/usr/bin/env python
import numpy as np
import tables
import nptdms
import tempfile
import os, sys
from glob import glob
import contextlib

def tdms_to_hdf5(
        tdms_file, h5_file, chan_map='',
        memmap=True, compression_level=0
        ):
    """
    Converts TDMS data to a more standard HDF5 format.

    Parameters
    ----------

    tdms_file : path (string)
    h5_file : path (string)
    chan_map : path (string)
      Optional table specifying a channel permutation. The first p rows
      of the outgoing H5 file will be the contents of these channels in
      sequence. The next (N-p) rows will be any channels not specified,
      in the order they are found.
    memmap : bool
    compression_level : int
      Optionally compress the outgoing H5 rows with zlib compression.
      This can reduce the time cost caused by disk access.
    """

    map_dir = tempfile.gettempdir() if memmap else None

    # put these operations in a context so that the temp files
    # get killed whether or not things go well inside
    with contextlib.nested(nptdms.TdmsFile(tdms_file, memmap_dir=map_dir),
                           tables.open_file(h5_file, mode='w')) as \
                           (tdms_file, h5_file):

        # assume for now there is only a single group -- see more files later
        t_group = tdms_file.groups()[0]

        g_obj = tdms_file.object(t_group)
        chans = tdms_file.group_channels(t_group)

        n_col = len(chans)
        n_row = chans[0].number_values

        # The H5 file will be constructed as follows:
        #  * create a Group for the info section
        #  * create a CArray with zlib(3) compression for the data channels
        #  * create separate Arrays for special values
        #    (SampRate[SamplingRate], numRow[nrRows], numCol[nrColumns],
        #     OSR[OverSampling], numChan[nrColumns+nrBNCs])
        special_conversion = dict(
            SamplingRate='sampRate', nrRows='numRow', nrColumns='numCol',
            OverSampling='OSR'
            )
        h5_info = h5_file.create_group(h5_file.root, 'info')
        for (key, val) in g_obj.properties.items():
            if type(val) == unicode:
                # HDF5 doesn't support unicode
                try:
                    val = str(val)
                except:
                    print '**** Cannot convert this value:'
                    print val
                    continue
            val = np.array([val])
            h5_file.create_array(h5_info, key, obj=val)
            if key in special_conversion:
                print 'caught', key
                h5_file.create_array(
                    h5_file.root, special_conversion[key], obj=val
                    )

        # do extra extra conversions
        num_chan = g_obj.properties['nrColumns'] + g_obj.properties['nrBNCs']
        Fs = float(g_obj.properties['SamplingRate']) / \
          (g_obj.properties['OverSampling'] * g_obj.properties['nrRows'])
        h5_file.create_array(h5_file.root, 'numChan', num_chan)
        h5_file.create_array(h5_file.root, 'Fs', Fs)

        h5_file.flush()

        # now get down to the data
        atom = tables.Float64Atom()
        if compression_level > 0:
            filters = tables.Filters(
                complevel=compression_level, complib='zlib'
                )
        else:
            filters = None

        d_array = h5_file.create_earray(
            h5_file.root, 'data', atom=atom, shape=(0, n_row),
            filters=filters, expectedrows=n_col
            )

        ## col_mapping = [ch.properties.values()[0] for ch in chans]

        # create a reverse lookup to index channels by number
        col_mapping = dict(( (ch.properties['NI_ArrayColumn'], ch)
                             for ch in chans ))
        # If a channel permutation is requested, lay down channels
        # in that order. Otherwise go in sequential order.
        if chan_map:
            chan_map = np.loadtxt(chan_map).astype('i')
            if chan_map.ndim > 1:
                print chan_map.shape
                # the actual channel permutation is in the 1st column
                # the array matrix coordinates are in the next columns
                chan_ij = chan_map[:,1:3]
                chan_map = chan_map[:,0]
            else:
                chan_ij = None
            # do any channels not specified at the end
            if len(chan_map) < n_col:
                left_out = set(range(n_col)).difference(chan_map.tolist())
                left_out = sorted(left_out)
                chan_map = np.r_[chan_map, left_out]
        else:
            chan_map = range(n_col)
            chan_ij = None

        for n in chan_map:
            # get TDMS column
            ch = col_mapping[n]
            # make a temp array here.. if all data in memory, then this is
            # slightly wasteful, but if it is mmap'd then this is more flexible
            d = ch.data[:]
            d_array.append(d[None,:])
            print 'copied channel', n, d_array.shape

        if chan_ij is not None:
            h5_file.create_array(h5_file.root, 'channel_ij', obj=chan_ij)

    return h5_file

if __name__ == '__main__':
    import argparse

    prs = argparse.ArgumentParser(description='Convert TDMS to HDF5')
    prs.add_argument(
        'tdms_file', nargs = '+',
        help='path to the TDMS file', type=str
        )
    prs.add_argument(
        'h5_file', nargs = '+',
        help='name of the HDF5 file to create', type=str
        )
    prs.add_argument(
        '-p', '--permutation', type=str, default='',
        help='file with table of channel permutations'
        )
    prs.add_argument(
        '-m', '--memmap', help='use disk mapping for large files',
        action='store_true'
        )
    prs.add_argument(
        '-z', '--compression', type=int, default=0,
        help='use zlib level # compression in HDF5'
        )
    prs.add_argument(
        '-b', '--batch', help='Batch process all matching files',
        action='store_true'
        )

    args = prs.parse_args()

    if args.batch:
        hp = args.h5_file[0]
        if not os.path.isdir(hp):
            (hp, pf) = os.path.split(hp)
        else:
            pf = ''
        if len(args.tdms_file) > 1:
            # shell has globbed *.tdms
            all_tdms = args.tdms_file
        else:
            tdms = args.tdms_file[0]
            (tp, _) = os.path.split(tdms)
            all_tdms = glob(os.path.join(tp, '*.tdms'))
        all_h5 = list()
        for tdms in all_tdms:
            (_, tf) = os.path.split(tdms)
            (tf, ext) = os.path.splitext(tf)
            conv_file = os.path.join(hp, pf+tf+'.h5')
            if not os.path.exists(conv_file):
                all_h5.append( conv_file )
            else:
                all_h5.append( None )
    else:
        all_tdms = args.tdms_file
        all_h5 = args.h5_file

    for tf, hf in zip(all_tdms, all_h5):
        if hf:
            print tf, '\t', hf
            tdms_to_hdf5(
                tf, hf, chan_map=args.permutation, memmap=args.memmap,
                compression_level=args.compression
                )

