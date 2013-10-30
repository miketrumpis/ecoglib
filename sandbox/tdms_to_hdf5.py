#!/usr/bin/env python
import numpy as np
import tables
import nptdms
import tempfile
import os, sys
import contextlib

def tdms_to_hdf5(tdms_file, h5_file, memmap=True, compression_level=0):

    #map_dir = tempfile.gettempdir() if memmap else None
    map_dir = './tmp' if memmap else None
    try:
        os.mkdir('./tmp')
    except OSError:
        pass

    # put these operations in a context so that the temp files
    # get killed whether or not things go well inside
    #with nptdms.TdmsFile(tdms_file, memmap_dir=map_dir) as tdms_file:
    with contextlib.nested(nptdms.TdmsFile(tdms_file, memmap_dir=map_dir),
                           tables.open_file(h5_file, mode='w')) as \
                           (tdms_file, h5_file):

        #h5_file = tables.open_file(h5_file, mode='w')

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
                val = str(val)
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
        ## d_array = h5_file.create_carray(
        ##     h5_file.root, 'data', atom=atom, shape=(n_row, n_col),
        ##     filters=filters
        ##     )
        # this one is transposed
        d_array = h5_file.create_earray(
            h5_file.root, 'data', atom=atom, shape=(0, n_row),
            filters=filters, expectedrows=n_col
            )

        # the following *should* be a dictionary lookup for 'NI_ArrayColumn'
        col_mapping = [ch.properties.values()[0] for ch in chans]
        col_mapping = np.argsort(col_mapping)

        for n in xrange(n_col):
            # get TDMS column
            ch = chans[ col_mapping[n] ]
            # make a temp array here.. if all data in memory, then this is
            # slightly wasteful, but if it is mmap'd then this is more flexible
            d = ch.data[:]
            d_array.append(d[None,:])
            print 'copied channel', n, d_array.shape

    #h5_file.flush()
    #h5_file.close()
    return h5_file




if __name__ == '__main__':
    argv = sys.argv[1:]
    t_file, h_file = argv[:2]
    memmap = bool(int(argv[2])) if len(argv) > 2 else True
    c_level = int(argv[3]) if len(argv) > 3 else 0

    tdms_to_hdf5(t_file, h_file, memmap=memmap, compression_level=c_level)
