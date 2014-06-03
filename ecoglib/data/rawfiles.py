# manipulations of raw data files
import tables
import numpy as np
import ConfigParser
import os
import contextlib
import tempfile
import nptdms
from glob import glob

def build_experiment_report(pth, ext='h5'):

    glob_ext = '*.'+ext
    all_h5 = glob(os.path.join(pth, glob_ext))
    config = ConfigParser.RawConfigParser()

    for f in all_h5:
        h5 = tables.open_file(f)

        try:
            info = h5.get_node(h5.root, 'info')
        except tables.NoSuchNodeError:
            continue

        exp_name = os.path.split(f)[-1]
        exp_name = os.path.splitext(exp_name)[0]

        config.add_section(exp_name)
        config.set(exp_name, 'Fs', str(h5.root.Fs.read()))
        for item in (
                'nrColumns', 'nrRows', 'nrBNCs', 
                'SampleStripLength', 'OverSampling', 
                'SamplingRate', 'ColumnMixVector', 'Note'):
            try:
                val = eval( 'info.'+item+'.read()' )
            except tables.NoSuchNodeError:
                val = 'ITEM NOT FOUND'
            if not np.iterable(val):
                val = str(val)
            config.set(exp_name, item, val)

        trig_info = find_triggers(h5)
        trig_fields = ('data_length', 'num_triggers', 
                       'first_trigger', 'last_trigger')
        for item, val in zip(trig_fields, trig_info):
            config.set(exp_name, item, str(val))

    return config

def find_triggers(h5_file):
    # returns data length, # triggers, first & last trigger times
    numcols = h5_file.root.info.nrColumns.read()
    numrows = h5_file.root.info.nrRows.read()
    dshape = h5_file.root.data.shape
    chan_dim = np.argmin(dshape)
    d_len = dshape[1-chan_dim]

    if not dshape[chan_dim] > numcols * numrows:
        return d_len, None, None, None

    if chan_dim:
        trigs = h5_file.root.data[:,numcols*numrows:(numcols+1)*numrows].T
    else:
        trigs = h5_file.root.data[numcols*numrows:(numcols+1)*numrows]
    # use 40 % of max as the logical threshold
    mx = np.median( trigs.max(axis=1) )
    thresh = 0.4 * mx

    # do a quick check to make sure this is a binary BNC
    a = np.var( trigs[ trigs > thresh] ) / mx**2
    b = np.var( trigs[ trigs < thresh] ) / mx**2
    if 0.5*(a+b) > 1e-2:
        return d_len, None, None, None
    
    trigger = np.any( trigs > thresh, axis=0 )
    pos_edge = np.where( np.diff(trigger) > 0 )[0]
    if len(pos_edge):
        return d_len, len(pos_edge), pos_edge[0], pos_edge[-1]
    else:
        return d_len, None, None, None

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

