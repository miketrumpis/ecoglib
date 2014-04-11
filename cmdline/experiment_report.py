#!/usr/bin/env python
import numpy as np
import tables
import ConfigParser
import os
from glob import glob

def build_experiment_report(pth, ext='h5'):

    report = os.path.join(pth, '__Experiment_Report.txt')
    with open(report, 'wb') as config_file:

        config_file.write(
"""
==== AUTO-GENERATED EXPERIMENT REPORT ====
====         *DO NOT EDIT*            ====

"""
            )
        
        glob_ext = '*.'+ext
        all_h5 = glob(os.path.join(pth, glob_ext))
        print all_h5
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
                    
                
        config.write(config_file)

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
    


if __name__=='__main__':
    import sys
    for pth in sys.argv[1:]:
        build_experiment_report(pth)
        print 'reported', pth
