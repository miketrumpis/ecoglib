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

        config.write(config_file)
        


if __name__=='__main__':
    import sys
    for pth in sys.argv[1:]:
        build_experiment_report(pth)
        print 'reported', pth
