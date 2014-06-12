#!/usr/bin/env python
import os
from ecoglib.data.rawfiles import build_experiment_report

def write_experiment_report(pth, ext='h5'):
    report = os.path.join(pth, '__Experiment_Report.txt')
    with open(report, 'w') as config_file:

        config_file.write(
"""
==== AUTO-GENERATED EXPERIMENT REPORT ====
====         *DO NOT EDIT*            ====

"""
            )
        config = build_experiment_report(pth, ext=ext)
        config.write(config_file)

if __name__=='__main__':
    import sys
    for pth in sys.argv[1:]:
        write_experiment_report(pth)
        print 'reported', pth
