#!/usr/bin/env python

# this will be an extensible tool for stripping
# condition labels, timings, and other customizable
# fields from the Expo XML record.
import sys
import os
from glob import glob
import warnings

import numpy as np
import scipy.io as sio

from sandbox.expo import *

def main(xml_file, mat_file):
    print 'creating XML parser for', xml_file
    exp = get_expo_experiment(xml_file, None)
    exp.fill_stims(xml_file)
    data = dict( (name, exp.__dict__[name]) for name in exp.event_tables )
    data.update(exp.stim_props)
    warnings.filterwarnings("ignore")
    sio.savemat(mat_file, data)

if __name__ == '__main__':

    import argparse

    dtext = """
    Strip Trial Info From Expo XML

    Arguments can be specified as one or more XML files followed by
    one or more corresponding MAT files in which to save sequence info.

    If running in batch mode (-b), then the program will process inputs
    globbed from the shell (i.e. expo_xml/*.xml), or it will do the
    wildcard globbing on its own. In batch mode, the MAT file argument
    indicates the path to save to, and optionally a file prefix. For
    example, in batch mode, the MAT file argument may be /matpath/prefix
    """

    #prs = argparse.ArgumentParser(description='Strip Trial Info From Expo XML')
    prs = argparse.ArgumentParser(description=dtext)
    prs.add_argument(
        'xml_file', nargs='+',
        help='path or pattern of the Expo XML file(s)', type=str
        )
    prs.add_argument(
        'mat_file', nargs='+',
        help='name or path of the MATLAB file(s) to create', type=str
        )
    prs.add_argument(
        '-b', '--batch', help='Batch process all matching files',
        action='store_true'
      )

    args = prs.parse_args()
    #print args

    if args.batch:
        mp = args.mat_file[0]
        if not os.path.isdir(mp):
            (mp, pf) = os.path.split(mp)
        else:
            pf = ''
        if len(args.xml_file) > 1:
            # shell has globbed *.xml
            all_xml = args.xml_file
        else:
            xml = args.xml_file[0]
            (xp, _) = os.path.split(xml)
            all_xml = glob(os.path.join(xp, '*.xml'))
        all_mat = list()
        for xml in all_xml:
            (_, xf) = os.path.split(xml)
            (xf, ext) = os.path.splitext(xf)
            all_mat.append( os.path.join(mp, pf+xf+'.mat') )
    else:
        all_xml = args.xml_file
        all_mat = args.mat_file

    for xf, mf in zip(all_xml, all_mat):
        print xf, '\t', mf
        check_closure(xf)
        try:
            main(xf, mf)
        except ValueError as ve:
            print ve
            if len(all_xml) > 1:
                print 'continuing with next file'


