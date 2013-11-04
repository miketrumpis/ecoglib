#!/usr/bin/env python

# this will be an extensible tool for stripping
# condition labels, timings, and other customizable
# fields from the Expo XML record.
import sys
import os
from glob import glob
import itertools
import warnings

import numpy as np
import scipy.io as sio

try:
  from lxml import etree
  print("running with lxml.etree")
except ImportError:
  try:
    # Python 2.5
    import xml.etree.cElementTree as etree
    print("running with cElementTree on Python 2.5+")
  except ImportError:
      print "What's wrong with your distro??"
      sys.exit(1)

def check_closure(xmlfile):
    f = open(xmlfile)
    for ln in f:
        pass
    f.close()
    if ln.find('ExpoXData') < 0:
        f = open(xmlfile, 'a')
        f.write('\n</ExpoXData>\n')
    f.close()

def itertag_wrap(xml_ish, tag):
    """lxml provides a filter based on element tag.
    This wraps cElementTree to do the same.
    """
    if etree.__file__.find('lxml') > 0:
        context = etree.iterparse(xml_ish, tag=tag)
        return context

    context = etree.iterparse(xml_ish)
    fcontext = itertools.ifilter(lambda x: x[0].tag==tag, context)
    return fcontext

class StimEvent(object):
    """The mother stimulation event with tag 'Pass'. Every stimulation
    event carries the attributes BlockID, StartTime, and EndTime. The
    Pass ID is encoded implicitly as the order of all info sequences.
    """
    # also SlotID might be an important code

    tag = 'Pass'

    attr_keys = ('BlockID', 'StartTime', 'EndTime')

    children = ()

    @classmethod
    def walk_events(cls, xml):
        """Going to return a dictionary of named sequences.
        The names are the attr_keys of the stim event itself, as well
        the names of all the child info elements
        """
        pass_iter = itertag_wrap(xml, cls.tag)
        ## with itertag_wrap(xml, cls.tag) as pass_iter:
        all_names = list(cls.attr_keys)
        for child in cls.children:
            all_names.extend(child.data_lookup.keys())

        # fill with empty sequences
        named_seqs = dict(( (name, list()) for name in all_names ))

        for _, elem in pass_iter:
            # get top-level stuff
            # XXX: also uniformly converting to float here
            for key in cls.attr_keys:
                named_seqs[key].append( float(elem.attrib[key]) )
            for child in cls.children:
                c_data = child.strip_data(elem)
                for cname, cval in c_data:
                    named_seqs[cname].append(cval)

        return named_seqs

class ChildInfo(object):
    """A stimulation event might have relevant info in the children. The
    child nodes have tag 'Event' and an 'RID' to distinguish the data.
    """

    tag = 'Event'

    def __init__(self, rid, data_lookup):
        """Give the value of the RID of interest, as well as a data_lookup
        dictionary. RID can be appended with '.0', '.1', '.2', etc if the
        RID is repeated in the stimulation event's children. *This* child
        will of course be indexed by the RID modifier.

        The lookup works like this:
        the 'Data' attribute of the event of iterest is a comma-separated
        code, e.g.

        Data="1,0,0.999793,0.999793,-3.29783,4.1484,0,0,0,0"

        We want to save elements {2,3,4,5} with names {'cell_wd',
        'cell_ht', 'cell_y', 'cell_x'}. Then the data_lookup would be
        specified as dict(cell_ht=2, cell_wd=3, cell_y=4, cell_x=5)
        """
        self.rid = rid # a string
        self.data_lookup = data_lookup # a dict

    def strip_data(self, pass_node):
        """Find myself within the children of the stim event node
        and extract the relevant data.
        """
        rid_code = self.rid.split('.')
        rid = rid_code.pop(0)
        rep = int(rid_code.pop()) if rid_code else 0
        ev = filter(
            lambda x: x.attrib['RID']==rid, pass_node.getchildren()
            )
        if len(ev) <= rep-1:
            raise RuntimeError(
                'This pass has an unexpected multiplicity of children '\
                'with this RID: (%s, %s)'%(pass_node.attrib['ID'], rid)
                )
        if not ev:
            raise RuntimeError(
                'This pass has no children '\
                'with this RID: (%s, %s)'%(pass_node.attrib['ID'], rid)
                )
        ev = ev[rep]
        all_data = ev.attrib['Data'].split(',')
        data = ()
        for name, pos in self.data_lookup.items():
            # XXX: always converting to float here -- watch
            # out if string values are important
            data = data + ( (name, float(all_data[pos])), )
        return data

class FlickerEvent(StimEvent):
    # no ornaments
    pass

class SparsenoiseEvent(StimEvent):
    children = (
        ChildInfo('32', dict(cell_ht=2, cell_wd=3, cell_y=4, cell_x=5)),
        )

class RingEvent(StimEvent):
    children = (
        ChildInfo('32.0', dict(outer_rad=2)),
        ChildInfo('32.1', dict(inner_rad=2)),
        )

class WedgeEvent(StimEvent):
    children = (
        ChildInfo('32', dict(rotation=1)),
        )

def get_expo_event(xml_file):
    i1 = xml_file.find('[') + 1
    i2 = xml_file.find(']')
    prog_name = xml_file[i1:i2]
    if prog_name == 'flicker':
        return FlickerEvent()
    if prog_name == 'wedgeout':
        return WedgeEvent()
    if prog_name == 'radialout':
        return RingEvent()
    if prog_name.startswith('sparsenoise'):
        return SparsenoiseEvent()
    else:
        raise ValueError('No translation for this program name: %s', prog_name)

def main(xml_file, mat_file):
    print 'creating XML parser for', xml_file
    event = get_expo_event(xml_file)
    data = event.walk_events(xml_file)
    keys = data.keys()
    for key in keys:
        arr = data.pop(key)
        data[key] = np.array(arr)

    # do a second spin through to pick up the units conversions
    context = itertag_wrap(xml_file, 'Environment')
    for _, elem in context:
        units = elem.getchildren()[0]
        # pix size is uncertain .. should be dva
        pix_size = float(units.attrib['PixelSize'])
        # tick duration is in micro-secs
        tick_len = float(units.attrib['TickDuration'])
    print 'got data:', data.keys()
    print [len(val) for val in data.values()]
    print 'writing mat file:', mat_file
    data.update(
        pix_size = np.array([pix_size]),
        tick_len = np.array([tick_len])
        )
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


