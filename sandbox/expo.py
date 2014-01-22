import numpy as np
import matplotlib as mpl

#import cmdline.strip_expo_xml as expo

import itertools
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
        'cell_ht', 'cell_x', 'cell_y'}. Then the data_lookup would be
        specified as dict(cell_ht=2, cell_wd=3, cell_x=4, cell_y=5)
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
        ChildInfo('32', dict(cell_ht=2, cell_wd=3, cell_x=4, cell_y=5)),
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


## XXX: the following section deserves better organization
from ecoglib.util import Bunch
class StimulatedExperiment(object):

    def __init__(
            self, trig_times=None, event_tables=dict(), **attrib
            ):
        self.trig_times = trig_times
        self.__dict__.update(event_tables)
        self.event_names = event_tables.keys()
        self.stim_props = Bunch(**attrib)
        
    def stim_str(self, n, mpl_text=False):
        return ''

    def __getitem__(self, slicing):
        sub_tables = dict()
        if self.trig_times is not None:
            sub_trigs = self.trig_times[slicing].copy()
        else:
            sub_trigs = None
        for name in self.event_names:
            table = self.__dict__[name]
            try:
                sub_tables[name] = table[slicing].copy()
            except:
                sub_tables[name] = table
        return type(self)(
            trig_times=sub_trigs, event_tables=sub_tables, 
            **self.stim_props
            )
    
class ExpoExperiment(StimulatedExperiment):

    event_type = StimEvent()
    skip_blocks = ()

    def __init__(self, trig_times=None, **tables):
        self.trig_times = trig_times
        self.event_tables = ()
        if tables:
            self.__dict__.update(tables)
            self._filled = True
        else:
            self._filled = False

    def fill_stims(self, xml_file, ignore_skip=False):
        if self._filled:
            for key in self.event_tables:
                del self.__dict__[key]
        data = self.event_type.walk_events(xml_file)
        keys = data.keys()
        if ignore_skip or not self.skip_blocks:
            keep_idx = slice(None)
        else:
            block_id = np.array(data['BlockID'])
            skipped_idx = [np.where(block_id == skip)[0] 
                           for skip in self.skip_blocks]
            skipped_idx = np.concatenate(skipped_idx)
            keep_idx = np.setdiff1d(np.arange(len(block_id)), skipped_idx)
        for key in keys:
            arr = data.pop(key)
            data[key] = np.array(arr)[keep_idx]

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
        self.stim_props = Bunch(pix_size=pix_size, tick_len=tick_len)
        self.__dict__.update(data)
        self.event_tables = data.keys()
        self._filled = True

class SparsenoiseExperiment(ExpoExperiment):

    event_type = SparsenoiseEvent()
    skip_blocks = (0,)

    def stim_str(self, n, mpl_text=False):
        if not self._filled:
            return 'empty experiment'
        con = (self.BlockID[n]-1) % 3
        cmap = mpl.cm.winter
        #cidx = np.linspace(0, cmap.N, 3)
        cidx = np.linspace(0, 1, 3)
        colors = cmap(cidx)
        if con == 0:
            contrast = 'dark'
        elif con == 1:
            contrast = 'mean'
        else:
            contrast = 'bright'

        x = self.cell_x[n]; y = self.cell_y[n]
        s = '(%1.1f, %1.1f)'%(x,y)
        if mpl_text:
            return mpl.text.Text(text=s, color=colors[con])
        return s + ' ' + contrast

class FlickerExperiment(ExpoExperiment):

    event_type = FlickerEvent

    def stim_str(self, n, mpl_text=False):
        return mpl.text.Text(text='*') if mpl_text else '*'

    
class WedgeExperiment(ExpoExperiment):

    event_type = WedgeEvent

    def stim_str(self, n, mpl_text=False):
        rot = self.rotation[n]
        s = '%1.2f deg'%self.rotation[n]
        if mpl_text:
            mn = self.rotation.min(); mx = self.rotation.max()
            c = mpl.cm.jet( (rot - mn)/(mx-mn) )
            return mpl.text.Text(text=s, color=c)
        return s

    
class RingExperiment(ExpoExperiment):

    event_type = RingEvent

    def stim_str(self, n, mpl_text=False):
        i_rad = self.inner_rad[n]
        o_rad = self.outer_rad[n]
        rad = (i_rad * o_rad)**0.5
        s = '%1.2f dva'%rad
        if mpl_text:
            i_mn = self.inner_rad.min(); i_mx = self.inner_rad.max()
            o_mn = self.outer_rad.min(); o_mx = self.outer_rad.max()
            mn = (i_mn * o_mn)**0.5
            mx = (i_mx * o_mx)**0.5
            c = mpl.cm.jet( (rad - mn)/(mx-mn) )
            return mpl.text.Text(text=s, color=c)
        return s

def get_expo_experiment(xml_file, trig_times, filled=True):
    i1 = xml_file.find('[') + 1
    i2 = xml_file.find(']')
    prog_name = xml_file[i1:i2]
    if prog_name == 'flicker':
        ex = FlickerExperiment(trig_times)
    elif prog_name == 'wedgeout':
        ex = WedgeExperiment(trig_times)
    elif prog_name == 'radialout':
        ex = RingExperiment(trig_times)
    elif prog_name.startswith('sparsenoise'):
        ex = SparsenoiseExperiment(trig_times)
    else:
        raise ValueError(
            'No translation for this program name: %s', prog_name
            )

    if filled:
        ex.fill_stims(xml_file)
    return ex
