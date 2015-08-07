from __future__ import division

import numpy as np
import matplotlib as mpl

#import cmdline.strip_expo_xml as expo

import itertools
try:
  from lxml import etree
except ImportError:
  try:
    # Python 2.5
    import xml.etree.cElementTree as etree
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
        ChildInfo('33', dict(contrast=3))
        )

class RingEvent(StimEvent):
    children = (
        ChildInfo('32.0', dict(outer_rad=2)),
        ChildInfo('32.1', dict(inner_rad=2)),
        )

class WedgeEvent(StimEvent):
    children = (
        ChildInfo('32', dict(rotation=1, radius=2)),
        )

class TickEvent(StimEvent):
    tag = 'tick'
    attr_keys = ('start', 'end', 'flush')

## XXX: the following section deserves better organization
from ecoglib.util import Bunch
class StimulatedExperiment(object):
    enum_tables = ()
    
    def __init__(self, trig_times=(), event_tables=dict(), **attrib):
        if trig_times is None:
            self.trig_times = ()
        else:
            self.trig_times = trig_times
        self._fill_tables(**event_tables)
        self.stim_props = Bunch(**attrib)

    def set_enum_tables(self, table_names):
        if isinstance(table_names, str):
            table_names = (table_names,)
        good_tabs = filter(lambda x: x in self.event_names, table_names)
        if len(good_tabs) < len(table_names):
            raise ValueError('some table names not found in this exp')
        self.enum_tables = table_names

    def enumerate_conditions(self):
        """
        Return the map of condition labels (counting numbers, beginning
        at 1), as well as tables to decode the labels into multiple
        stimulation parameters.

        Returns
        -------

        conditions : ndarray, len(experiment)
            The condition label (1 <= c <= n_conditions) at each stim event

        cond_table : Bunch
            This Bunch contains an entry for every stimulation parameter.
            The entries are lookup tables for the parameter values.

        """
        
        if not len(self.trig_times):
            return (), Bunch()
        tab_len = len(self.trig_times)
        if not self.enum_tables:
            return np.ones(tab_len, 'i'), Bunch()

        all_uvals = []
        conditions = np.zeros(len(self.trig_times), 'i')
        for name in self.enum_tables:
            tab = self.__dict__[name][:tab_len]
            uvals = np.unique(tab)
            all_uvals.append(uvals)
            conditions *= len(uvals)
            for n, val in enumerate(uvals):
                conditions[ tab==val ] += n

        n_vals = map(len, all_uvals)
        for n in xrange(len(all_uvals)):
            # first tile the current table of values to
            # match the preceeding "most-significant" values,
            # then repeat the tiled set to match to following
            # "least-significant" values
            utab = all_uvals[n]
            if all_uvals[n+1:]:
                rep = reduce(np.multiply, n_vals[n+1:])
                utab = np.repeat(utab, rep)
            if n > 0:
                tiles = reduce(np.multiply, n_vals[:n])
                utab = np.tile(utab, tiles)
            all_uvals[n] = utab
        
        #utab = np.tile(utab, n_vals / len(utab))
        #all_uvals[-1] = utab
        # 1-offset or 0-offset?
        conditions += 1
        return conditions, Bunch(**dict(zip(self.enum_tables, all_uvals)))

    def rolled_conditions_shape(self):
        _, ctab = self.enumerate_conditions()
        return tuple(
            [len(np.unique(ctab[tab])) for tab in self.enum_tables]
            )

    def iterate_for(self, tables, c_slice=False):
        if isinstance(tables, str):
            tables = (tables,)
        t_vals = [ np.unique(getattr(self, t)) for t in tables ]
        if c_slice:
            conds, tabs = self.enumerate_conditions()
            t_idx = [ self.enum_tables.index(t) for t in tables ]
            slices = [slice(None)] * len(self.enum_tables)
        for combo in itertools.product(*t_vals):
            mask = [ getattr(self, t) == combo[i] 
                     for i, t in enumerate(tables) ]
            mask = np.row_stack(mask)
            if c_slice:
                sl = slices[:]
                for i in xrange(len(combo)):
                    sl[ t_idx[i] ] = t_vals[i].searchsorted(combo[i])
                yield mask.all(axis=0), sl
            else:
                yield mask.all(axis=0)
    
    def stim_str(self, n, mpl_text=False):
        if mpl_text:
            return mpl.text.Text(text='')
        return ''

    def _fill_tables(self, **tables):
        self.__dict__.update(tables)
        self.event_names = tables.keys()
        for k, v in tables.items():
            setattr(self, 'u'+k, np.unique(v))
    
    def __getitem__(self, slicing):
        sub_tables = dict()
        if len(self.trig_times):
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

    def __len__(self):
        return len(self.trig_times)

    def subexp(self, indices):
        if hasattr(indices, 'dtype') and indices.dtype.char == '?':
            indices = np.where(indices)[0]
        # take advantage of fancy indexing?
        return self.__getitem__(indices)

    def extend(self, experiment, offset):
        if type(self) != type(experiment):
            raise TypeError('Can only join experiments of the same type')
        first_trigs = self.trig_times
        second_trigs = experiment.trig_times + offset
        trigs = np.r_[first_trigs, second_trigs]

        new_tables = dict()
        for name in self.event_names:
            tab1 = eval('self.%s'%name)
            tab2 = eval('experiment.%s'%name)
            new_tables[name] = np.r_[tab1, tab2]

        new_props = self.stim_props # ??? 

        return type(self)(
            trig_times=trigs, event_tables=new_tables, **new_props
            )

def repeat_tonotopy_sequences(trig_times, tones_pattern, amps_pattern):
    n_trig = len(trig_times)
    seq_len = len(tones_pattern)
    n_rep = n_trig // seq_len + 1
    tones = np.tile(tones_pattern, n_rep)[:n_trig]
    amps = np.tile(amps_pattern, n_rep)[:n_trig]
    #self._fill_tables(tones=tones, amps=amps)
    return tones, amps

    
class TonotopyExperiment(StimulatedExperiment):

    ## These patterns were used in one set of exps..
    fixed_tones_pattern = \
      (2831, 8009, 5663, 8009, 2831, 1000, 1000, 16018, 1415, 4004, 
       4004, 22651, 708, 1415, 500, 5663, 2000, 1000, 5663, 22651, 
       16018, 2000, 500, 32036, 2831, 708, 500, 11326, 11326, 708, 
       1415, 2000, 16018, 32036, 32036, 8009, 4004, 11326, 22651)

    fixed_amps_pattern =  (30, 50, 30, 30, 70, 70, 50, 70, 30, 30, 70, 30, 30, 
                           50, 50, 70, 70, 30, 50, 50, 30, 30, 30, 50, 50, 50, 
                           70, 30, 50, 70, 70, 50, 50, 30, 70, 70, 50, 70, 70)

    def __init__(self, trig_times=None, tone_tab='', amp_tab='', **kwargs):
        # * trig_times is the sequence of trial event timestamps
        # * tone_tab is 
        #     1) a path to a table with the tone pattern sequence
        #     2) a tone pattern sequence 
        # * amp_tab is like tone tab, specifying the corresponding 
        #   amplitude sequence
        # given tables are over-ridden by the 'event_tables' dictionary
        
        
        if trig_times is None or not len(trig_times):
            #raise ValueError('needs trig_times to proceed')
            trig_times = np.array([])

        # The tone/amp tables may be given in the event_tables 
        # dictionary, which should be preferred to 'tone_tab'
        # and 'amp_tab' keyword args.

        # The tone/amp tables can come in fully specified (i.e. 
        # with the same length as the trigger times. If not,
        # then repeat the tables to match the number of trials

        event_tables = kwargs.pop('event_tables', None)
        if not event_tables:
        
            if tone_tab == '' or amp_tab == '':
                tones = np.array(self.fixed_tones_pattern)
                amps = np.array(self.fixed_amps_pattern)
            else:
                if not isinstance(tone_tab, np.ndarray):
                    # treat it as a text table
                    tones = np.loadtxt(tone_tab)
                else:
                    tones = tone_tab
                if tones[0] == 0:
                    tones= tones[1:]
                if not isinstance(amp_tab, np.ndarray):
                    # treat it as a text table
                    amps = np.loadtxt(amp_tab)
                else:
                    amps = amp_tab

            if len(tones) != len(trig_times):
                tones, amps = repeat_tonotopy_sequences(
                    trig_times, tones, amps
                    )
            event_tables = dict(tones=tones, amps=amps)
                
        super(TonotopyExperiment, self).__init__(
            trig_times=trig_times, event_tables=event_tables, **kwargs
            )
            
        self.set_enum_tables( ('tones', 'amps') )


    def stim_str(self, n, mpl_text=False):
        tone = self.tones[n]
        amp = self.amps[n]

        tone_khz = tone // 1000
        tone_dec = tone - tone_khz * 1000
        tone_dec = int( float(tone_dec) / 100 + 0.5 )

        s = '%d.%d KHz'%(tone_khz, tone_dec)
        s = s + ' (%d)'%amp
        if mpl_text:
            #ctab = mpl.cm.gnuplot2([0.1, 0.5, 0.9])
            u_amps = np.unique(self.amps)
            ctab = mpl.cm.jet(np.linspace(0.1, 0.9, len(u_amps)))
            cidx = u_amps.searchsorted(amp)
            return mpl.text.Text(text=s, color=ctab[cidx])
        else:
            return s
    
class ExpoExperiment(StimulatedExperiment):

    event_type = StimEvent()
    skip_blocks = ()

    def __init__(self, trig_times=None, event_tables=dict(), **attrib):
        super(ExpoExperiment, self).__init__(
            trig_times=trig_times, event_tables=event_tables, **attrib
            )
        self._filled = len(self.event_names) > 0

    def fill_stims(self, xml_file, ignore_skip=False):
        if self._filled:
            for key in self.event_names:
                del self.__dict__[key]
        # get tick events for good measure
        ticks = TickEvent.walk_events(xml_file)
        for attrib in TickEvent.attr_keys:
            val = ticks.pop(attrib)
            ticks['tick_'+attrib] = val
        data = self.event_type.walk_events(xml_file)
        data.update(ticks)
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
        ## print 'got data:', data.keys()
        ## print [len(val) for val in data.values()]
        self.stim_props = Bunch(pix_size=pix_size, tick_len=tick_len)
        self._fill_tables(**data)
        self._filled = True

class SparsenoiseExperiment(ExpoExperiment):

    event_type = SparsenoiseEvent()
    skip_blocks = (0,)
    enum_tables = ('cell_x', 'cell_y', 'contrast')

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
    enum_tables = ('rotation',)

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
    enum_tables = ('inner_rad',)

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

def join_experiments(exps, offsets):
    if len(exps) < 1 or not len(offsets):
        raise ValueError('Empty experiment or offset list')
    if len(exps) < 2:
        return exps[0]
    if len(offsets) < len(exps) - 1:
        raise ValueError('Not enough offset points given to join experiments')
    new_exp = exps[0].extend(exps[1], offsets[0])
    for n in xrange(2, len(exps)):
        new_exp = new_exp.extend(exps[n], offsets[n-1])
    return new_exp

def ordered_epochs(exptab, fixed_vals, group_sizes=False):
    """
    Returns an index into the StimulatedExperiment tables that fixes
    all but one parameter, and returns the floating parameter in sorted
    order.

    Parameters
    ----------

    exptab : StimulatedExperiment

    fixed_vals : sequence
        A sequence of parameter names and fixed values:
        ( (param1, val1), (param2, val2), ... )

    group_sizes : bool
        Indicate whether to return the number of events found for
        each value of the floating parameter.
    
    """
    
    event_labels, ctab = exptab.enumerate_conditions()
    ## if len(fixed_vals) != len(ctab) - 1:
    ##     raise ValueError('must fix all but one of the stim parameters')

    n_conds = len( np.unique(event_labels) )
    
    c_mask = np.ones(n_conds, dtype='?')
    for param, val in fixed_vals:
        c_mask = c_mask & (ctab[param] == val)

    ordered_conds = np.where(c_mask)[0] + 1

    events = [ np.where(event_labels==c)[0] for c in ordered_conds ]
    sizes = [len(ev) for ev in events]
    events = np.concatenate(events)
    if group_sizes:
        return events, sizes
    return events

    
