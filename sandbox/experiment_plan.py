import ConfigParser
import os
import numpy as np
from ecoglib.util import Bunch


_def_experiment_params = dict(
    shape_to='-1', # no sub-conditions, so don't reshape
    var1_name='cond', # maybe not needed here
    baseline='0'
    )

class _config_cache(object):
    cfg_file = os.path.join(
        os.path.split(__file__)[0], 'experimental_configurations.cfg'
        )
    cfg = ConfigParser.SafeConfigParser(_def_experiment_params)
    def __init__(self, file):
        self.cfg.read(file)
    
    @staticmethod
    def load(file=''):
        if not file:
            file = _config_cache.cfg_file
        return _config_cache(file)
    
_cfg_cache = _config_cache.load()

class ExperimentPlan(object):
    
    def __init__(self, exp_name, conditions):
        config = _cfg_cache.cfg
        if exp_name not in config.sections():
            raise ValueError('There is no configuration for %s'%exp_name)
        
        cond_shape = config.get(exp_name, 'shape_to')
        cond_shape = map(int, cond_shape.split('x'))
        
        self.condition_sequence = conditions
        conds = np.unique(conditions)
        if len(cond_shape) == 1:
            self.conds = conds.reshape(-1, 1)
            self.var_names = ()
        
        else:
            # find which dimension codes for variations, and
            # find the variation names
            self.conds = conds.reshape(cond_shape)
            var = filter(
                lambda x: x[0]>0, zip(cond_shape, range(len(cond_shape)))
                )
            if len(var) > 1:
                raise ValueError('only one reshape dim can be > -1')
            var = var[0]
            var_len = var[0]
            var_dim = var[1]
            self.var_names = list()
            for n in xrange(var_len):
                self.var_names.append(
                    config.get(exp_name, 'var%d_name'%(n+1))
                    )
        (self.n_conds, self.n_var) = self.conds.shape
        baseline = config.get(exp_name, 'baseline')
        try:
            self.baseline = 'interval'
            self.b_itvl = map(float, baseline.split(','))
        except ValueError:
            # baseline may be one of the variations
            baseline = baseline.strip()
            if not baseline.startswith('var'):
                raise ValueError(
                    'I do not understand the baseline: %s'%baseline
                    )
            self.baseline = 'variation'
            self.baseline = int(baseline.split('var')[1])
        
        self._init_maps()
                
        return
    
    def _init_maps(self):
        # for each "meta" condition return a map of indices at which
        maps = list()
        cseq = self.condition_sequence
        for cond in self.conds:
            c_maps = [ np.where(cseq==var)[0] for var in cond ]
            maps.append(c_maps)
        self.maps = maps
                                
    def walk_conditions(self):
        for c in xrange(self.n_conds):
            # so far so good
            b = Bunch(maps=self.maps[c])
            yield b
    
