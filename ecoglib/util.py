# ye olde utilities module

# ye olde Bunch object
class Bunch(dict):
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self

    def __repr__(self):
        return [(k, type(v)) for (k,v) in self.iteritems()]
