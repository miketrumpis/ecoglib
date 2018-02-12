import numpy as np
import itertools
import scipy.misc as spmisc
from matplotlib.colors import BoundaryNorm, Normalize
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation, LinearTriInterpolator, \
     CubicTriInterpolator
## #from .util import Bunch, mat_to_flat, flat_to_mat, flat_to_flat
## import util as ut

def map_intersection(maps):
    geometries = set([m.geometry for m in maps])
    if len(geometries) > 1:
        raise ValueError('cannot intersect maps with different geometries')
    bin_map = maps[0].embed(np.ones(len(maps[0])), fill=0)
    for m in maps[1:]:
        bin_map *= m.embed(np.ones(len(m)), fill=0)
    return bin_map.astype('?')

class ChannelMap(list):
    "A map of sample vector(s) to a matrix representing 2D sampling space."
    
    def __init__(self, chan_map, geo, col_major=True, pitch=1.0):
        list.__init__(self)
        self[:] = chan_map
        self.col_major = col_major
        self.geometry = geo
        self.pitch = pitch
        self._combs = None

    @staticmethod
    def from_mask(mask, col_major=True, pitch=1.0):
        # create a ChannelMap from a binary grid
        i, j = mask.nonzero()
        geo = mask.shape
        from .util import mat_to_flat
        map = mat_to_flat(geo, i, j, col_major=col_major)
        return ChannelMap(map, geo, col_major=col_major, pitch=pitch)

    @property
    def site_combinations(self):
        if self._combs is None:
            self._combs = channel_combinations(self, scale=self.pitch)
        return self._combs
    
    def as_row_major(self):
        from .util import flat_to_flat
        if self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:]),
                self.geometry, col_major=False, pitch=self.pitch
                )
        return self

    def as_col_major(self):
        from .util import flat_to_flat
        if not self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:], col_major=False),
                self.geometry, col_major=True, pitch=self.pitch
                )
        return self

    def to_mat(self):
        from .util import flat_to_mat
        return flat_to_mat(self.geometry, self, col_major=self.col_major)

    def lookup(self, i, j):
        from .util import mat_to_flat
        flat_idx = mat_to_flat(self.geometry, i, j, col_major=self.col_major)
        if np.iterable(flat_idx):
            return np.array([self.index(fi) for fi in flat_idx])
        return self.index(flat_idx)

    def rlookup(self, c):
        from .util import flat_to_mat
        return flat_to_mat(self.geometry, self[c], col_major=self.col_major)

    def subset(self, sub, as_mask=False, map_intersect=False):
        """
        Behavior depends on the type of "sub":

        Most commonly, sub is a sequence (list, tuple, array) of subset
        indices.
        
        ChannelMap: return the subset map for the intersecting sites

        ndarray: if NOT subset indices (i.e. a binary mask), then the
        mask is converted to indices. If the array is a 2D binary mask,
        then site-lookup is used.
        
        """
        if isinstance(sub, type(self)):
            # check that it's a submap
            submap = map_intersection([self, sub])
            if submap.sum() < len(sub):
                raise ValueError(
                    'The given channel map is not a subset of this map'
                    )
            # get the channels/indices of the subset of sites
            sub = self.lookup(*submap.nonzero())
        elif isinstance(sub, np.ndarray):
            if sub.ndim==2:
                # Get the channels/indices of the subset of sites.
                # This needs to be sorted to look up the subset of
                # channels in sequence
                if map_intersect:
                    # allow 2d binary map to cover missing sites
                    ii, jj = sub.nonzero()
                    sites = []
                    for i, j in zip(ii, jj):
                        try:
                            sites.append(self.lookup(i, j))
                        except ValueError:
                            pass
                else:
                    # if this looks up missing sites, then raise
                    sites = self.lookup(*sub.nonzero())
                sub = np.sort( sites )
            elif sub.ndim==1:
                if sub.dtype.kind in ('b',):
                    sub = sub.nonzero()[0]
            else:
                raise ValueError('Cannot interpret subset array')
        elif not isinstance(sub, (list, tuple)):
            raise ValueError('Unknown subset type')

        if as_mask:
            mask = np.zeros( (len(self),), dtype='?' )
            mask[sub] = True
            return mask

        cls = type(self)
        return cls(
            [self[i] for i in sub], self.geometry,
            col_major=self.col_major, pitch=self.pitch
            )

    def __getslice__(self, i, j):
        cls = type(self)
        return cls(
            super(cls, self).__getslice__(i,j),
            self.geometry, col_major=self.col_major, pitch=self.pitch
            )

    def embed(self, data, axis=0, fill=np.nan):
        """
        Embed the data in electrode array geometry, mapping channels
        on the given axis
        """
        data = np.atleast_1d(data)
        shape = list(data.shape)
        if shape[axis] != len(self):
            raise ValueError(
                'Data array does not have the correct number of channels'
                )
        shape.pop(axis)
        shape.insert(axis, self.geometry[0]*self.geometry[1])
        array = np.empty(shape, dtype=data.dtype)
        if not isinstance(fill, str):
            array.fill(fill)
        slicing = [slice(None)] * len(shape)
        slicing[axis] = self.as_row_major()[:]
        array[slicing] = data
        shape.pop(axis)
        shape.insert(axis, self.geometry[1])
        shape.insert(axis, self.geometry[0])
        array.shape = shape
        if isinstance(fill, str):
            return self.interpolated(array, axis=axis)
        return array

    def as_channels(self, matrix, axis=0):
        """
        Take the elements of a matrix into the "natural" channel ordering.
        """
        m_shape = matrix.shape
        m_flat = m_shape[axis] * m_shape[axis+1]
        c_dims = m_shape[:axis] + (m_flat,) + m_shape[axis+2:]
        matrix = matrix.reshape(c_dims)
        return np.take(matrix, self, axis=axis)

    def inpainted(self, image, axis=0, **kwargs):
        pass

    def interpolated(self, image, axis=0, method='median'):
        # acts in-place
        mask = self.embed(np.zeros(len(self), dtype='?'), fill=1)
        missing = np.where( mask )
        g = self.geometry
        def _slice(i, j, w):
            before = [slice(None)] * axis
            after = [slice(None)] * (image.ndim - axis - 2)
            if w:
                isl = slice( max(0, i-w), min(g[0], i+w+1) )
                jsl = slice( max(0, j-w), min(g[1], j+w+1) )
            else:
                isl = i; jsl = j
            before.extend( [isl, jsl] )
            before.extend( after )
            return tuple(before)

        # first pass, tag all missing sites with nan
        for i, j in zip(*missing):
            image[ _slice(i, j, 0) ] = np.nan
        for i, j in zip(*missing):
            # do a +/- 2 neighborhoods (8 neighbors)
            patch = image[ _slice(i, j, 1) ].copy()
            s = list( patch.shape )
            s = s[:axis] + [ s[axis]*s[axis+1] ] + s[axis+2:]
            patch.shape = s
            fill = np.nanmedian( patch, axis=axis )
            image[ _slice(i, j, 0) ] = fill
        return image

    def image(
            self, arr=None, cbar=True, nan='//',
            fill=np.nan, ax=None, **kwargs
            ):
        kwargs.setdefault('origin', 'upper')
        if ax is None:
            import matplotlib.pyplot as pp
            f = pp.figure()
            ax = pp.subplot(111)
        else:
            f = ax.figure

        if arr is None:
            # image self
            arr = self.embed( np.ones(len(self), 'd'), fill=fill )
            kwargs.setdefault('clim', (0, 1))
            kwargs.setdefault('norm', BoundaryNorm([0, 0.5, 1], 2))
            kwargs.setdefault('cmap', cm.binary)
            
        if arr.shape != self.geometry:
            arr = self.embed(arr, fill=fill)

        nans = zip(*np.isnan(arr).nonzero())
        im = ax.imshow(arr, **kwargs)
        ext = kwargs.pop('extent', ax.get_xlim() + ax.get_ylim())
        dx = abs(float(ext[1] - ext[0])) / arr.shape[1]
        dy = abs(float(ext[3] - ext[2])) / arr.shape[0]
        x0 = min(ext[:2]); y0 = min(ext[2:])
        def s(x):
            return (x[0] * dy + y0, x[1] * dx + x0)
        if len(nan):
            for x in nans:
                r = Rectangle( s(x)[::-1], dx, dy, hatch=nan, fill=False )
                ax.add_patch(r)
        #ax.set_ylim(ext[2:][::-1])
        if cbar:
            cb = f.colorbar(im, ax=ax, use_gridspec=True)
            cb.solids.set_edgecolor('face')
            return f, cb
        return f

class CoordinateChannelMap(ChannelMap):
    "A map of sample vector(s) to a coordinate space."

    def __init__(self, coordinates, geometry='auto', pitch=1.0, col_major=True):
        """
        Parameters
        ----------
        coordinates : sequence
            sequence of (y, x) values
        geometry : pair (optional)
            Geometry is determined from coordinate range if set to 'auto'.
        
        """
        list.__init__(self)
        self[:] = coordinates
        if isinstance(geometry, (str, unicode)) and geometry.lower() == 'auto':
            yy, xx = zip(*self)
            # repurpose geometry to specify rectangle (??)
            # maintain "matrix" style coordinates, i.e. (y, x) <==> (i, j)
            self.geometry = (min(yy), max(yy), min(xx), max(xx))
        else:
            self.geometry = geometry
        self._combs = None
        self.pitch = pitch
        # this is nonsense, but to satisfy parent class
        self.col_major = col_major
        
    def to_mat(self):
        return map(np.array, zip(*self))

    def lookup(self, y, x):
        coords = np.array( self )
        dist = np.apply_along_axis(
            np.linalg.norm, 1, coords - np.array([y, x])
            )
        site = np.argmin( dist )
        return site

    def rlookup(self, c):
        return self[c]

    def subset(self, sub, as_mask=False):
        """
        Works mainly as ChannelMap.subset, with these exceptions:

        * sub may not be another ChannelMask type
        * sub may not be a 2D binary mask
                
        """

        if isinstance(sub, np.ndarray):
            if sub.ndim == 2:
                raise ValueError('No binary maps')
        elif isinstance(sub, ChannelMap) or not isinstance(sub, (tuple, list)):
            raise ValueError(
                "Can't interpret subset type: {0}".format(type(sub))
                )
        return super(CoordinateChannelMap, self).subset(sub, as_mask=as_mask)

    def image(
            self, arr=None, cbar=True, ax=None, interpolate='linear',
            grid_pts=100, norm=None, clim=None, cmap='gray',
            scatter_kw={}, contour_kw={}
            ):
        y, x = self.to_mat()
        if ax is None:
            import matplotlib.pyplot as pp
            f = pp.figure()
            ax = pp.subplot(111)
        else:
            f = ax.figure
        if arr is None:
            arr = np.ones_like(y)
            clim = (0, 1)
            norm = BoundaryNorm([0, 0.5, 1.0], 2)
            cmap = 'binary_r'
            interpolate = False

        if not clim:
            clim = arr.min(), arr.max()
        if not norm:
            norm = Normalize(*clim)
            
        if interpolate:
            arrg, coords = self.embed(arr, interpolate=interpolate,
                                      grid_pts=grid_pts, grid_coords=True)
            xg, yg = coords
            CS = ax.contourf(xg, yg, arrg, 10, clim=clim,
                             cmap=cmap, norm=norm, **contour_kw)
            if cbar:
                cb = f.colorbar(CS, ax=ax, use_gridspec=True)
                cb.solids.set_edgecolor('face')

        sct = ax.scatter(x, y, c=arr, norm=norm, cmap=cmap, **scatter_kw)
        if cbar:
            if not interpolate:
                cb = f.colorbar(sct, ax=ax, use_gridspec=True)
                cb.solids.set_edgecolor('face')
            return f, cb
        return f
            
    def embed(
            self, data, axis=0, interpolate='linear', grid_pts=100,
            grid_coords=False
            ):
        """
        Interpolates sample vector(s) in data onto a grid using Delauney
        triangulation. Interpolation modes may be "linear" or "cubic"
        """
        y, x = self.to_mat()
        triang = Triangulation(x, y)
        g = self.geometry
        yg = np.linspace(g[0], g[1], grid_pts)
        xg = np.linspace(g[2], g[3], grid_pts)
        xg, yg = np.meshgrid(xg, yg, indexing='xy')
        #xg = xg.ravel()
        #yg = yg.ravel()
        #arrg = griddata(x, y, arr, xg, yg)
        def f(x, interp_mode):
            xgr = xg.ravel()
            ygr = yg.ravel()
            if interp_mode == 'linear':
                interp = LinearTriInterpolator(triang, x)
            else:
                interp = CubicTriInterpolator(triang, x)
            return interp(xgr, ygr).reshape(grid_pts, grid_pts)
        #arrg = interp(xg.ravel(), yg.ravel()).reshape(grid_pts, grid_pts)
        arrg = np.apply_along_axis(f, axis, data, interpolate)
        return ( arrg, (xg, yg) ) if grid_coords else arrg
    
    # many methods no longer make sense with coordinates
    @staticmethod
    def as_col_major(self):
        raise NotImplementedError
    def as_row_major(self):
        raise NotImplementedError
    @staticmethod
    def from_mask(self, *args, **kwargs):
        raise NotImplementedError
    def as_channels(self, *args, **kwargs):
        raise NotImplementedError
    def interpolated(self, *args, **kwargs):
        raise NotImplementedError
    

def channel_combinations(chan_map, scale=1.0, precision=4):
    """Compute tables identifying channel-channel pairs.

    Parameters
    ----------
    chan_map : ChannelMap
    scale : float or pair
        The constant pitch or the (dx, dy) pitch between electrodes
        precision : number of decimals for distance calculation (it seems
        some distances are not uniquely determined in floating point).

    Returns
    -------
    chan_combs : Bunch
        Lists of channel # and grid location of electrode pairs and
        distance between each pair.
    """
    
    combs = itertools.combinations(np.arange(len(chan_map)), 2)
    from .util import Bunch
    chan_combs = Bunch()
    npair = spmisc.comb(len(chan_map),2,exact=1)
    chan_combs.p1 = np.empty(npair, 'i')
    chan_combs.p2 = np.empty(npair, 'i')
    chan_combs.idx1 = np.empty((npair,2), 'i')
    chan_combs.idx2 = np.empty((npair,2), 'i')
    chan_combs.dist = np.empty(npair)
    ii, jj = chan_map.to_mat()
    # Distances are measured between grid locations (i1,j1) to (i2,j2)
    # Define a (s1,s2) scaling to multiply these distances
    if np.iterable(scale):
        s_ = np.array( scale[::-1] )
    else:
        s_ = np.array( [scale, scale] )
    for n, c in enumerate(combs):
        c0, c1 = c
        chan_combs.p1[n] = c0
        chan_combs.p2[n] = c1
        idx1 = np.array( [ii[c0], jj[c0]] )
        idx2 = np.array( [ii[c1], jj[c1]] )
        chan_combs.dist[n] = np.round(np.linalg.norm( (idx1-idx2) * s_),
                                      decimals=precision)
        chan_combs.idx1[n] = idx1
        chan_combs.idx2[n] = idx2
    return chan_combs
