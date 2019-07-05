from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as pp
import numpy as np
from ecoglib.util import Bunch, mat_to_flat, flat_to_mat


def make_rectangles(
        locs, xy_size, facecolors='black', edgecolor='none', alpha=1,
        **patch_kws
        ):
    # locs should be an N x 2 matrix of (x,y) rectangle centers, or a
    # N-long sequence of (x,y) pairs
    nlocs = len(locs)
    nverts = nlocs * 5
    
    verts = np.zeros( (nverts, 2) )
    codes = np.ones( nverts, 'i' ) * Path.LINETO

    xy_size = map(float, xy_size)
    xh, yh = map(lambda x: x/2, xy_size)
    locs = np.asarray(locs)

    # bottom left (x,y)
    verts[0::5,0] = locs[:,0] - xh
    verts[0::5,1] = locs[:,1] - yh
    # bottom right (x,y)
    verts[1::5,0] = locs[:,0] + xh
    verts[1::5,1] = locs[:,1] - yh
    # top right (x,y)
    verts[2::5,0] = locs[:,0] + xh
    verts[2::5,1] = locs[:,1] + yh
    # top left (x,y)
    verts[3::5,0] = locs[:,0] - xh
    verts[3::5,1] = locs[:,1] + yh

    codes[0::5] = Path.MOVETO
    codes[4::5] = Path.CLOSEPOLY

    rectpath = Path(verts, codes)
    # colors ducked here
    return patches.PathPatch(
        rectpath, facecolor=facecolors, edgecolor=edgecolor,
        alpha=alpha, **patch_kws
        )

def make_circles(
        locs, diameter, facecolors='black', edgecolor='none', alpha=1,
        **patch_kws
        ):

    if isinstance(facecolors, str):
        facecolors = [facecolors] * len(locs)
    ellipses = list()
    for loc, fc in zip(locs, facecolors):
        ell = patches.Arc(
            loc, diameter[0], diameter[1], fc=fc, ec=edgecolor
            )
        ell.set_fill(True)
        ellipses.append(ell)
            
    return ellipses
                 

# array features given in mm units

_array_features = dict(
    psv_244 = Bunch(
        edge=200e-3, spacing=750e-3, geo=(16,16),
        missing_locs=( (0,0), (0,1), (0,14), (0,15),
                       (1,0), (1,15), (14,0), (14,15),
                       (15,0), (15,1), (15,14), (15,15) ),
        patch_method=make_rectangles
        ),
    psv_32 = Bunch(
        edge=400e-3, spacing=3, geo=(6,6),
        missing_locs=( (0,0), (0,5), (1,0), (1,5) ),
        patch_method=make_rectangles
        ),

    # these should be circles not squares 
    psv_61 = Bunch(
        edge=200e-3, spacing=406e-3, geo=(8,8),
        missing_locs=( (0,0), (0,7), (7,0), (7,7) ),
        patch_method=make_circles
        )
    )

def draw_array(
        arr_name, chan_map, colors='black', chan_set=(), zoom=1, **patch_kws
        ):
    # first get the locations for the given array
    aspec = _array_features[arr_name]

    geo = aspec['geo']
    spacing = aspec['spacing']

    y, x = flat_to_mat(geo, np.arange(geo[0]*geo[1]), col_major=False)
    y = y.astype('d'); x = x.astype('d')
    if 'missing_locs' in aspec:
        skipped = np.array( aspec['missing_locs'] )
        skipped = mat_to_flat(geo, skipped[:,1], skipped[:,0])
        keep = set(range(geo[0]*geo[1]))
        keep.difference_update(skipped.tolist())
        keep = list(keep)
        y = y[keep]
        x = x[keep]

    y = (geo[0] - y - 1) * spacing
    x *= spacing

    # get the edge sizes
    if 'edge_x' in aspec:
        xy_size = (aspec['edge_x'], aspec['edge_y'])
    else:
        xy_size = (aspec['edge'], aspec['edge'])

    # make the compound patch
    e_patches = aspec.patch_method(
        np.c_[x, y], xy_size, facecolors=colors, **patch_kws
        )
    #patches = make_rectangles(np.c_[x,y], xy_size, **patch_kws)

    # make the fig and axes, and add the patch
    ## figsize = map(
    ##     lambda x: (22/100. * x * spacing)*zoom,
    ##     geo
    ##     )
    figsize = map(
        lambda x: (0.5 * x * spacing)*zoom,
        geo
        )

    figsize = (geo[0] * spacing * zoom + 0.5, 
               geo[1] * spacing * zoom)
    fig = pp.figure(figsize=figsize[::-1], dpi=100)

    ax = fig.add_subplot(111, aspect='equal')
    fig.subplots_adjust(right=0.98, bottom=0, left=0.2, top=1.0)
    if isinstance(e_patches, patches.Patch):
        ax.add_patch(e_patches)
    elif isinstance(e_patches, (list, tuple)):
        for ep in e_patches:
            ax.add_patch(ep)
    xl = map(lambda x: x*spacing, (-0.5, geo[1]-0.5))
    yl = map(lambda x: x*spacing, (-0.5, geo[0]-0.5))
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.axis('off')

    xbar = Line2D( [xl[0], xl[0] + spacing], [yl[0], yl[0]],
                   color='black', linewidth=2 )
    ybar = Line2D( [xl[0], xl[0]], [yl[0], yl[0] + spacing],
                   color='black', linewidth=2 )
    ax.add_line(xbar)
    ax.add_line(ybar)
    units = '$\mu$m' if spacing < 1 else 'mm'
    bar_len = int(spacing*1000) if spacing < 1 else int(spacing)
    ax.text(
        xl[0]+spacing/2, yl[0]-0.5,
        '%d x %d %s'%(bar_len, bar_len, units),
        ha='center', fontsize=14
        )

    if chan_set is None:
        return fig

    # Now go and add the channel names
    chan_map = chan_map.as_row_major()
    if not len(chan_set):
        chan_set = range(len(chan_map))

    yy, xx = chan_map.to_mat()
    for y, x, chan_name in zip(yy, xx, chan_set):
        chan_name = str(chan_name)
        #y, x = flat_to_mat(geo, chan_idx, col_major=False)
        x *= spacing
        y = (geo[0] - y - 1 + 0.25) * spacing
        ax.text(
            x, y, chan_name, ha='center', va='bottom', fontsize=10
            )
    return fig
        
            
            
                