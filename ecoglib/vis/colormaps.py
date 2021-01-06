"""
Color and colormap tricks, extending Matplotlib and Seaborn.
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
from itertools import cycle
from . import plotters


__all__ = ['nancmap',
           'diverging_cm',
           'rgba_field',
           'composited_color_palette',
           'GroupPlotColors']


_cmap_db = dict()


def nancmap(cmap_name, nanc=(1, 1, 1, 1), underc=None, overc=None, N=None):
    """Create a matplotlib colormap with NaN (and other special values
    mapped to given color(s). Since matplotlib colormaps are
    persistent objects in the namespace, creating a new colormap
    prevents changing the nan-color for all time.

    Parameters
    ----------
    cmap_name : str
        Name of the existing colormap (can be found in cmap_d)
    nanc : color
        The NaN color is a sequence or hex string that can be
        interpreted as RGB(A).
    underc : color
        The color for scalars below the color limits.
    overc : color
        The color for scalars above the color limits.
    N : int
        The number of quantization levels (defaults to rcParams)

    Returns
    -------
    cmap : matplotlib colormap

    """
    if not N:
        mpl = plotters.mpl
        N = mpl.rcParams['image.lut']

    cmap = cm._generate_cmap(cmap_name, N)
    if isinstance(nanc, str):
        name = nanc if nanc[0] == '#' else colors.cnames[nanc]
        nanc = colors.hex2color(name)
    if isinstance(overc, str):
        name = overc if overc[0] == '#' else colors.cnames[overc]
        overc = colors.hex2color(name)
    if isinstance(underc, str):
        name = underc if underc[0] == '#' else colors.cnames[underc]
        underc = colors.hex2color(name)

    cmap.set_bad(nanc)
    if overc:
        cmap.set_over(overc)
    if underc:
        cmap.set_under(underc)

    return cmap


def z_cmap(cmap='bwr', N=None, z_max=4):
    """
    Return a colormap to be used for Normal "Z" scores, where saturation is tied to the Normal CDF.

    Brightness = 2 * (CDF(|z|) - 1 / 2)

    Parameters
    ----------
    cmap: str, matplotlib.colors.Colormap
        Base map to convert (diverging maps put white in the z=0 zone).
    z_max: float
        Saturate at this value (put set_under and set_over at this value).

    Returns
    -------
    z_map: colors.ListedColormap

    """
    if not N:
        mpl = plotters.mpl
        N = mpl.rcParams['image.lut']
    if not isinstance(cmap, colors.Colormap):
        cmap = cm.get_cmap(cmap)
    from scipy.stats.distributions import norm
    hsv_colors = colors.rgb_to_hsv(cmap(np.linspace(0, 1, N))[:, :3])
    z_values = np.linspace(-z_max, z_max, N)
    hsv_colors[:, 2] = 2 * (norm.cdf(np.abs(z_values)) - 0.5)
    rgb_colors = colors.hsv_to_rgb(hsv_colors)
    rgb_colors = np.c_[rgb_colors, np.ones(N)]
    z_map = colors.ListedColormap(rgb_colors, name=cmap.name + '_z')
    return z_map


def diverging_cm(
        mn, mx, cmap='bwr', zero='white', compression=1.0
):
    """Build a potentially non-symmetric diverging colormap.

    Parameters
    ----------
    mn, mx : scalars
        The limits of a range mn < 0 < mx.
    cmap
        cmap can be the name of a matplotlib colormap. Otherwise it can
        be a pair of colors specified in hex or RGB(A).
    zero : color
        Color of the zero value (will change style of an existing
        colormap such as 'bwr').
    compression : scalar
        The color gradient from zero to mn/mx takes on the shape
	x**compression for 0 < x < 1. A value less than one creates a
	steep rise with broad saturation. A value greater than one
	creates a broad range close to the zero color.

    Returns
    -------
    cmap : matplotlib colormap

    Note
    ----
    The compressed range feature is not a literal function of input
    scalar value, it only creates a shape for the color
    gradient. Specifically, if the maximum scalar value is twice the
    magnitude of the minimum scalar value, then the gradient for the
    positive range is expanded by a factor of two with respect to the
    negative range.

    """

    mn, mx = map(float, (mn, mx))
    if mn > 0 or mx < 0:
        raise ValueError('Range of values must span zero')

    if isinstance(cmap, str):
        cmap = cm.cmap_d[cmap]
    elif isinstance(cmap, tuple) or isinstance(cmap, list):
        # parse colors
        cneg, cpos = map(colors.colorConverter.to_rgb, cmap)
        ## if isinstance(cneg, str):
        ##     cneg, cpos = map(colors.cnames.get, (cneg, cpos))
        ##     if cneg == None:
        ##         cneg = colors.hex2color(cmap[0])
        ##     if cpos == None:
        ##         cpos = colors.hex2color(cmap[1])
    if isinstance(cmap, colors.Colormap):
        # get the neg and pos colors
        cneg = cmap(0.0)
        cpos = cmap(1.0)

    # build new colormap with unbalanced poles
    zcolor = colors.colorConverter.to_rgb(zero)
    zero = abs(mn) / (mx - mn)

    n_cmap = len(_cmap_db)
    cm_name = 'div_cmap_%d' % (n_cmap + 1,)
    cmap = colors.LinearSegmentedColormap.from_list(
        cm_name, [(0, cneg), (zero, zcolor), (1, cpos)]
    )
    if compression != 1:
        # make a piecewise polynomial input-output curve bilateral around zero
        N = 1000
        pN = int((1 - zero) * N)
        nN = N - pN
        p_r = np.power(np.linspace(0, 1, pN), compression) * (1 - zero) + zero
        n_r = np.power(np.linspace(0, 1, nN), compression)
        n_r = zero - (1 - zero) * n_r[::-1]
        c = cmap(np.r_[n_r, p_r])
        cmap = colors.ListedColormap(c, name=cm_name)
    _cmap_db[cm_name] = cmap
    return cmap


def rgba_field(cmap, sfield, afield=None, clim=(), alim=()):
    """Map scalars to an array of RGBA values with varying alpha field.

    This method extends the matplotlib image mapping to allow for a
    non-constant field of transparency (alpha) values.

    Parameters
    ----------
    cmap : matplotlib colormap
    sfield : ndarray, 2D
        The field of scalars to map to RGB values (using clim)
    afield : ndarray, 2D
        The field of scalars to map to alpha values (using alim)

    Returns
    -------
    rgba : uint ndarray
        RGBA values with shape sfield.shape + (4,). Can be imaged
        directly with pyplot.imshow(rgba)
    _make_cbar : method
        Callable to create the correct scalar colormap for rgba (see
        Note).

    Note
    ----
    _make_cbar(axes, ticks=[...], orientation='horizontal')
    This creates a colorbar for the scalar map in the given axes.
    If the alpha field was supplied, the short dimension of the
    colorbar rectangle is graded with alpha values from 1:0,
    indicating the full range of colors used in the RGBA map.

    """

    norm = colors.Normalize(*clim)
    n_sfield = norm(sfield)
    cmap = cm.cmap_d.get(cmap, cmap)
    rgba = cmap(n_sfield, bytes=True)
    if afield is not None:
        n_afield = colors.Normalize(*alim)(afield)
        rgba[..., -1] = np.clip(np.round(n_afield * 255), 0, 255).astype(rgba.dtype)

    def _make_cbar(cax, ticks=(), orientation='vertical'):

        pos = cax.get_position()
        cbar_img = cmap(
            np.tile(np.linspace(0, 1, 100), (20, 1)).T, bytes=True
        )
        if afield is not None:
            saturation = 255 * np.linspace(1, 0, 20, endpoint=False) ** 2
            cbar_img[:, :, -1] = saturation.astype('B')

        # make the extent run from scalar min-max in the long
        # direction, and an amount in the short dimension
        # to match the proportions of the axes
        s_min = norm.vmin;
        s_max = norm.vmax

        ax_wd = pos.x1 - pos.x0
        ax_ht = pos.y1 - pos.y0

        if orientation == 'horizontal':
            extent = [s_min, s_max, 0, ax_ht / ax_wd * (s_max - s_min)]
            cbar_img = cbar_img.transpose(1, 0, 2)
        else:
            extent = [0, ax_wd / ax_ht * (s_max - s_min), s_min, s_max]

        cax.imshow(cbar_img, extent=extent)
        # cax.axis('equal')
        cax.set_aspect('auto')
        cax.set_xlim(*extent[:2]);
        cax.set_ylim(extent[2:])

        if not len(ticks):
            ticks = np.linspace(s_min, s_max, 6)

        if orientation == 'vertical':
            cax.yaxis.tick_right()
            cax.xaxis.set_visible(False)
            cax.set_yticks(ticks)
        else:
            cax.xaxis.tick_bottom()
            cax.yaxis.set_visible(False)
            cax.set_xticks(ticks)

    return rgba, _make_cbar


def composited_color_palette(alpha=1.0, **pargs):
    """Create a white-mixed color palette from a Seaborn palette.

    """
    sns = plotters.sns
    try:
        colors = sns.color_palette(**pargs)
    except TypeError:
        # try to catch API change
        pargs['palette'] = pargs.pop('name')
        colors = sns.color_palette(**pargs)
    if alpha < 1.0:
        colors = np.array(colors) * alpha + (1 - alpha)
        colors = [tuple(c) for c in colors]
    return colors


class GroupPlotColors(object):
    """
    Yields color tables based on a rotating-hue scheme. Each "group"
    to be plotted is assigned a new hue, and colors are chosen between
    the 25% and 75% values in HSV specs. Linestyles may also be cycled
    between groups (max of 4 styles?)
    """

    linestyles = ['-', '--', '-.', ':']

    def __init__(self, n_groups=None, lines_table=[], styles=True):
        if not n_groups and not len(lines_table):
            raise ValueError('no instantiation parameters!')

        if len(lines_table):
            self.n_groups = len(lines_table)
            self.n_lines = list(map(len, lines_table))
        else:
            self.n_groups = n_groups
            self.n_lines = list()

        if self.n_groups > len(GroupPlotColors.linestyles):
            print('warning: this many groups requires cycling the linestyles')
        if styles:
            try:
                self._linestyles = cycle(styles)
            except:
                styles = GroupPlotColors.linestyles
                self._linestyles = cycle(styles)
        # shave off the extreme ends
        # self._hue_idx = np.linspace(0, 1, self.n_groups+2)[1:-1]
        self._hue_idx = np.linspace(0, 0.8, self.n_groups)
        self._g_count = 0
        self.styles = len(styles) > 0

    def next(self, n_lines=None):
        g = self._g_count
        if len(self.n_lines):
            n_lines = self.n_lines[g]
        h = self._hue_idx[g]
        self._g_count += 1

        # divide brightness levels over 2 values of saturation,
        # decreasing in brightness/saturation

        v = np.linspace(0.5, 1.0, max(2, int(n_lines / 2.0 + 0.5)))
        s = np.tile(np.array([[0.7], [1.0]]), (1, len(v)))
        v = np.tile(v, (2, len(v)))
        print(s.ravel()[::-1][:n_lines])
        print(v.ravel()[::-1][:n_lines])

        hsv = np.zeros((1, n_lines, 3))
        hsv[..., 0] = h
        hsv[..., 1] = s.ravel()[::-1][:n_lines]
        hsv[..., 2] = v.ravel()[::-1][:n_lines]
        rgb = colors.hsv_to_rgb(hsv)[0]
        if self.styles:
            return rgb, next(self._linestyles)
        return rgb

