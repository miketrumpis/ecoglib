import numpy as np
import statsmodels.formula.api as smf
import scipy.special as spfn
from scipy.optimize import curve_fit, minimize, minimize_scalar
import math

from ecogdata.channel_map import ChannelMap

from .variogram import cxx_to_pairs, concat_bins, binned_variance


__all__ = ['matern_correlation', 'matern_spectrum', 'matern_covariance_matrix', 'matern_semivariogram',
           'simulate_matern_process', 'effective_range', 'exponential_fit']


# handcock & wallis def (also Rasmussen & Williams 2006 ch4)
def matern_correlation(x, theta=1.0, nu=0.5, d=1.0, **kwargs):
    """
    Compute the normalized Matérn correlation kernel for values in x. The parameterization here is similar to
    using the Stein model in "vgm" from gstat, with a simple scaling of the range:

    theta_py = theta_r / sqrt(2)

    Parameters
    ----------
    x: float or ndarray
        Lag values for the correlation kernel
    theta: float
        Range parameter (mm, >0)
    nu: float
        Smoothness parameter (unitless, >0)
    d: int
        Dimensionality of the domain (default 1)

    Returns
    -------
    cf: float or ndarray
        Correlation kernel C(x)

    """
    theta, nu, d = map(float, (theta, nu, d))
    if 'nugget' in kwargs or 'sill' in kwargs:
        print
        'The matern_correlation() method no longer scales past [0, 1]'
    if theta < 1e-6 or nu < 1e-2:
        # argument error, return null vector
        return np.zeros_like(x)

    def _comp(x_):
        scl = 2 ** (1.0 - nu) / math.gamma(nu)
        z = (2 * nu) ** 0.5 * x_ / theta
        return scl * (z ** nu) * spfn.kv(nu, z)

    cf = _comp(x)
    if np.iterable(cf):
        cf[np.isnan(cf)] = 1.0
    elif np.isnan(cf):
        cf = 1.0
    return cf


# from Rasmussen & Williams 2006 MIT press ch4
def matern_spectrum(k, theta=1.0, nu=0.5, d=1.0):
    """
    Return the Matérn power spectrum for unit-variance.

    Parameters
    ----------
    k: ndarray
        Vector of spatial frequencies
    theta: float
        Range parameter
    nu: float
        Smoothness parameter
    d: int
        Domain dimensionality

    Returns
    -------
    p: ndarray
        Power spectrum P(k)

    """

    const = 2 ** d * np.pi ** (d / 2.) * math.gamma(nu + d / 2.) * (2 * nu) ** nu
    const = const / math.gamma(nu) / (theta ** (2 * nu))
    spec = (2 * nu / theta ** 2 + 4 * np.pi ** 2 * k ** 2) ** (nu + d / 2.0)
    return const / spec


def matern_covariance_matrix(chan_map, channel_variance=(), **matern_params):
    """Returns a model covariance for sites from a ChannelMap

    If channel_variance is given, probably a good idea that
    sill + nugget + kappa = 1 (which is default)

    If channel_variance is not given, then the diagonal will be sill+kappa
    """
    n_chan = len(chan_map)
    combs = chan_map.site_combinations
    # estimated correlation matrix with nuisance terms
    prm = matern_params.copy()
    nugget = prm.pop('nugget', 0)
    sill = prm.pop('sill', 1)
    kappa = prm.pop('kappa', 0)
    udist = np.unique(combs.dist)
    covar_values = (sill - nugget) * matern_correlation(udist, **prm) + kappa
    udist = np.round(udist, decimals=3)
    dist_hash = dict(zip(udist, covar_values))
    Kg_flat = [dist_hash[d] for d in np.round(combs.dist, decimals=3)]
    # Kg_flat = (sill - nugget) * matern_correlation(combs.dist, **prm) + kappa
    Kg = np.zeros((n_chan, n_chan))
    Kg.flat[0::n_chan + 1] = (sill + kappa) / 2.0
    Kg[np.triu_indices(n_chan, k=1)] = Kg_flat
    Kg = Kg + Kg.T
    if len(channel_variance):
        cv = np.sqrt(channel_variance / sill)
        Kg = Kg * np.outer(cv, cv)
    return Kg


def simulate_matern_process(extent, dx, theta=1.0, nu=0.5, nugget=0, sill=1, kappa=0, nr=1, mu=(), cxx=False):
    """
    (Dense) simulation of spatial fields using Matern covariance.
    The terminology of sill is slightly adapted from typical usage.

    * The covariance function shape is parameterized by (nu, theta)
    * "nugget" codes absolute noise strength, and not a proportion
    * "kappa" is optional and represents a minimum covariance > 0
    * "sill" is slightly different than typical usage. Sill refers to
      marginal site variance (including nugget + kappa components)

    """
    # extent -- if number then generate samples along an axis spaced by dx
    #        -- if a ChannelMap, then generate correlated timeseries
    #           samples on the grid with spacing of dx
    from scipy.linalg import toeplitz
    if isinstance(extent, ChannelMap):
        chan_combs = extent.site_combinations
        h = chan_combs.dist
        nrow, ncol = extent.geometry
        nchan = len(extent)
        cx = np.zeros((nchan, nchan), 'd')
        upper = np.triu_indices(len(cx), k=1)
        cx = matern_covariance_matrix(extent, theta=theta, nu=nu, sill=sill, nugget=nugget, kappa=kappa)
        x = np.linspace(0, ncol - 1, ncol) - ncol / 2.0
        y = np.linspace(0, nrow - 1, nrow) - nrow / 2.0
        x = (x * dx, y * dx)
        if not len(mu):
            mu = np.zeros((nchan,))
    else:
        extent_ = round(extent / dx)
        x = np.arange(-extent_, extent_ + 1, dtype='d') * dx
        rx = np.abs(x[0] - x)
        # print rx
        mask = rx < 1e-3
        cm = matern_correlation(rx, theta=theta, nu=nu)
        cx = (sill - nugget) * cm + kappa
        cx[mask] = sill + kappa
        cx = toeplitz(cx)
        if not len(mu):
            mu = np.zeros_like(x)
    if cxx:
        return cx
    try:
        samps = np.random.multivariate_normal(mu, cx, nr)
    except np.linalg.LinAlgError:
        # perturb matrix slightly
        n = len(cx)
        cx.flat[::n + 1] += 1e-5
        samps = np.random.multivariate_normal(mu, cx, nr)
    if isinstance(extent, ChannelMap):
        samps = extent.embed(samps, axis=1)
    return x, samps.squeeze()


# BUG -- there is still bias when fitting a cloud with fit_mean=False using
# IRLS weights. It looks like the 1 / semivar weighting pushes the
# estimate very high (at least pushes the nugget high, if it is a free
# variable).
def matern_semivariogram(x, theta=1.0, nu=1.0, nugget=0, sill=None, y=(), free=('theta', 'nu'), wls_mode='irls',
                         fit_mean=False, binsize=None, dist_limit=None, bin_limit=None, bounds=None, weights=(),
                         fraction_nugget=False, **kwargs):
    """
    Estimates Matérn kernel parameters from (x, y) data or evaluates a kernel for x data given parameters.

    Parameters
    ----------
    x: ndarray
        Spatial lags for computing or fitting a Matérn variogram kernel.
    theta: float
        Range parameter
    nu: float
        Smoothness parameter
    nugget: float
        Size of noise (nugget). Same "units" as sill.
    sill: float
        Total process variance. The "partial" sill would be sill - nugget.
    y: ndarray
        If given, then this method estimates the kernel parameters listed in the "free" variable.
    free: sequence
        Free parameters to optimize, subset of ('theta', 'nu', 'nugget', 'sill'). Any parameter not in "free" is fixed.
    wls_mode: str
        Weighted least squares mode, default 'irls' for iteratively reweighted least squares per Stein. Use 'var' for
        traditional weights based on the variance per binned spatial lag. Use 'none' to disable weighted LS.
    fit_mean: bool
        If (x, y) is a variogram cloud, fit_mean=True computes squared error for bin means rather than all cloud points.
    binsize: float or None
        If (x, y) is a variogram cloud, bin at this approximate x spacing. If binsize=None, then binning takes place
        at natural grid spacings.
    dist_limit: float or None
        If given, restrict optimization to the variogram for x in [min(x), dist_limit * max(x)].
    bin_limit: int or None
        If given, discard bins with less than bin_limit entries.
    bounds: dict
        Lower and upper bounds (lb, ub) for any free parameter, e.g. dict(theta=(0.2, 10)). If a bound is one-sided,
        use None for the opposite bound.
    weights: ndarray
        Weights to be used for weighted LS. Typically number of points per bin for wls_mode='irls'.
    fraction_nugget: bool
        Optimize nugget to be a fraction of the sill, rather than a free number. This can be more convenient for
        defining constraints.
    kwargs: junk

    Returns
    -------
    params: dict
        The estimated values of the free parameters.

    """

    if 'ci' in kwargs.keys():
        print
        'Conf intervals not supported anymore'
    if 'wls' in kwargs.keys():
        print
        "Use wls_mode='none' to turn off weighted least squares"

    # change default sill value depending on whether this is
    # a fitting or an evaluation
    if sill is None:
        if not len(y):
            sill = 1.0
        else:
            sill = y.max()

    if not len(y):
        mf = matern_correlation(x, theta, nu)
        if fraction_nugget:
            nugget = sill * nugget
        return (sill - nugget) * (1 - mf) + nugget

    if bounds is None:
        bounds = {}
    if dist_limit is not None:
        keep = x < dist_limit * x.max()
        x = x[keep]
        y = y[keep]

    if wls_mode is None:
        wls_mode = 'none'
    if wls_mode.lower() == 'textbook':
        wls_mode = 'irls'

    # Set up data -- always "fit" only at unique x points, but
    # calculate error at
    # * all points, if fit_mean == False
    # * mean points, if fit_mean == True

    # First bin data (if given a cloud) or create pseudo-bins otherwise
    if len(x) > len(np.unique(x)):
        xbinned, ybinned = binned_variance(x, y, binsize=binsize)
    else:
        fit_mean = True
        xbinned = x.copy()
        ybinned = [np.array([yi]) for yi in y]
        # I don't think it makes sense to replicate the mean value Ni times..
        # we already have the correct bin weights with IRLS
        # if wls_mode.lower() == 'irls' and len(weights):
        #     ybinned = [ np.array([yi] * Ni) for yi, Ni in zip(y, weights) ]
        # else:
        #     ybinned = [np.array([yi]) for yi in y]

    # If weights are supplied in IRLS mode, then they are
    # the bin counts of the semivariance estimates.
    if wls_mode.lower() == 'irls' and len(weights):
        Nh = weights
        if dist_limit is not None:
            Nh = Nh[keep]
    else:
        # else get the observed bin counts (even if ones)
        Nh = np.array(map(len, ybinned), dtype='d')
    # If weights are *not* supplied in inverse-variance weighted mode,
    # then define weights
    if wls_mode.lower() == 'var' and not len(weights):
        weights = np.array(map(np.var, ybinned))
        # this corrects for single-entry bins
        weights[weights == 0] = max(1, weights.max())
    elif wls_mode.lower() == 'none':
        norm_weight = 1 / y.var()
        weights = [norm_weight] * len(xbinned)
    # This condition seems redundant with the else of "len(x) > len(np.unique(x))" from above
    if len(xbinned) == len(x):
        fit_mean = True

    if bin_limit is not None:
        keep = Nh > bin_limit
        xbinned = xbinned[keep]
        ybinned_ = [ybinned[n] for n in range(len(keep)) if keep[n]]
        ybinned = ybinned_
        Nh = Nh[keep]

    if fit_mean:
        x_, y_ = xbinned, np.array(map(np.mean, ybinned))
    else:
        # x_, y_ = concat_bins(xbinned, ybinned)
        _, y_ = concat_bins(xbinned, ybinned)
        x_ = xbinned
        # now bin-counts are redundant
        # Nh = np.ones_like(y_)
        Nh = np.ones_like(x_)

    # Set up bounds defaults
    # maximum value for nu -- the variation in chaning nu past 10 is < 1ppm
    bounds.setdefault('nu', (.1, 10))

    # ensure theta > small
    bounds.setdefault('theta', (1e-6, None))

    if fraction_nugget:
        # nugget will be fit as a proportion [0, 1)
        bounds.setdefault('nugget', (0, 1))
    else:
        bounds.setdefault('nugget', (0, None))

    # sill must be positive
    bounds.setdefault('sill', (y_.min(), None))

    cons = [dict(type='ineq', fun=lambda p: np.array(p))]
    solver_bounds = [bounds.get(f, (None, None)) for f in free]
    # print zip(free, solver_bounds)
    cons = tuple(cons)

    def _split_free_fixed(params):
        pd = dict(zip(free, params))
        # fixed is globally set
        pd.update(fixed)
        return pd

    def calc_weight(x, y):
        if wls_mode == 'irls':
            if (y == 0).all():
                w = Nh
            else:
                y = np.clip(y, y[y > 0].min(), y.max())
                w = Nh / y ** 2
        elif wls_mode == 'var':
            # not iterative (since zero variance for model values)
            # w = 1 / np.array( map(np.var, ybinned) )
            w = weights
        else:
            # w = np.ones_like(y)
            w = weights
        # w /= ( w.sum() / len(w) )
        return w

    def err(p):
        mf_params = _split_free_fixed(p)
        if fraction_nugget:
            if 'nugget' in free:
                # if optimizing for nugget, then convert to units here
                nz_prop = mf_params['nugget']
                mf_params['nugget'] = mf_params['sill'] * nz_prop
                mf_params['fraction_nugget'] = False
            else:
                # else preserve the fraction info
                mf_params['fraction_nugget'] = True
        y_est = matern_semivariogram(x_, **mf_params)
        w = calc_weight(x_, y_est)
        if not fit_mean:
            # reproduce weights and estimates
            w, _ = concat_bins(w, ybinned)
            y_est, _ = concat_bins(y_est, ybinned)
        return np.sqrt(w) * (y_ - y_est)

    def loss(p):
        mf_params = _split_free_fixed(p)
        # if nugget is in real units, then
        # return a large loss if it is larger than the sill
        if 'nugget' in free and not fraction_nugget:
            if mf_params['nugget'] >= mf_params['sill']:
                return 1e20
        resid = err(p)
        return np.sum(resid ** 2)

    if 'nugget' in free and nugget > 1 and fraction_nugget:
        # nugget = 0.01
        nugget /= sill

    fixed = dict(theta=theta, nu=nu, nugget=nugget, sill=sill)
    p0 = [fixed.pop(f) for f in free]

    r = minimize(loss, p0, constraints=cons, method='SLSQP', bounds=solver_bounds)
    params = dict(zip(free, r.x))
    if 'nugget' in params and fraction_nugget:
        sill = params.get('sill', sill)
        nz_prop = params['nugget']
        params['nugget'] = sill * nz_prop

    return params


def effective_range(p, mx):
    """
    Effective range of a Matérn variogram kernel, defined by its crossing with the value mx.

    Parameters
    ----------
    p: dict
        Parameters for `matern_semivariogram`
    mx: float
        Where to search for a line crossing.

    Returns
    -------
    x: float
        Effective range of the variogram

    """

    # theta = p[0]
    # x0 = 2*theta
    def fx(x):
        return (matern_semivariogram(x, **p) - mx) ** 2
    return minimize_scalar(fx).x


# keep this around just in case

def exponential_fit(cxx, chan_map, bin_lim=10, Tv=np.log, lsq='linear', nugget=False, sill=False, cov=False):
    if isinstance(chan_map, ChannelMap):
        dist, cxx_pairs = cxx_to_pairs(cxx, chan_map)
    else:
        cxx_pairs = cxx
        dist = chan_map

    xb, yb = binned_variance(dist, cxx_pairs)
    #T = np.log
    if Tv is None:
        def Tv(x): return x
    yv = np.array([np.var(Tv(y_)) if len(y_) > bin_lim else -1 for y_ in yb])
    yv[yv < 0] = np.max(yv)
    # need to put weights and data in the same order
    yv, _ = concat_bins(yv, yb)
    x, y = concat_bins(xb, yb)
    if lsq == 'linear':
        # model (exponential) covariance has full range (0, 1)
        # covariance measures with noise & redundancy have:
        # minimum (floor): 1-sill
        # maximum: 1-nugget
        # So scale measurements by
        # 1) subtracting (1-sill)
        # 2) expanding by (1-nugget) - (1 - sill) = 1 / (sill - nugget)
        nugget = float(nugget)
        if isinstance(sill, float):
            sill = float(sill)
        else:
            sill = 1.0
        yt = (y - 1 + sill) / (sill - nugget)
        d_obj = dict(dist=x, cxx=yt)
        wls = smf.wls(
            'I(-np.log(np.abs(cxx))) ~ 0 + dist', d_obj, weights=1/yv
        ).fit()
        lam = 1.0/float(wls.params.dist)
        return (lam, wls) if cov else lam
    else:
        # def _expdecay(x_, lam_):
        # return np.exp(-x_ / lam_)
        def _expdecay(x_, lam_, nug_=0, sill_=1):
            nug_ = nug_ if est_nugget else fixed_nugget
            sill_ = sill_ if est_sill else fixed_sill
            if lam_ < 1e-2 or (est_nugget and nug_ < 1e-6) or (est_sill and sill > 1):
                return np.ones_like(x_) * 1e5
            #lam_ = p[0]
            return (sill_ - nug_) * np.exp(-x_ / lam_) + (1-sill_)
        p0 = [1.0]
        if isinstance(nugget, bool) and nugget:
            est_nugget = True
            #p0 = [1.0, 0.5]
            p0.append(0.5)
        elif isinstance(nugget, float):
            est_nugget = False
            #p0 = [1.0]
            fixed_nugget = nugget
        else:
            est_nugget = False
            #p0 = [1.0]
            fixed_nugget = 0
        if isinstance(sill, bool) and sill:
            est_sill = True
            p0.append(1.0)
        elif isinstance(sill, float):
            est_sill = False
            fixed_sill = sill
        else:
            est_sill = False
            fixed_sill = 1.0
            print
        p, pcov = curve_fit(
            _expdecay, x, y, p0=p0, sigma=np.sqrt(yv),
            absolute_sigma=True
        )
        lam = p[0]
        return (p, pcov) if cov else p
