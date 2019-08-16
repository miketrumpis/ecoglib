import numpy as np
import ecogana.anacode.spatial_profiles as sp
from ecoglib.numutil import fenced_out
from scipy.optimize import brent, minimize_scalar

def auto_tune_lambda(Kg, Kt, scl=1.0):
    var_x = Kg.diagonal()
    scl = np.median( Kt.diagonal() ) / np.median(var_x)
    print(np.median( Kt.diagonal() ), np.median(var_x))
    if (var_x[0] == var_x).all():
        inliers = np.ones( len(var_x), '?' )
    else:
        inliers = fenced_out(np.log(var_x))
    
    w = np.linalg.eigvalsh(Kg)
    lam = w.max() / 1000

    def _loss(lam):
        Kg_i = np.linalg.inv(Kg + np.eye(len(Kg)) * lam)
        C_eta = Kt.dot(Kg_i).dot(Kt)
        if Kt.ndim > 1:
            var_y = C_eta.diagonal()
            return np.linalg.norm(var_x[inliers] * scl - var_y[inliers])
        else:
            return np.abs( var_x[inliers].mean() * scl - C_eta )

    #r = brent(_loss, brack=[w.min(), w.max()])
    r = minimize_scalar(_loss, bracket=[w.min(), w.max()],
                        bounds=[0, w.max()], method='Bounded')
    return r.x
    
## def parametric_covariance(chan_map, channel_variance=(), **matern_params):
##     """Returns a model covariance for sites from a ChannelMap"""
##     n_chan = len(chan_map)
##     combs = chan_map.site_combinations
##     # estimated correlation matrix with nuisance terms
##     Kg_flat = sp.matern_correlation(combs.dist, **matern_params)
##     Kg = np.zeros( (n_chan, n_chan) )
##     Kg.flat[0::n_chan+1] = 0.5
##     Kg[ np.triu_indices(n_chan, k=1) ] = Kg_flat
##     Kg = Kg + Kg.T
##     if len(channel_variance):
##         cv = np.sqrt(channel_variance)
##         Kg = Kg * np.outer( cv, cv )
##     return Kg
        
def conditional_mean_filter(chan_map, pitch=None, x_samp=None, lam='auto', 
                            channel_variance=(), preconditioner=False,
                            normalize_output=False,
                            target_sites = (),
                            target_params={}, **matern_params):
    # E{y | x1, ..., xN}
    # if no target params are given, then output is effectively
    # normalized (unless channel_variance is given)
    target_params.setdefault('theta', matern_params.get('theta', 1))
    target_params.setdefault('nu', matern_params.get('nu', 1))
    n_chan = len(chan_map)
    combs = chan_map.site_combinations
    if x_samp is None:
        Kg = sp.matern_covariance_matrix(
            chan_map, channel_variance=channel_variance, **matern_params
            )
    else:
        if x_samp.shape == (n_chan, n_chan):
            Kg = x_samp
        else:
            Kg = np.cov(x_samp)
    #channel_variance = Kg.diagonal()

    # target correlation matrix
    sill = target_params.get('sill', 1.0)
    # would assume this is desired to be zero!
    nugget = target_params.get('nugget', 0.0)
    kappa = target_params.get('kappa', 0.0)
    # XXX: this doesn't work as intended when channel_variance is provided:
    # The target nugget would typically be set to zero, meaning that
    # the nominal channel variance is not scaled down.
    if normalize_output or len(channel_variance):
        all_var = sill + kappa
        target_params['sill'] = sill / all_var
        target_params['kappa'] = kappa / all_var
        target_params['nugget'] = nugget / all_var
        if len(channel_variance):
            channel_variance *= (all_var - nugget) / all_var
    if normalize_output:
        channel_variance = ()
    if not target_sites:
        # estimate y underlying all the sampling sites x
        Kt = sp.matern_covariance_matrix(
            chan_map, channel_variance=channel_variance, **target_params
            )
    else:
        # estimate y at an unknown location
        pitch = chan_map.pitch
        if pitch is None:
            raise ValueError(
                'Need to know pitch to compute interpolated covariance'
                )
        # make into sep_i, sep_j
        if not np.iterable(pitch):
            pitch = np.array( [pitch, pitch] )
        pitch = np.asarray(pitch)
        ii, jj = chan_map.to_mat()
        Kt = np.empty( (len(chan_map), len(target_sites)) )
        for n, target_site in enumerate( target_sites ):
            dy = (ii - target_site[0]) * pitch[0]
            dx = (jj - target_site[1]) * pitch[1]
            h = np.sqrt( dx**2 + dy**2 )
            Kt[:,n] = sp.matern_correlation(
                h, theta=target_params['theta'], nu=target_params['nu']
                )
        # scale (may just be identity)
        Kt = (sill - nugget) * Kt + kappa
        # don't yet allow for channel variance to be set in target
        # (that would require a different vector than the data model
        # covariance)
        # Kt.flat[::n_chan+1] = channel_variance
        
    
    if preconditioner:
        lam_x, v_x = np.linalg.eigh(Kg)
        lam_e, v_e = np.linalg.eigh(Kt)
        # Solve dot(Kg, W) = C'
        # where C' = (Cx)^1/2 (Ce)^T/2
        
        Cx_half = v_x.dot(np.diag(lam_x)**0.5)
        Ce_half = v_e.dot(np.diag(lam_e)**0.5)
        Kt = Cx_half.dot(Ce_half.T)
        lam = 0

    if not preconditioner and \
      isinstance(lam, str) and lam.lower() == 'auto':
        Kt_temp = Kt
        print(sill, nugget)
        lam = auto_tune_lambda(Kg, Kt_temp)
    W = np.linalg.solve(Kg + lam*np.eye(n_chan), Kt)
    return Kg, Kt, W.T, lam #, (dbin, Kt_lin)


def mean_estimator(
        chan_map, pitch=None, R=None, x_samp=None, lam=1e-8,
        channel_variance=(), **matern_params
        ):

    n_chan = len(chan_map)
    if x_samp is None:
        Kg = sp.matern_covariance_matrix(
            chan_map, channel_variance=channel_variance, **matern_params
            )
    else:
        if x_samp.shape == (n_chan, n_chan):
            Kg = x_samp
        else:
            Kg = np.cov(x_samp)
            
    if R is None:
        R = np.ones( (n_chan, 1) )
    
    Kg_i = np.linalg.inv(Kg + lam * np.eye(n_chan))
    M1 = np.dot( R.T, np.dot(Kg_i, R) )
    M2 = np.dot( R.T, Kg_i )
    GLS_estimator = np.linalg.solve(M1, M2)
    
    return Kg, GLS_estimator
