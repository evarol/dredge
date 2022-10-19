"""LFP registration code

This file is organized in 3 sections:
    - Library: code for LFP raster, computing pairwise displacement
      and corr estimates, and solving for displacement vector
    - Rigid: a rigid registration function.
    - Nonrigid: Libraries for nonrigid reg and a function for users to call.

Usage:
    - Pick the channels in your recording you want to keep.
    - Use those to call `lfpraster(...)`
    - Use the output of `lfpraster` to call either `register_rigid`
      or `register_nonrigid`
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from tqdm.auto import trange

from .ibme_corr import calc_corr_decent, psolvecorr, online_register_rigid
from .ibme import compose_shifts_in_orig_domain


# -- rigid registration


def channels_to_microns(p, geom):
    """
    Convert a displacement estimate `p` from units of channels
    to microns using the geometry array `geom`.

    This just attempts to find the pitch of the probe and
    multiplies by that number.

    Arguments
    ---------
    p : np.array
        A displacement estimate.
    geom : np.array
        The geometry of the recording or raster used to compute `p`

    Returns
    -------
    p_um : np.array
    """
    # find the pitch
    y = np.sort(geom[:, 1])
    dy = np.diff(y)
    pitch = dy[dy > 0].min()

    # scale and return
    p_um = pitch * p
    return p_um


def register_rigid(
    raster,
    mincorr=0.7,
    disp=None,
    batch_size=32,
    step_size=1,
    adaptive_mincorr_percentile=None,
    prior_lambda=0,
):
    """Rigid LFP registration

    Arguments
    ---------
    raster : array (D, T)
    mincorr : float
        Correlation threshold
    disp : int, optional
        Maximum displacement during pairwise displacement estimates.
        If `None`, half of the depth domain's length will be used.
    batch_size, step_size : int
        See `calc_corr_decent`
    Returns: p, array (T,)
    """
    D, C = calc_corr_decent(
        raster,
        disp=disp,
        batch_size=batch_size,
        step_size=step_size,
    )
    if adaptive_mincorr_percentile is not None:
        mincorr = np.percentile(np.diagonal(C, 1), adaptive_mincorr_percentile)
    p = psolvecorr(D, C, mincorr=mincorr, prior_lambda=prior_lambda)
    return p


def rigid_registered_raster(raster, p):
    """Apply your displacement vector to the raster."""
    D, T = raster.shape
    total_shift = np.broadcast_to(p[None, :], (D, T))
    depth_domain = np.arange(D)
    time_domain = np.arange(T)
    dd, tt = np.meshgrid(depth_domain, time_domain, indexing="ij")
    raster_lerp = RectBivariateSpline(
        depth_domain,
        time_domain,
        raster,
        kx=1,
        ky=1,
    )
    uu = (dd + total_shift).ravel()
    return raster_lerp(uu, tt.ravel(), grid=False).reshape(D, T)


# -- nonrigid registration


def register_nonrigid(
    raster,
    mincorr=0.7,
    disp=None,
    n_windows=5,
    widthmul=0.5,
    batch_size=32,
    step_size=1,
    rigid_init=True,
):
    """Nonrigid LFP registration

    Returns:
    ps : array of shape (n_windows, T)
        Displacement vector in each window
    registered raster : depth x time array
        The interpolated raster. Just uses simple linear interpolation
        without much attention to the padding, so it is more of a
        diagnostic.
    total_shift: depth x time array
        The estimated displacement at each depth and time.
    """

    # 1. initialize displacement map with rigid reg result if requested
    total_shift = np.zeros_like(raster)
    if rigid_init:
        p = register_rigid(
            raster,
            mincorr=mincorr,
            disp=disp,
            batch_size=batch_size,
            step_size=step_size,
        )
        total_shift[:, :] = p[None, :]

    # interpolation object for image warping later
    D, T = raster.shape
    depth_domain = np.arange(raster.shape[0])
    time_domain = np.arange(T)
    dd, tt = np.meshgrid(depth_domain, time_domain, indexing="ij")
    raster_lerp = RectBivariateSpline(
        depth_domain,
        time_domain,
        raster,
        kx=1,
        ky=1,
    )

    # sensible default max disp
    disp = disp if disp is not None else D // 2

    # 2. apply shift to raster
    uu = (dd + total_shift).ravel()
    raster_ = raster_lerp(uu, tt.ravel(), grid=False).reshape(D, T)
    D_ = raster_.shape[0]

    # make gaussian windows (truncated where too small)
    windows = np.empty((n_windows, D))
    slices = []
    space = D // (n_windows + 1)
    locs = np.linspace(space, D - space, n_windows)
    scale = widthmul * D / n_windows
    for k, loc in enumerate(locs):
        windows[k, :] = norm.pdf(np.arange(D), loc=loc, scale=scale)
        domain_large_enough = np.flatnonzero(windows[k, :] > 1e-5)
        slices.append(slice(domain_large_enough[0], domain_large_enough[-1]))
    windows /= windows.sum(axis=0, keepdims=True)

    # 3. estimate each window's displacement
    ps = np.empty((n_windows, T))
    for k in trange(n_windows, desc="windows"):
        ps[k] = register_rigid(
            (windows[k, :, None] * raster_)[slices[k]],
            mincorr=mincorr,
            disp=min(2 * D_ / n_windows, disp),
            batch_size=batch_size,
            step_size=step_size,
        )

    # 4. update displacement map
    dispmap = windows.T @ ps
    total_shift[...] = compose_shifts_in_orig_domain(total_shift, dispmap)

    # 2'. apply shift to raster for caller's convenience
    uu = (dd + total_shift).ravel()
    registered_raster = raster_lerp(uu, tt.ravel(), grid=False).reshape(D, T)

    return ps, registered_raster, total_shift
