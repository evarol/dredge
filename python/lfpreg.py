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
import gc
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.stats import norm, zscore
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import trange, tqdm

from pixelCSD import pixelCSD


# -- library functions


def lfpraster(
    lfp, geom, channels=None, decorr_iters=4, csd=False
):
    """LFP raster

    Grabs some channels, applies zscore decorrelation/standardization iters,
    and averages at each vertical position along the probe. I.e., for each
    unique value along the geometry `geom`'s vertical axis (restricted to those
    channels in `channels`), we average the signal in the good channels of
    `lfp`.

    Arguments
    ---------
    lfp : array (or memmap as the case may be) of shape (C, T)
    geom : array of shape (C, 2)
        Each row of this matrix is an (x, z) or (horizontal, vertical)
        coordinate.
    channels : array or indexing object, subset of range(C)
        The channels that should be used during registration
    decorr_iters : int
        The number of alternating zscores to apply. Set to 0 to skip this step.

    Returns:
    --------
    raster : array of shape (# of unique values in geom[channels, 1], T)
    """
    C, T = lfp.shape
    assert geom.shape == (C, 2)

    # -- grab good channels from geometry
    if channels is None:
        channels = range(C)
    geom = geom[channels]
    print("orig shape", lfp.shape)
    data = lfp[channels]
    print("sub shape", lfp.shape)

    # -- apply CSD
    if csd:
        data, y_locs = pixelCSD(data, geom)

    # -- decorrelation / standardization iters
    for _ in range(decorr_iters):
        data = zscore(data, axis=0)
        data = zscore(data, axis=1)

    # -- average at each z
    if csd:
        # pixelCSD has already averaged for us
        raster = data
    else:
        z_unique = np.unique(geom[:, 1])
        raster = np.empty((len(z_unique), T), dtype=data.dtype)
        for i, z in enumerate(tqdm(z_unique, desc="averaging each z")):
            where = np.flatnonzero(geom[:, 1] == z)
            raster[i] = data[where].mean(axis=0)

    return raster


def psolvecorr(D, C, mincorr=0.7):
    """Solve for rigid displacement given pairwise disps + corrs"""
    T = D.shape[0]
    assert (T, T) == D.shape == C.shape

    # subsample where corr > mincorr
    S = C >= mincorr
    I, J = np.where(S == 1)
    n_sampled = I.shape[0]

    # construct Kroneckers
    ones = np.ones(n_sampled)
    M = sparse.csr_matrix((ones, (range(n_sampled), I)), shape=(n_sampled, T))
    N = sparse.csr_matrix((ones, (range(n_sampled), J)), shape=(n_sampled, T))

    # solve sparse least squares problem
    p, *_ = sparse.linalg.lsqr(M - N, D[I, J])
    return p


def calc_corr_decent(
    raster, disp=None, batch_size=32, step_size=1, device=None
):
    """Calculate TxT normalized xcorr and best displacement matrices

    Given a DxT raster, this computes normalized cross correlations for
    all pairs of time bins at offsets in the range [-disp, disp], by
    increments of step_size. Then it finds the best one and its
    corresponding displacement, resulting in two TxT matrices: one for
    the normxcorrs at the best displacement, and the matrix of the best
    displacements.

    Note the correlations are normalized but not centered (no mean is
    subtracted).

    Arguments
    ---------
    raster : DxT array
    batch_size : int
        How many raster rows to xcorr against the whole raster
        at once.
    step_size : int
        Displacement increment. Not implemented yet but easy to do.
    disp : int
        Maximum displacement
    device : torch device

    Returns: D, C: TxT arrays
    """
    # this is not implemented but could be done easily via stride
    if step_size > 1:
        raise NotImplementedError(
            "Have not implemented step_size > 1 yet, reach out if wanted"
        )

    D, T = raster.shape

    # sensible default: at most half the domain.
    disp = disp or D // 2
    assert disp > 0

    # pick torch device if unset
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # range of displacements
    possible_displacement = np.arange(-disp, disp + step_size, step_size)

    # process raster into the tensors we need for conv2ds below
    raster = torch.tensor(
        raster.T, dtype=torch.float32, device=device, requires_grad=False
    )
    # normalize over depth for normalized (uncentered) xcorrs
    raster /= torch.sqrt((raster ** 2).sum(dim=1, keepdim=True))
    # conv weights
    image = raster[:, None, None, :]  # T11D - NCHW
    weights = image  # T11D - OIHW

    D = np.empty((T, T), dtype=np.float32)
    C = np.empty((T, T), dtype=np.float32)
    for i in trange((T + 1) // batch_size):
        batch = image[i * batch_size : (i + 1) * batch_size]
        corr = F.conv2d(  # BT1P
            batch,  # B11D
            weights,
            padding=[0, possible_displacement.size // 2],
        )
        max_corr, best_disp_inds = torch.max(corr[:, :, 0, :], dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[i * batch_size : (i + 1) * batch_size] = best_disp
        C[i * batch_size : (i + 1) * batch_size] = max_corr.cpu()

    # free GPU memory (except torch drivers... happens when process ends)
    del raster, corr, batch, max_corr, best_disp_inds, image, weights
    gc.collect()
    torch.cuda.empty_cache()

    return D, C


# -- rigid registration


def register_rigid(
    raster,
    mincorr=0.7,
    disp=None,
    batch_size=32,
    step_size=1,
):
    """Rigid LFP registration

    A simple wrapper function taking in the output of the function
    `lfpraster` defined above, and returning a vector of decentralized
    displacement outputs.

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
    p = psolvecorr(D, C, mincorr=mincorr)
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


def compose_shifts_in_orig_domain(orig_shift, new_shift):
    """
    Compose two displacement maps such that the result can be applied
    to points in the domain of the first.

    If `orig_shift` is f(x, t) and `new_shift` is g(x', t),
    this returns h(x, t) = f(x, t) + g(f(x, t) + x, t)
    """
    D, T = orig_shift.shape
    D_, T_ = new_shift.shape
    assert T == T_

    orig_depth_domain = np.arange(D, dtype=float)
    g_domain = np.arange(D_, dtype=float)
    time_domain = np.arange(T, dtype=float)

    x_plus_f = orig_depth_domain[:, None] + orig_shift

    g_lerp = RectBivariateSpline(
        g_domain,
        time_domain,
        new_shift,
        kx=1,
        ky=1,
    )
    h = g_lerp(x_plus_f.ravel(), np.tile(time_domain, D), grid=False)

    return orig_shift + h.reshape(orig_shift.shape)


def gaussian_windows(nwin, D, widthmul):
    """Make nwin Gaussian windows

    D: domain length, widthmul: controls the scale / rolloff.
    """
    windows = np.empty((nwin, D))
    pad = D // (nwin + 1)
    locs = np.linspace(pad, D - pad, nwin)
    scale = widthmul * D / nwin
    scale -= scale % 2
    dd = np.arange(D)
    for k, loc in enumerate(locs):
        windows[k, :] = norm.pdf(dd, loc=loc, scale=scale)
    # make it row stochastic instead of col stochastic
    windows /= windows.sum(axis=0, keepdims=True)
    return windows


def register_nonrigid(
    raster,
    mincorr=0.7,
    disp=None,
    n_windows=[5],
    widthmul=0.25,
    batch_size=32,
    step_size=1,
):
    """Nonrigid LFP registration"""

    # 1. initialize displacement map with rigid reg result
    total_shift = np.empty_like(raster)
    p = register_rigid(
        raster,
        mincorr=mincorr,
        disp=disp,
        batch_size=batch_size,
        step_size=step_size,
    )
    total_shift[:, :] = p[None, :]

    # for image warping later
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

    # sensible max disp
    disp = disp or D // 2
    assert disp > 0

    for nwin in tqdm(n_windows):
        # 2. apply shift to raster
        uu = (dd + total_shift).ravel()
        raster_ = raster_lerp(uu, tt.ravel(), grid=False).reshape(D, T)
        D_ = raster_.shape[0]

        # make gaussian windows
        windows = gaussian_windows(nwin, D_, widthmul)

        # 3. estimate each window's displacement
        ps = np.empty((nwin, T))
        for k, window in enumerate(tqdm(windows, desc="windows")):
            ps[k] = register_rigid(
                window[:, None] * raster_,
                mincorr=mincorr,
                disp=min(2 * D_ / nwin, disp),
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
