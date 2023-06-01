import numpy as np
import scipy.linalg as la
import torch
import torch.nn.functional as F
from scipy.linalg import solve
from tqdm.auto import trange

from .motion_util import (
    spike_raster,
    get_bins,
    get_motion_estimate,
    get_window_domains,
)


default_raster_kw = dict(
    amp_scale_fn=None,
    post_transform=np.log1p,
    gaussian_smoothing_sigma_um=1,
)


# -- linear algebra, Newton method solver, block tridiagonal (Thomas) solver

def laplacian(n, wink=True, eps=1e-3, lambd=1.0):
    lap = (lambd + eps) * np.eye(n)
    if wink:
        lap[0, 0] -= 0.5 * lambd
        lap[-1, -1] -= 0.5 * lambd
    lap -= np.diag(0.5 * lambd * np.ones(n - 1), k=1)
    lap -= np.diag(0.5 * lambd * np.ones(n - 1), k=-1)
    return lap


def neg_hessian_likelihood_term(Ub, Ub_prevcur=None, Ub_curprev=None):
    # negative Hessian of p(D | p) inside a block
    negHUb = -Ub.copy()
    negHUb -= Ub.T
    diagonal_terms = np.diagonal(negHUb) + Ub.sum(1) + Ub.sum(0)
    if Ub_prevcur is None:
        np.fill_diagonal(negHUb, diagonal_terms)
    else:
        diagonal_terms += Ub_prevcur.sum(0) + Ub_curprev.sum(1)
        np.fill_diagonal(negHUb, diagonal_terms)
    return negHUb


def newton_rhs(
    Db,
    Ub,
    Pb_prev=None,
    Db_prevcur=None,
    Ub_prevcur=None,
    Db_curprev=None,
    Ub_curprev=None,
):
    UDb = Ub * Db
    grad_at_0 = UDb.sum(1) - UDb.sum(0)

    # batch case
    if Pb_prev is None:
        return grad_at_0

    # online case
    align_term = (Ub_prevcur.T + Ub_curprev) @ Pb_prev
    rhs = (
        align_term
        + grad_at_0
        + (Ub_curprev * Db_curprev).sum(1)
        - (Ub_prevcur * Db_prevcur).sum(0)
    )

    return rhs


def newton_solve_rigid(
    D,
    U,
    Sigma0inv,
    Pb_prev=None,
    Db_prevcur=None,
    Ub_prevcur=None,
    Db_curprev=None,
    Ub_curprev=None,
):
    """D is TxT displacement, U is TxT subsampling or soft weights matrix"""
    negHU = neg_hessian_likelihood_term(
        U,
        Ub_prevcur=Ub_prevcur,
        Ub_curprev=Ub_curprev,
    )
    targ = newton_rhs(
        D,
        U,
        Pb_prev=Pb_prev,
        Db_prevcur=Db_prevcur,
        Ub_prevcur=Ub_prevcur,
        Db_curprev=Db_curprev,
        Ub_curprev=Ub_curprev,
    )
    p = solve(Sigma0inv + negHU, targ, assume_a="pos")
    return p, negHU


def thomas_solve(
    Ds,
    Us,
    lambda_t=1.0,
    lambda_s=1.0,
    eps=1e-4,
    P_prev=None,
    Ds_prevcur=None,
    Us_prevcur=None,
    Ds_curprev=None,
    Us_curprev=None,
):
    """Block tridiagonal algorithm, special cased to our setting

    This code solves for the displacement estimates across the nonrigid windows,
    given blockwise, pairwise (BxTxT) displacement and weights arrays `Ds` and `Us`.

    If `lambda_t>0`, a temporal prior is applied to "fill the gaps", effectively
    interpolating through time to avoid artifacts in low-signal areas. Setting this
    to 0 can lead to numerical warnings and should be done with care.

    If `lambda_s>0`, a spatial prior is applied. This can help fill gaps more
    meaningfully in the nonrigid case, using information from the neighboring nonrigid
    windows to inform the estimate in an untrusted region of a given window.

    If arguments `P_prev,Ds_prevcur,Us_prevcur` are supplied, this code handles the
    online case. The return value will be the new chunk's displacement estimate,
    solving the online registration problem.
    """
    Ds = np.asarray(Ds, dtype=np.float64)
    Us = np.asarray(Us, dtype=np.float64)
    online = P_prev is not None
    online_kw_rhs = online_kw_hess = lambda b: {}
    if online:
        assert Ds_prevcur is not None
        assert Us_prevcur is not None
        online_kw_rhs = lambda b: dict(  # noqa
            Pb_prev=P_prev[b].astype(np.float64),
            Db_prevcur=Ds_prevcur[b].astype(np.float64),
            Ub_prevcur=Us_prevcur[b].astype(np.float64),
            Db_curprev=Ds_curprev[b].astype(np.float64),
            Ub_curprev=Us_curprev[b].astype(np.float64),
        )
        online_kw_hess = lambda b: dict(  # noqa
            Ub_prevcur=Us_prevcur[b].astype(np.float64),
            Ub_curprev=Us_curprev[b].astype(np.float64),
        )

    B, T, T_ = Ds.shape
    assert T == T_
    assert Us.shape == Ds.shape
    # temporal prior matrix
    L_t = laplacian(T, eps=eps, lambd=lambda_t)
    extra = dict(L_t=L_t)

    # just solve independent problems when there's no spatial regularization
    # not that there's much overhead to the backward pass etc but might as well
    if B == 1 or lambda_s == 0:
        P = np.zeros((B, T))
        extra["HU"] = np.zeros((B, T, T))
        for b in range(B):
            P[b], extra["HU"][b] = newton_solve_rigid(
                Ds[b], Us[b], L_t, **online_kw_rhs(b)
            )
        return P, extra

    # spatial prior is a sparse, block tridiagonal kronecker product
    # the first and last diagonal blocks are
    # Lambda_s_diag0 = (lambda_s / 2) * (L_t + eps * np.eye(T))
    # the other diagonal blocks are
    Lambda_s_diag1 = (lambda_s + eps) * laplacian(T, eps=eps, lambd=1.0)
    # and the off-diagonal blocks are
    Lambda_s_offdiag = (-lambda_s / 2) * laplacian(T, eps=eps, lambd=1.0)

    # initialize block-LU stuff and forward variable
    alpha_hat_b = (
        L_t
        + Lambda_s_diag1 / 2
        + neg_hessian_likelihood_term(Us[0], **online_kw_hess(0))
    )
    targets = np.c_[Lambda_s_offdiag, newton_rhs(Us[0], Ds[0], **online_kw_rhs(0))]
    res = solve(alpha_hat_b, targets, assume_a="pos")
    # res = solve(alpha_hat_b, targets, assume_a="pos")
    assert res.shape == (T, T + 1)
    gamma_hats = [res[:, :T]]
    ys = [res[:, T]]

    # forward pass
    for b in range(1, B):
        s_factor = 1 if b < B - 1 else 0.5
        Ab = (
            L_t
            + Lambda_s_diag1 * s_factor
            + neg_hessian_likelihood_term(Us[b], **online_kw_hess(b))
        )
        alpha_hat_b = Ab - Lambda_s_offdiag @ gamma_hats[b - 1]
        targets[:, T] = newton_rhs(Us[b], Ds[b], **online_kw_rhs(b))
        targets[:, T] -= Lambda_s_offdiag @ ys[b - 1]
        res = solve(alpha_hat_b, targets)
        assert res.shape == (T, T + 1)
        gamma_hats.append(res[:, :T])
        ys.append(res[:, T])

    # back substitution
    xs = [None] * B
    xs[-1] = ys[-1]
    for b in range(B - 2, -1, -1):
        xs[b] = ys[b] - gamma_hats[b] @ xs[b + 1]

    # un-vectorize
    P = np.concatenate(xs).reshape(B, T)

    return P, extra


# -- correlation weighting and thresholding helpers


def get_weights_in_window(window, db_unreg, db_reg, raster_reg):
    ilow, ihigh = np.flatnonzero(window)[[0, -1]]
    window_sliced = window[ilow : ihigh + 1]
    tbcu = 0.5 * (db_unreg[1:] + db_unreg[:-1])
    dlow, dhigh = tbcu[[ilow, ihigh]]
    tbcr = 0.5 * (db_reg[1:] + db_reg[:-1])
    rilow, rihigh = np.flatnonzero((tbcr >= dlow) & (tbcr <= dhigh))[[0, -1]]
    return window_sliced @ raster_reg[rilow : rihigh + 1]


def get_weights(
    Ds,
    Ss,
    Sigma0inv_t,
    windows,
    amps,
    depths,
    times,
    weights_threshold_low=0.0,
    weights_threshold_high=np.inf,
    raster_kw=default_raster_kw,
    pbar=False,
):
    r, dbe, tbe = spike_raster(amps, depths, times, **raster_kw)
    assert windows.shape[1] == dbe.size - 1
    weights = []
    p_inds = []
    # start with rigid registration with weights=inf independently in each window
    for b in trange((len(Ds)), desc="Weights") if pbar else range(len(Ds)):
        # rigid motion estimate in this window
        p = newton_solve_rigid(Ds[b], Ss[b], Sigma0inv_t)[0]
        p_inds.append(p)
        me = get_motion_estimate(p, time_bin_edges_s=tbe)
        depths_reg = me.correct_s(times, depths)

        # raster just our window's bins
        # take care when tracking start/end indices of bin centers v bin edges
        ilow, ihigh = np.flatnonzero(windows[b])[[0, -1]]
        ihigh = ihigh + 1
        window_sliced = windows[b, ilow:ihigh]
        rr, dbr, tbr = spike_raster(
            amps,
            depths_reg,
            times,
            spatial_bin_edges_um=dbe[ilow : ihigh + 1],
            time_bin_edges_s=tbe,
            **raster_kw,
        )
        assert (rr.shape[0],) == window_sliced.shape
        if rr.sum() <= 0:
            raise ValueError("Convergence issue when getting weights.")
        weights.append(window_sliced @ rr)

    weights_orig = np.array(weights)

    scale_fn = raster_kw["post_transform"] or raster_kw["amp_scale_fn"]
    if isinstance(weights_threshold_low, tuple):
        nspikes_threshold_low, amp_threshold_low = weights_threshold_low
        unif = np.full_like(windows[0], 1 / len(windows[0]))
        weights_threshold_low = (
            scale_fn(amp_threshold_low) * windows @ (nspikes_threshold_low * unif)
        )
        weights_threshold_low = weights_threshold_low[:, None]
    if isinstance(weights_threshold_high, tuple):
        nspikes_threshold_high, amp_threshold_high = weights_threshold_high
        unif = np.full_like(windows[0], 1 / len(windows[0]))
        weights_threshold_high = (
            scale_fn(amp_threshold_high) * windows @ (nspikes_threshold_high * unif)
        )
        weights_threshold_high = weights_threshold_high[:, None]
    weights_thresh = weights_orig.copy()
    weights_thresh[weights_orig < weights_threshold_low] = 0
    weights_thresh[weights_orig > weights_threshold_high] = np.inf

    return weights, weights_thresh, p_inds


def threshold_correlation_matrix(
    Cs,
    mincorr=0.0,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=20,
    max_dt_s=0,
    in_place=False,
    bin_s=1,
    T=None,
    soft=True,
):
    if mincorr_percentile is not None:
        diags = [
            np.diagonal(Cs, offset=j, axis1=1, axis2=2).ravel()
            for j in range(1, mincorr_percentile_nneighbs)
        ]
        mincorr = np.percentile(
            np.concatenate(diags),
            mincorr_percentile,
        )

    # need abs to avoid -0.0s which cause numerical issues
    if in_place:
        Ss = Cs
        if soft:
            Ss[Ss < mincorr] = 0
        else:
            Ss = (Ss >= mincorr).astype(Cs.dtype)
        np.square(Ss, out=Ss)
    else:
        if soft:
            Ss = np.square((Cs >= mincorr) * Cs)
        else:
            Ss = (Cs >= mincorr).astype(Cs.dtype)
    if max_dt_s is not None and max_dt_s > 0 and T is not None and max_dt_s < T:
        mask = la.toeplitz(
            np.r_[
                np.ones(int(max_dt_s // bin_s), dtype=Ss.dtype),
                np.zeros(T - int(max_dt_s // bin_s), dtype=Ss.dtype),
            ]
        )
        Ss *= mask[None]
    return Ss, mincorr


def weight_correlation_matrix(
    Ds,
    Cs,
    amps,
    depths_um,
    times_s,
    windows,
    mincorr=0.0,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=20,
    max_dt_s=None,
    lambda_t=1,
    eps=1e-10,
    do_window_weights=True,
    weights_threshold_low=0.0,
    weights_threshold_high=np.inf,
    raster_kw=default_raster_kw,
    pbar=True,
):
    extra = {}
    bin_s = raster_kw["bin_s"]
    bin_um = raster_kw["bin_um"]

    # TODO: upsample_to_histogram_bin
    # handle shapes and the rigid case
    Ds = np.asarray(Ds)
    Cs = np.asarray(Cs)
    if Ds.ndim == 2:
        Ds = Ds[None]
        Cs = Cs[None]
    B, T, T_ = Ds.shape
    assert T == T_
    assert Ds.shape == Cs.shape
    spatial_bin_edges_um, time_bin_edges_s = get_bins(depths_um, times_s, bin_um, bin_s)
    assert (T + 1,) == time_bin_edges_s.shape
    extra = {}

    Ss, mincorr = threshold_correlation_matrix(
        Cs,
        mincorr=mincorr,
        mincorr_percentile=mincorr_percentile,
        mincorr_percentile_nneighbs=mincorr_percentile_nneighbs,
        max_dt_s=max_dt_s,
        bin_s=bin_s,
        T=T,
    )
    extra["S"] = Ss
    extra["mincorr"] = mincorr

    if not do_window_weights:
        return Ss, extra

    # get weights
    L_t = lambda_t * laplacian(T, eps=max(1e-5, eps))
    weights_orig, weights_thresh, Pind = get_weights(
        Ds,
        Ss,
        L_t,
        windows,
        amps,
        depths_um,
        times_s,
        weights_threshold_low=weights_threshold_low,
        weights_threshold_high=weights_threshold_high,
        raster_kw=raster_kw,
        pbar=pbar,
    )
    extra["weights_orig"] = weights_orig
    extra["weights_thresh"] = weights_thresh
    extra["Pind"] = Pind

    # update noise model. we deliberately divide by zero and inf here.
    with np.errstate(divide="ignore"):
        invW = 1.0 / weights_thresh
        invWbtt = invW[:, :, None] + invW[:, None, :]
        Us = np.abs(1.0 / (invWbtt + 1.0 / Ss))
    extra["U"] = Us

    return Us, extra


# -- cross-correlation tools


default_xcorr_kw = dict(
    centered=True,
    normalized=True,
)


def xcorr_windows(
    raster_a,
    windows,
    spatial_bin_edges_um,
    win_scale_um,
    raster_b=None,
    rigid=False,
    bin_um=1,
    max_disp_um=None,
    pbar=True,
    xcorr_kw=default_xcorr_kw,
    device=None,
):
    if xcorr_kw is None:
        xcorr_kw = default_xcorr_kw
    else:
        xcorr_kw = default_xcorr_kw | xcorr_kw

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if max_disp_um is None:
        if rigid:
            max_disp_um = int(spatial_bin_edges_um.ptp() // 4)
        else:
            max_disp_um = int(win_scale_um // 4)

    max_disp_bins = int(max_disp_um // bin_um)
    slices = get_window_domains(windows)
    B, D = windows.shape
    D_, T0 = raster_a.shape
    assert D == D_

    # torch versions on device
    windows_ = torch.as_tensor(windows, dtype=torch.float, device=device)
    raster_a_ = torch.as_tensor(raster_a, dtype=torch.float, device=device)
    if raster_b is not None:
        assert raster_b.shape[0] == D
        T1 = raster_b.shape[1]
        raster_b_ = torch.as_tensor(raster_b, dtype=torch.float, device=device)
    else:
        T1 = T0
        raster_b_ = raster_a_

    # estimate each window's displacement
    Ds = np.empty((B, T0, T1), dtype=np.float32)
    Cs = np.empty((B, T0, T1), dtype=np.float32)
    block_iter = trange(B, desc="Cross correlation") if pbar else range(B)
    for b in block_iter:
        window = windows_[b]

        # we search for the template (windowed part of raster a)
        # within a larger-than-the-window neighborhood in raster b
        targ_low = slices[b].start - max_disp_bins
        b_low = max(0, targ_low)
        targ_high = slices[b].stop + max_disp_bins
        b_high = min(D, targ_high)
        padding = max(b_low - targ_low, targ_high - b_high)

        # arithmetic to compute the lags in um corresponding to
        # corr argmaxes
        n_left = padding + slices[b].start - b_low
        n_right = padding + b_high - slices[b].stop
        poss_disp = -np.arange(-n_left, n_right + 1) * bin_um

        Ds[b], Cs[b] = calc_corr_decent_pair(
            raster_a_[slices[b]],
            raster_b_[b_low:b_high],
            weights=window[slices[b]],
            disp=padding,
            possible_displacement=poss_disp,
            device=device,
            **xcorr_kw,
        )

    return Ds, Cs, max_disp_um


def calc_corr_decent_pair(
    raster_a,
    raster_b,
    weights=None,
    disp=None,
    batch_size=32,
    normalized=True,
    centered=True,
    possible_displacement=None,
    device=None,
):
    """Calculate TxT normalized xcorr and best displacement matrices
    Given a DxT raster, this computes normalized cross correlations for
    all pairs of time bins at offsets in the range [-disp, disp], by
    increments of step_size. Then it finds the best one and its
    corresponding displacement, resulting in two TxT matrices: one for
    the normxcorrs at the best displacement, and the matrix of the best
    displacements.

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
    D, Ta = raster_a.shape
    D_, Tb = raster_b.shape

    # sensible default: at most half the domain.
    if disp is None:
        disp == D // 2

    # range of displacements
    if D == D_:
        if possible_displacement is None:
            possible_displacement = np.arange(-disp, disp + 1)
    else:
        assert possible_displacement is not None
        assert disp is not None

    # pick torch device if unset
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # process rasters into the tensors we need for conv2ds below
    raster_a = torch.as_tensor(raster_a, dtype=torch.float32, device=device).T
    # normalize over depth for normalized (uncentered) xcorrs
    raster_b = torch.as_tensor(raster_b, dtype=torch.float32, device=device).T

    D = np.empty((Ta, Tb), dtype=np.float32)
    C = np.empty((Ta, Tb), dtype=np.float32)
    for i in range(0, Tb, batch_size):
        corr = normxcorr1d(
            raster_a,
            raster_b[i : i + batch_size],
            weights=weights,
            padding=disp,
            normalized=normalized,
            centered=centered,
        )
        max_corr, best_disp_inds = torch.max(corr, dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[:, i : i + batch_size] = best_disp.T
        C[:, i : i + batch_size] = max_corr.cpu().T

    return D, C


def normxcorr1d(
    template,
    x,
    weights=None,
    centered=True,
    normalized=True,
    padding="same",
    conv_engine="torch",
):
    """normxcorr1d: Normalized cross-correlation, optionally weighted

    The API is like torch's F.conv1d, except I have accidentally
    changed the position of input/weights -- template acts like weights,
    and x acts like input.

    Returns the cross-correlation of `template` and `x` at spatial lags
    determined by `mode`. Useful for estimating the location of `template`
    within `x`.

    This might not be the most efficient implementation -- ideas welcome.
    It uses a direct convolutional translation of the formula
        corr = (E[XY] - EX EY) / sqrt(var X * var Y)

    This also supports weights! In that case, the usual adaptation of
    the above formula is made to the weighted case -- and all of the
    normalizations are done per block in the same way.

    Arguments
    ---------
    template : tensor, shape (num_templates, length)
        The reference template signal
    x : tensor, 1d shape (length,) or 2d shape (num_inputs, length)
        The signal in which to find `template`
    weights : tensor, shape (length,)
        Will use weighted means, variances, covariances if supplied.
    centered : bool
        If true, means will be subtracted (per weighted patch).
    normalized : bool
        If true, normalize by the variance (per weighted patch).
    padding : int, optional
        How far to look? if unset, we'll use half the length
    conv_engine : string, one of "torch", "numpy"
        What library to use for computing cross-correlations.
        If numpy, falls back to the scipy correlate function.

    Returns
    -------
    corr : tensor
    """
    if conv_engine == "torch":
        conv1d = F.conv1d
        npx = torch
    elif conv_engine == "numpy":
        conv1d = scipy_conv1d
        npx = np
    else:
        raise ValueError(f"Unknown conv_engine {conv_engine}")

    x = npx.atleast_2d(x)
    num_templates, lengtht = template.shape
    num_inputs, lengthx = x.shape

    # generalize over weighted / unweighted case
    device_kw = {} if conv_engine == "numpy" else dict(device=x.device)
    onesx = npx.ones((1, 1, lengthx), dtype=x.dtype, **device_kw)
    no_weights = weights is None
    if no_weights:
        weights = npx.ones((1, 1, lengtht), dtype=x.dtype, **device_kw)
        wt = template[:, None, :]
    else:
        assert weights.shape == (lengtht,)
        weights = weights[None, None]
        wt = template[:, None, :] * weights

    # conv1d valid rule:
    # (B,1,L),(O,1,L)->(B,O,L)

    # compute expectations
    # how many points in each window? seems necessary to normalize
    # for numerical stability.
    Nx = conv1d(onesx, weights, padding=padding)
    if centered:
        Et = conv1d(onesx, wt, padding=padding)
        Et /= Nx
        Ex = conv1d(x[:, None, :], weights, padding=padding)
        Ex /= Nx

    # compute (weighted) covariance
    # important: the formula E[XY] - EX EY is well-suited here,
    # because the means are naturally subtracted correctly
    # patch-wise. you couldn't pre-subtract them!
    cov = conv1d(x[:, None, :], wt, padding=padding)
    cov /= Nx
    if centered:
        cov -= Ex * Et

    # compute variances for denominator, using var X = E[X^2] - (EX)^2
    if normalized:
        var_template = conv1d(
            onesx, wt * template[:, None, :], padding=padding
        )
        var_template /= Nx
        var_x = conv1d(npx.square(x)[:, None, :], weights, padding=padding)
        var_x /= Nx
        if centered:
            var_template -= npx.square(Et)
            var_x -= npx.square(Ex)

        # fill in zeros to avoid problems when dividing
        var_template[var_template == 0] = 1
        var_x[var_x == 0] = 1

    # now find the final normxcorr
    corr = cov  # renaming for clarity
    if normalized:
        corr /= npx.sqrt(var_x)
        corr /= npx.sqrt(var_template)

    return corr


def scipy_conv1d(input, weights, padding="valid"):
    """SciPy translation of torch F.conv1d"""
    from scipy.signal import correlate

    n, c_in, length = input.shape
    c_out, in_by_groups, kernel_size = weights.shape
    assert in_by_groups == c_in == 1

    if padding == "same":
        mode = "same"
        length_out = length
    elif padding == "valid":
        mode = "valid"
        length_out = length - 2 * (kernel_size // 2)
    elif isinstance(padding, int):
        mode = "valid"
        input = np.pad(
            input, [*[(0, 0)] * (input.ndim - 1), (padding, padding)]
        )
        length_out = length - (kernel_size - 1) + 2 * padding
    else:
        raise ValueError(f"Unknown padding {padding}")

    output = np.zeros((n, c_out, length_out), dtype=input.dtype)
    for m in range(n):
        for c in range(c_out):
            output[m, c] = correlate(input[m, 0], weights[c, 0], mode=mode)

    return output
