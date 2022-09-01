import gc
import numpy as np
import scipy.linalg as la
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import zscore
from tqdm.auto import trange


def register_raster_rigid(
    raster,
    mincorr=0.0,
    disp=None,
    robust_sigma=0.0,
    normalized=True,
    max_dt=None,
    batch_size=8,
    device=None,
    pbar=True,
):
    """Rigid correlation subsampled registration

    Arguments
    ---------
    raster : array (D, T)
    mincorr : float
        Correlation threshold
    disp : int, optional
        Maximum displacement during pairwise displacement estimates.
        If `None`, half of the depth domain's length will be used.
    batch_size : int
        See `calc_corr_decent`
    Returns: p, array (T,)
    """
    D, C = calc_corr_decent(
        raster,
        disp=disp,
        normalized=normalized,
        batch_size=batch_size,
        device=device,
        pbar=pbar,
    )
    p = psolvecorr(D, C, mincorr=mincorr, robust_sigma=robust_sigma, max_dt=max_dt)
    return p, D, C


def weighted_lsqr(Wij, Dij, I, J, T, p0):
    W = sparse.csr_matrix((Wij, (I, J)), shape=(T, T))
    WD = sparse.csr_matrix((Wij * Dij, (I, J)), shape=(T, T))
    fixed_terms = (W @ WD).diagonal() - (WD @ W).diagonal()
    diag_WW = (W @ W).diagonal()
    Wsq = W.power(2)

    def obj(p):
        return 0.5 * np.square(Wij * (Dij - (p[I] - p[J]))).sum()

    def jac(p):
        return fixed_terms - 2 * (Wsq @ p) + 2 * p * diag_WW

    res = minimize(
        fun=obj, jac=jac, x0=p0, method="L-BFGS-B"
    )
    if not res.success:
        print("Global displacement gradient descent had an error")
    p = res.x

    return p


def psolvecorr(D, C, mincorr=0.0, robust_sigma=0, robust_iter=5, max_dt=None):
    """Solve for rigid displacement given pairwise disps + corrs"""
    T = D.shape[0]
    assert (T, T) == D.shape == C.shape

    # subsample where corr > mincorr
    S = C >= mincorr
    if max_dt is not None and max_dt > 0:
        S &= la.toeplitz(
            np.r_[np.ones(max_dt, dtype=bool), np.zeros(T - max_dt, dtype=bool)]
        )
    I, J = np.where(S == 1)
    n_sampled = I.shape[0]

    # construct Kroneckers
    ones = np.ones(n_sampled)
    M = sparse.csr_matrix((ones, (range(n_sampled), I)), shape=(n_sampled, T))
    N = sparse.csr_matrix((ones, (range(n_sampled), J)), shape=(n_sampled, T))
    A = M - N
    V = D[I, J]

    # solve sparse least squares problem
    if robust_sigma is not None and robust_sigma > 0:
        idx = slice(None)
        for _ in trange(robust_iter, desc="robust lsqr"):
            p, *_ = sparse.linalg.lsqr(A[idx], V[idx])
            idx = np.flatnonzero(np.abs(zscore(A @ p - V)) <= robust_sigma)
    else:
        p, *_ = sparse.linalg.lsqr(A, V)

    return p


@torch.no_grad()
def calc_corr_decent(
    raster,
    disp=None,
    normalized=True,
    batch_size=32,
    step_size=1,
    device=None,
    pbar=True,
):
    """Calculate TxT normalized xcorr and best displacement matrices

    Given a DxT raster, this computes normalized cross correlations for
    all pairs of time bins at offsets in the range [-disp, disp]. Then it
    finds the best one and its corresponding displacement, resulting in
    two TxT matrices: one for the normxcorrs at the best displacement,
    and the matrix of the best displacements.

    Arguments
    ---------
    raster : DxT array
        Depth by time. We want to find the best spatial (depth) displacement
        for all pairs of time bins.
    disp : int
        Maximum displacement (translates to padding for xcorr)
    normalized : bool
        If True, use normalized and centered cross correlations. Otherwise,
        no normalization or centering is used.
    batch_size : int
        How many raster columns to xcorr against the whole raster at once.
    step_size : int
        Displacement increment for coarse-grained search.
        Not implemented yet but easy to do.
    device : torch.device (optional)
    pbar : bool
        Display a progress bar.

    Returns
    -------
    D : TxT array of best displacments
    C : TxT array of best xcorrs or normxcorrs
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # range of displacements
    possible_displacement = np.arange(-disp, disp + step_size, step_size)

    # process raster into the tensors we need for conv below
    # note the transpose from DxT to TxD (batches on first dimension)
    raster = torch.as_tensor(raster, dtype=torch.float32, device=device).T
    if not normalized:
        # if we're not doing full normxcorr, we still want to keep
        # the outputs between 0 and 1
        raster /= torch.sqrt((raster ** 2).sum(dim=1, keepdim=True))

    D = np.empty((T, T), dtype=np.float32)
    C = np.empty((T, T), dtype=np.float32)
    xrange = trange if pbar else range
    for i in xrange(0, T, batch_size):
        if normalized:
            corr = normxcorr(
                raster,
                raster[i : i + batch_size],
                padding=possible_displacement.size // 2,
            )
        else:
            corr = F.conv1d(
                raster[i : i + batch_size, None, :],
                raster[:, None, :],
                padding=possible_displacement.size // 2,
            )
        max_corr, best_disp_inds = torch.max(corr, dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[i : i + batch_size] = best_disp
        C[i : i + batch_size] = max_corr.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    return D, C


def normxcorr(template, x, padding=None):
    """normxcorr: Normalized cross-correlation

    Returns the cross-correlation of `template` and `x` at spatial lags
    determined by `mode`. Useful for estimating the location of `template`
    within `x`.

    This might not be the most efficient implementation -- ideas welcome.
    It uses a direct convolutional translation of the formula
        corr = (E[XY] - EX EY) / sqrt(var X * var Y)

    Arguments
    ---------
    template : tensor, shape (num_templates, length)
        The reference template signal
    x : tensor, 1d shape (length,) or 2d shape (num_inputs, length)
        The signal in which to find `template`
    padding : int, optional
        How far to look? if unset, we'll use half the length
    assume_centered : bool
        Avoid a copy if your data is centered already.

    Returns
    -------
    corr : tensor
    """
    template = torch.as_tensor(template)
    x = torch.atleast_2d(torch.as_tensor(x))
    assert x.device == template.device
    num_templates, length = template.shape
    num_inputs, length_ = template.shape
    assert length == length_

    if padding is None:
        padding = length // 2

    # compute expectations
    ones = torch.ones((1, 1, length), dtype=x.dtype, device=x.device)
    # how many points in each window? seems necessary to normalize
    # for numerical stability.
    N = F.conv1d(ones, ones, padding=padding)
    Et = F.conv1d(ones, template[:, None, :], padding=padding) / N
    Ex = F.conv1d(x[:, None, :], ones, padding=padding) / N

    # compute covariance
    corr = F.conv1d(x[:, None, :], template[:, None, :], padding=padding) / N
    corr -= Ex * Et

    # compute variances for denominator, using var X = E[X^2] - (EX)^2
    var_template = F.conv1d(
        ones, torch.square(template)[:, None, :], padding=padding
    ) / N - torch.square(Et)
    var_x = F.conv1d(
        torch.square(x)[:, None, :], ones, padding=padding
    ) / N - torch.square(Ex)

    # now find the final normxcorr and get rid of NaNs in zero-variance areas
    corr /= torch.sqrt(var_x * var_template)
    corr[~torch.isfinite(corr)] = 0

    return corr


# -- online methods: not exposed in `ibme` API yet


def calc_corr_decent_pair(
    raster_a, raster_b, disp=None, batch_size=32, step_size=1, device=None
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

    D, Ta = raster_a.shape
    D_, Tb = raster_b.shape
    assert D == D_

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

    # process rasters into the tensors we need for conv2ds below
    raster_a = torch.tensor(
        raster_a.T, dtype=torch.float32, device=device, requires_grad=False
    )
    # normalize over depth for normalized (uncentered) xcorrs
    raster_a /= torch.sqrt((raster_a ** 2).sum(dim=1, keepdim=True))
    image = raster_a[:, None, None, :]  # T11D - NCHW
    raster_b = torch.tensor(
        raster_b.T, dtype=torch.float32, device=device, requires_grad=False
    )
    # normalize over depth for normalized (uncentered) xcorrs
    raster_b /= torch.sqrt((raster_b ** 2).sum(dim=1, keepdim=True))
    weights = raster_b[:, None, None, :]  # T11D - OIHW

    D = np.empty((Ta, Tb), dtype=np.float32)
    C = np.empty((Ta, Tb), dtype=np.float32)
    for i in range(0, Ta, batch_size):
        batch = image[i : i + batch_size]
        corr = F.conv2d(  # BT1P
            batch,  # B11D
            weights,
            padding=[0, possible_displacement.size // 2],
        )
        max_corr, best_disp_inds = torch.max(corr[:, :, 0, :], dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[i : i + batch_size] = best_disp
        C[i : i + batch_size] = max_corr.cpu()

    # free GPU memory (except torch drivers... happens when process ends)
    del (
        raster_a,
        raster_b,
        corr,
        batch,
        max_corr,
        best_disp_inds,
        image,
        weights,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return D, C


def psolveonline(D01, C01, D11, C11, p0, mincorr):
    # subsample where corr > mincorr
    i0, j0 = np.nonzero(C01 >= mincorr)
    n0 = i0.shape[0]
    t1 = D01.shape[1]
    i1, j1 = np.nonzero(C11 >= mincorr)
    n1 = i1.shape[0]
    assert t1 == D11.shape[0]

    # construct Kroneckers
    ones0 = np.ones(n0)
    ones1 = np.ones(n1)
    U = sparse.coo_matrix((ones1, (range(n1), i1)), shape=(n1, t1))
    V = sparse.coo_matrix((ones1, (range(n1), j1)), shape=(n1, t1))
    W = sparse.coo_matrix((ones0, (range(n0), j0)), shape=(n0, t1))

    # build lsqr problem
    A = sparse.vstack([U - V, W]).tocsc()
    b = np.concatenate([D11[i1, j1], -(D01 - p0[:, None])[i0, j0]])
    p1, *_ = sparse.linalg.lsmr(A, b)
    return p1


def online_register_rigid(
    raster,
    batch_length=10000,
    time_downsample_factor=1,
    mincorr=0.7,
    disp=None,
    batch_size=32,
    device=None,
):
    """Online rigid registration

    Lower memory and faster for large recordings.

    Returns:
    p : the vector of estimated displacements
    """
    T = raster.shape[1]

    # -- initialize
    raster0 = raster[:, 0:batch_length:time_downsample_factor]
    D00, C00 = calc_corr_decent(
        raster0,
        disp=disp,
        # pbar=False,
        batch_size=batch_size,
        device=device,
        pbar=True,
    )
    p0 = psolvecorr(D00, C00, mincorr=mincorr)

    # -- loop
    ps = [p0]
    for bs in trange(batch_length, T, batch_length, desc="batches"):
        be = min(T, bs + batch_length)
        raster1 = raster[:, bs:be:time_downsample_factor]
        D01, C01 = calc_corr_decent_pair(
            raster0,
            raster1,
            disp=disp,
            batch_size=batch_size,
            device=device,
        )
        D11, C11 = calc_corr_decent(
            raster1,
            disp=disp,
            pbar=False,
            batch_size=batch_size,
            device=device,
        )
        p1 = psolveonline(D01, C01, D11, C11, p0, mincorr)
        ps.append(p1)

        # update loop variables
        raster0 = raster1
        D00 = D11
        C00 = C11
        p0 = p1

    p = np.concatenate(ps)
    return p
