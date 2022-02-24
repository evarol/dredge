import gc
import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F
from tqdm.auto import trange
import time

import lfpreg


class timer:
    def __init__(self, name="timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


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


def psolvesparse(D):
    """Solve for rigid displacement given pairwise disps + corrs"""
    # subsample where corr > mincorr
    T = D.shape[0]
    I, J = D.nonzero()
    n_sampled = I.shape[0]
    # np.matrix -> row vector
    D_nz = np.squeeze(np.array(D[I, J]))

    # construct Kroneckers
    ones = np.ones(n_sampled)
    M = sparse.csc_matrix((ones, (range(n_sampled), I)), shape=(n_sampled, T))
    N = sparse.csc_matrix((ones, (range(n_sampled), J)), shape=(n_sampled, T))

    # solve sparse least squares problem
    print("lsqr problem shape", M.shape, D_nz.shape)
    # p, *_ = sparse.linalg.lsmr(M - N, D_nz, show=True)
    p, *_ = sparse.linalg.lsmr(M - N, D_nz, show=True)
    return p


def batch_register_rigid(
    raw_recording,
    geom,
    batch_length=25_000,
    time_downsample_factor=5,
    mincorr=0.7,
    disp=None,
    csd=False,
    channels=None,
):
    C, T = raw_recording.shape
    nbatches = T // batch_length + (T % batch_length > 0)
    assert nbatches > 1
    T_ds = T // time_downsample_factor + (T % time_downsample_factor > 0)
    print("C", C, "T", T, "T_ds", T_ds)

    # store sparse displacements and correlations
    corrs = []
    disps = []
    iis = []
    jjs = []

    # loop over batches to fill in the displacments and correlations
    for b in trange(1, nbatches, desc="batches"):
        t_prev = (b - 1) * batch_length
        t_cur = b * batch_length
        t_next = min(T, (b + 1) * batch_length)
        t_prev_ds = t_prev // time_downsample_factor
        t_cur_ds = t_cur // time_downsample_factor

        # grab adjacent raw chunks and process
        raster_a = lfpreg.lfpraster(
            raw_recording[:, t_prev:t_cur:time_downsample_factor],
            geom,
            channels=channels,
            csd=csd,
        )
        raster_b = lfpreg.lfpraster(
            raw_recording[:, t_cur:t_next:time_downsample_factor],
            geom,
            channels=channels,
            csd=csd,
        )

        # fill in raw_a's displacements vs itself
        Daa, Caa = calc_corr_decent_pair(raster_a, raster_a)
        ii, jj = np.nonzero(Caa >= mincorr)
        iis.append(t_prev_ds + ii)
        jjs.append(t_prev_ds + jj)
        corrs.append(Caa[ii, jj])
        disps.append(Daa[ii, jj])

        # fill in the pair's displacements
        Dab, Cab = calc_corr_decent_pair(raster_a, raster_b)
        ii, jj = np.nonzero(Cab >= mincorr)
        iis.append(t_prev_ds + ii)
        jjs.append(t_cur_ds + jj)
        corrs.append(Cab[ii, jj])
        disps.append(Dab[ii, jj])
        iis.append(t_cur_ds + jj)
        jjs.append(t_prev_ds + ii)
        corrs.append(Cab[ii, jj])
        disps.append(-Dab[ii, jj])

        # if last batch, also fill in daigonal block for raw_b
        if b == nbatches - 1:
            Dbb, Cbb = calc_corr_decent_pair(raster_b, raster_b)
            ii, jj = np.nonzero(Cbb >= mincorr)
            iis.append(t_cur_ds + ii)
            jjs.append(t_cur_ds + jj)
            corrs.append(Cbb[ii, jj])
            disps.append(Dbb[ii, jj])

    # build sparse matrices
    with timer("Build matrices..."):
        iis = np.hstack(iis)
        jjs = np.hstack(jjs)
        corrs = sparse.coo_matrix(
            (np.hstack(corrs), (iis, jjs)), shape=(T_ds, T_ds)
        ).tocsr()
        disps = sparse.coo_matrix(
            (np.hstack(disps), (iis, jjs)), shape=(T_ds, T_ds)
        ).tocsr()

    # centralize
    print("Kronecker start")
    with timer("Kronecker..."):
        p = psolvesparse(disps)

    return p, disps, corrs


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
    raw_recording,
    geom,
    batch_length=50_000,
    time_downsample_factor=5,
    mincorr=0.7,
    disp=None,
    csd=False,
    channels=slice(None),
):
    T = raw_recording.shape[1]

    # -- initialize
    raster0 = lfpreg.lfpraster(
        raw_recording[channels, 0:batch_length:time_downsample_factor],
        geom[channels],
        channels=channels,
        csd=csd,
    )
    D00, C00 = lfpreg.calc_corr_decent(
        raster0,
        disp=disp,
        pbar=False,
    )
    p0 = lfpreg.psolvecorr(D00, C00, mincorr=mincorr)

    # -- loop
    ps = [p0]
    for bs in trange(batch_length, T, batch_length, desc="batches"):
        be = min(T, bs + batch_length)
        raster1 = lfpreg.lfpraster(
            raw_recording[channels, bs:be:time_downsample_factor],
            geom[channels],
            channels=channels,
            csd=csd,
        )
        D01, C01 = calc_corr_decent_pair(raster0, raster1, disp=disp)
        D11, C11 = lfpreg.calc_corr_decent(
            raster1,
            disp=disp,
            pbar=False,
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
