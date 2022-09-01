import numpy as np
import pywt
import torch
from scipy.stats import norm
from scipy.interpolate import interp1d, RectBivariateSpline
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm.auto import tqdm

from .ibme_fast_raster import raster as _raster
from .ibme_corr import register_raster_rigid


# -- main functions: rigid and nonrigid registration


def register_rigid(
    amps,
    depths,
    times,
    disp=None,
    robust_sigma=0.0,
    denoise_sigma=0.1,
    corr_threshold=0.0,
    normalized=True,
    destripe=False,
    max_dt=None,
    batch_size=1,
    return_extra=False,
):
    """1D rigid registration

    Arguments
    ---------
    amps, depths, times : np.arrays of shape (N_spikes,)
        Spike amplitudes, depths (locations along probe),
        and times (in seconds)
    disp : int, optional
        Maximum displacement to search over. By default,
        will use half of the length of the depth domain.
    robust_sigma : float
        If >0, a robust least squares procedure similar to
        least trimmed squares will be used. A smaller choice
        means that more points are considered outliers.
    denoise_sigma : float
        Strength of Poisson denoising applied to the raster
    corr_threshold : float
        Exclude pairs of timebins from the optimization if
        their maximum correlation does not exceed this value
    destripe : bool
        Apply a wavelet-based destriping to the raster

    Returns
    -------
    depths_reg : np.array of shape (N_spikes,)
        The registered depths
    p : np.array
        The estimated displacement for each time bin.
    """
    # -- compute time/depth binned amplitude averages
    raster, dd, tt = fast_raster(
        amps, depths, times, sigma=denoise_sigma, destripe=destripe
    )

    # -- main routine from other file `ibme_corr`
    p, D, C = register_raster_rigid(
        raster,
        mincorr=corr_threshold,
        robust_sigma=robust_sigma,
        disp=disp,
        batch_size=batch_size,
        normalized=normalized,
        max_dt=max_dt,
    )
    extra = dict(D=D, C=C)

    # return new interpolated depths for the caller
    depths_reg = warp_rigid(depths, times, tt, p)

    if return_extra:
        return depths_reg, p, extra

    return depths_reg, p


@torch.no_grad()
def register_nonrigid(
    amps,
    depths,
    times,
    robust_sigma=0.0,
    denoise_sigma=0.1,
    corr_threshold=0.0,
    normalized=True,
    rigid_disp=400,
    disp=800,
    rigid_init=False,
    n_windows=10,
    widthmul=0.5,
    max_dt=None,
    destripe=False,
    device=None,
    batch_size=1,
):
    """1D nonrigid registration

    This is optionally an iterative procedure if rigid_init=True
    or n_windows is iterable. If rigid_init is True, first a rigid
    registration is applied to depths. Then for each number of
    windows, a nonrigid registration with so many windows is applied.
    These are done cumulatively, making it possible to take a
    coarse-to-fine approach.

    Arguments
    ---------
    amps, depths, times : np.arrays of shape (N_spikes,)
        Spike amplitudes, depths (locations along probe),
        and times (in seconds)
    rigid_disp : int, optional
        Maximum displacement to search over during rigid
        initialization when rigid_init=True.
    disp : int
        Base displacement during nonrigid registration.
        The actual value used is disp / n_windows
    rigid_init : bool
        Should we initialize with a rigid registration?
    robust_sigma : float
        If >0, a robust least squares procedure similar to
        least trimmed squares will be used. A smaller choice
        means that more points are considered outliers.
    denoise_sigma : float
        Strength of Poisson denoising applied to the raster
    corr_threshold : float
        Exclude pairs of timebins from the optimization if
        their maximum correlation does not exceed this value
    destripe : bool
        Apply a wavelet-based destriping to the raster

    Returns
    -------
    depths : np.array of shape (N_spikes,)
        The registered depths
    total_shift : DxT array
        The displacement map
    """
    if isinstance(n_windows, int):
        n_windows = [n_windows]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offset = depths.min()

    # set origin to min z
    depths = depths - offset

    # initialize displacement map
    D = 1 + int(np.floor(depths.max()))
    T = int(np.floor(times.max())) + 1
    total_shift = np.zeros((D, T))

    # first pass of rigid registration
    if rigid_init:
        depths, p = register_rigid(
            amps,
            depths,
            times,
            robust_sigma=robust_sigma,
            corr_threshold=corr_threshold,
            normalized=normalized,
            batch_size=batch_size,
            disp=rigid_disp,
            denoise_sigma=denoise_sigma,
            destripe=destripe,
            max_dt=max_dt,
        )
        total_shift[:, :] = p[None, :]

    for nwin in tqdm(n_windows):
        raster, dd, tt = fast_raster(
            amps, depths, times, sigma=denoise_sigma, destripe=destripe
        )
        D, T = raster.shape

        # gaussian windows
        windows = np.empty((nwin, D))
        slices = []
        space = D // (nwin + 1)
        locs = np.linspace(space, D - space, nwin)
        scale = widthmul * D / nwin
        for k, loc in enumerate(locs):
            windows[k, :] = norm.pdf(np.arange(D), loc=loc, scale=scale)
            domain_large_enough = np.flatnonzero(windows[k, :] > 1e-5)
            slices.append(slice(domain_large_enough[0], domain_large_enough[-1]))
        windows /= windows.sum(axis=0, keepdims=True)

        # torch versions on device
        windows_ = torch.as_tensor(windows, dtype=torch.float, device=device)
        raster_ = torch.as_tensor(raster, dtype=torch.float, device=device)

        # estimate each window's displacement
        ps = np.empty((nwin, T))
        for k, window in enumerate(tqdm(windows_, desc="windows")):
            p, D, C = register_raster_rigid(
                (window[:, None] * raster_)[slices[k]],
                mincorr=corr_threshold,
                normalized=normalized,
                robust_sigma=robust_sigma,
                max_dt=max_dt,
                disp=max(25, int(np.ceil(disp / nwin))),
                batch_size=batch_size,
                pbar=False,
            )
            ps[k] = p

        # warp depths
        dispmap = windows.T @ ps
        depths = warp_nonrigid(
            depths, times, dispmap, depth_domain=dd, time_domain=tt
        )
        depths -= depths.min()

        # update displacement map
        total_shift[:, :] = compose_shifts_in_orig_domain(total_shift, dispmap)

    # back to original coordinates
    depths -= depths.min()
    depths += offset

    return depths, total_shift


# -- nonrigid reg helpers


def compose_shifts_in_orig_domain(orig_shift, new_shift):
    """
    Context: we are in the middle of an iterative algorithm which
    computes displacements. Previously, we had computed some DxT
    displacement matrix `orig_shift`, giving a displacement at all
    depths and times. Then, we used that to compute a new registered
    raster, and we got a new displacement estimate from that new
    raster, `new_shift`. The problem is, that new raster may have a
    different depth domain than the original -- so, how do we combine
    these?

    Like so: if (x, t) is a point in the original domain, then think of
    `orig_shift` as a function f(x, t) giving the displacement at depth
    x and time t. Similarly, write `new_shift` as g(x', t), where x' is
    a point in the domain after it has been displaced by f, i.e.,
    x' = f(x, t) + x. Then we want to know, how much is a point (x,t)
    displaced by applying both f and g?

    The answer:
        (x, t) -> (f(x, t) + g(f(x, t) + x, t), t).
    So, this function returns h(x, t) = f(x, t) + g(f(x, t) + x, t)
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


def warp_nonrigid(depths, times, dispmap, depth_domain=None, time_domain=None):
    if depth_domain is None:
        depth_domain = np.arange(int(np.floor(depths.max())) + 1)
    if time_domain is None:
        time_domain = np.arange(int(np.floor(times.max())) + 1)

    lerp = RectBivariateSpline(
        depth_domain,
        time_domain,
        dispmap,
        kx=1,
        ky=1,
    )

    return depths - lerp(depths, times, grid=False)


def warp_rigid(depths, times, time_domain, p):
    warps = interp1d(time_domain + 0.5, p, fill_value="extrapolate")(times)
    depths_reg = depths - warps
    depths_reg -= depths_reg.min()
    return depths_reg


# -- code for making rasters


def fast_raster(
    amps,
    depths,
    times,
    dd=None,
    tt=None,
    sigma=None,
    destripe=False,
    return_N=False,
):
    # could do [0, D] instead of [min depth, max depth]
    # dd = np.linspace(depths.min(), depths.max(), num=D)
    if dd is None:
        D = 1 + int(np.floor(depths.max()))
        dd = np.arange(0, D)
    else:
        D = dd.size

    if tt is None:
        T = 1 + int(np.floor(times.max()))
        tt = np.arange(0, T)

    M = np.zeros((D, T))
    N = np.zeros((D, T))
    R = np.zeros((D, T))
    times = np.ascontiguousarray(times, dtype=np.float64)
    depths = np.ascontiguousarray(depths, dtype=np.float64)
    amps = np.ascontiguousarray(amps, dtype=np.float64)
    _raster(times, depths, amps, M, N)

    Nnz = np.nonzero(N)
    R[Nnz] = M[Nnz] / N[Nnz]

    if destripe:
        R = wtdestripe(R)

    if sigma is not None:
        nz = np.flatnonzero(R.max(axis=1))
        if nz.size > 0:
            R[nz, :] = cheap_anscombe_denoising(
                R[nz, :], estimate_sig=False, sigma=sigma
            )

    if return_N:
        return R, N, dd, tt

    return R, dd, tt


def wtdestripe(raster):
    D, W = raster.shape
    LL0 = raster
    wlet = "db5"
    coeffs = pywt.wavedec2(LL0, wlet)
    L = len(coeffs)
    for i in range(1, L):
        HL = coeffs[i][1]
        Fb = np.fft.fft2(HL)
        Fb = np.fft.fftshift(Fb)
        mid = Fb.shape[0] // 2
        Fb[mid, :] = 0
        Fb[mid - 1, :] /= 3
        Fb[mid + 1, :] /= 3
        Fb = np.fft.ifftshift(Fb)
        coeffs[i] = (coeffs[i][0], np.real(np.fft.ifft2(Fb)), coeffs[i][2])
    LL = pywt.waverec2(coeffs, wlet)
    LL = np.ascontiguousarray(LL[:D, :W], dtype=raster.dtype)
    return LL


def cheap_anscombe_denoising(
    z,
    sigma=1,
    h=0.1,
    estimate_sig=True,
    fast_mode=True,
    multichannel=False,
    patch_size=5,
    patch_distance=5,
):
    minmax = (z - z.min()) / (z.max() - z.min())  # scales data to 0-1

    # Gaussianizing Poissonian data
    z_anscombe = 2.0 * np.sqrt(minmax + (3.0 / 8.0))

    if estimate_sig:
        sigma = np.mean(estimate_sigma(z_anscombe, multichannel=multichannel))
        print(f"estimated sigma: {sigma}")

    # Gaussian denoising
    z_anscombe_denoised = denoise_nl_means(
        z_anscombe,
        h=h * sigma,
        sigma=sigma,
        fast_mode=fast_mode,
        patch_size=patch_size,
        patch_distance=patch_distance,
    )

    z_inverse_anscombe = (
        (z_anscombe_denoised / 2.0) ** 2
        + 0.25 * np.sqrt(1.5) * z_anscombe_denoised ** -1
        - (11.0 / 8.0) * z_anscombe_denoised ** -2
        + (5.0 / 8.0) * np.sqrt(1.5) * z_anscombe_denoised ** -3
        - (1.0 / 8.0)
    )

    z_inverse_anscombe_scaled = (z.max() - z.min()) * (
        z_inverse_anscombe - z_inverse_anscombe.min()
    )

    return z_inverse_anscombe_scaled
