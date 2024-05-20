import gc

import numpy as np

from .dredgelib import (DEFAULT_EPS, DEFAULT_LAMBDA_T, thomas_solve,
                        weight_correlation_matrix, xcorr_windows)
from .motion_util import get_motion_estimate, get_windows, spike_raster


def register(
    amps,
    depths_um,
    times_s,
    rigid=False,
    bin_um=1.0,
    bin_s=1.0,
    max_disp_um=None,
    max_dt_s=1000,
    mincorr=0.1,
    # nonrigid window construction arguments
    win_shape="gaussian",
    win_step_um=400,
    win_scale_um=450,
    win_margin_um=None,
    # weights arguments
    do_window_weights=True,
    weights_threshold_low=0.2,
    weights_threshold_high=0.2,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=None,
    # raster arguments
    amp_scale_fn=None,
    post_transform=np.log1p,
    gaussian_smoothing_sigma_um=1,
    gaussian_smoothing_sigma_s=1,
    avg_in_bin=False,
    count_masked_correlation=False,
    count_bins=401,
    count_bin_min=2,
    # low-level keyword args
    thomas_kw=None,
    xcorr_kw=None,
    # misc
    device=None,
    pbar=True,
    save_full=False,
    precomputed_D_C_maxdisp=None,
):
    """Estimate motion from spikes

    Spikes located at depths specified in `depths` along the probe, occurring at times in
    seconds specified in `times` with amplitudes `amps` are used to create a 2d image of
    the spiking activity. This image is cross-correlated with itself to produce a displacement
    matrix (or several, one for each nonrigid window). This matrix is used to solve for a
    motion estimate.

    Arguments
    ---------
    amps : np.array of shape (n_spikes,)
    depths: np.array of shape (n_spikes,)
    times : np.array of shape (n_spikes,)
        The amplitudes, depths (microns) and times (seconds) of input
        spike events.
    rigid : bool, default=False
        If True, ignore the nonrigid window args (win_shape, win_step_um, win_scale_um,
        win_margin_um) and do rigid registration (equivalent to one flat window, which
        is how it's implemented).
    bin_um: float
    bin_s : float
        The size of the bins along depth in microns and along time in seconds.
        The returned object's .displacement array will respect these bins.
        Increasing these can lead to more stable estimates and faster runtimes
        at the cost of spatial and/or temporal resolution.
    max_disp_um : float
        Maximum possible displacement in microns. If you can guess a number which is larger
        than the largest displacement possible in your recording across a span of `max_dt_s`
        seconds, setting this value to that number can stabilize the result and speed up
        the algorithm (since it can do less cross-correlating).
        By default, this is set to win-scale_um / 4, or 112.5 microns. Which can be a bit
        large!
    max_dt_s : float
        "Time horizon" parameter, in seconds. Time bins separated by more seconds than this
        will not be cross-correlated. So, if your data has nonstationarities or changes which
        could lead to bad cross-correlations at some timescale, it can help to input that
        value here. If this is too small, it can make the motion estimation unstable.
    mincorr : float, between 0 and 1
        Correlation threshold. Pairs of frames whose maximal cross correlation value is smaller
        than this threshold will be ignored when solving for the global displacement estimate.
    win_shape : str, default="gaussian"
        Nonrigid window shape
    win_step_um : float
        Spacing between nonrigid window centers in microns
    win_scale_um : float
        Controls the width of nonrigid windows centers
    win_margin_um : float
        Distance of nonrigid windows centers from the probe boundary (-1000 means there will
        be no window center within 1000um of the edge of the probe)
    thomas_kw, xcorr_kw, raster_kw, weights_kw
        These dictionaries allow setting parameters for fine control over the registration
    device : str or torch.device
        What torch device to run on? E.g., "cpu" or "cuda" or "cuda:1".

    Returns
    -------
    motion_est : a motion_util.MotionEstimate object
        This has a .displacement attribute which is the displacement estimate in a
        (num_nonrigid_blocks, num_time_bins) array. It also has properties describing
        the time and spatial bins, and methods for getting the displacement at a particular
        time and depth. See the documentation of these classes in motion_util.py.
    extra : dict
        This has extra info about what happened during registration, including the nonrigid
        windows if one wants to visualize them. Set `save_full` to also save displacement
        and correlation matrices.
    """
    thomas_kw = thomas_kw if thomas_kw is not None else {}
    xcorr_kw = xcorr_kw if xcorr_kw is not None else {}
    if max_dt_s:
        xcorr_kw["max_dt_bins"] = np.ceil(max_dt_s / bin_s)
    raster_kw = dict(
        amp_scale_fn=amp_scale_fn,
        post_transform=post_transform,
        gaussian_smoothing_sigma_um=gaussian_smoothing_sigma_um,
        gaussian_smoothing_sigma_s=gaussian_smoothing_sigma_s,
        bin_s=bin_s,
        bin_um=bin_um,
        avg_in_bin=avg_in_bin,
        return_counts=count_masked_correlation,
        count_bins=count_bins,
        count_bin_min=count_bin_min,
    )
    weights_kw = dict(
        mincorr=mincorr,
        max_dt_s=max_dt_s,
        do_window_weights=do_window_weights,
        weights_threshold_low=weights_threshold_low,
        weights_threshold_high=weights_threshold_high,
    )

    # this will store return values other than the MotionEstimate
    extra = {}

    raster_res = spike_raster(
        amps,
        depths_um,
        times_s,
        **raster_kw,
    )
    if count_masked_correlation:
        raster, spatial_bin_edges_um, time_bin_edges_s, counts = raster_res
    else:
        raster, spatial_bin_edges_um, time_bin_edges_s = raster_res
    windows, window_centers = get_windows(
        # pseudo geom to fool spikeinterface
        np.c_[np.zeros_like(spatial_bin_edges_um), spatial_bin_edges_um],
        win_step_um,
        win_scale_um,
        spatial_bin_edges=spatial_bin_edges_um,
        margin_um=-win_scale_um / 2 if win_margin_um is None else win_margin_um,
        win_shape=win_shape,
        zero_threshold=1e-5,
        rigid=rigid,
    )
    if save_full and count_masked_correlation:
        extra["counts"] = counts

    # cross-correlate to get D and C
    if precomputed_D_C_maxdisp is None:
        Ds, Cs, max_disp_um = xcorr_windows(
            raster,
            windows,
            spatial_bin_edges_um,
            win_scale_um,
            rigid=rigid,
            bin_um=bin_um,
            max_disp_um=max_disp_um,
            pbar=pbar,
            device=device,
            masks=(counts > 0) if count_masked_correlation else None,
            **xcorr_kw,
        )
    else:
        Ds, Cs, max_disp_um = precomputed_D_C_maxdisp

    # turn Cs into weights
    Us, wextra = weight_correlation_matrix(
        Ds,
        Cs,
        windows,
        raster,
        spatial_bin_edges_um,
        time_bin_edges_s,
        raster_kw,
        lambda_t=thomas_kw.get("lambda_t", DEFAULT_LAMBDA_T),
        eps=thomas_kw.get("eps", DEFAULT_EPS),
        pbar=pbar,
        in_place=not save_full,
        **weights_kw,
    )
    extra.update({k: wextra[k] for k in wextra if k not in ("S", "U")})
    if save_full:
        extra.update({k: wextra[k] for k in wextra if k in ("S", "U")})
    del wextra
    if save_full:
        extra["D"] = Ds
        extra["C"] = Cs
    del Cs
    gc.collect()

    # solve for P
    # now we can do our tridiag solve
    displacement, textra = thomas_solve(Ds, Us, pbar=pbar, **thomas_kw)
    if save_full:
        extra.update(textra)
    del textra
    me = get_motion_estimate(
        displacement,
        spatial_bin_centers_um=window_centers,
        time_bin_edges_s=time_bin_edges_s,
    )

    extra["windows"] = windows
    extra["window_centers"] = window_centers
    extra["max_disp_um"] = max_disp_um

    return me, extra
