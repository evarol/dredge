import numpy as np
from .motion_util import (
    get_windows,
    spike_raster,
    get_motion_estimate,
)
from .dredgelib import (
    default_raster_kw,
    weight_correlation_matrix,
    xcorr_windows,
    thomas_solve,
    DEFAULT_LAMBDA_T,
    DEFAULT_EPS,
)


default_weights_kw_ap = dict(
    mincorr=0.1,
    max_dt_s=1000,
    do_window_weights=True,
    weights_threshold_low=0.2,
    weights_threshold_high=0.2,
)


def register(
    amps,
    depths_um,
    times_s,
    rigid=False,
    bin_um=1.0,
    bin_s=1.0,
    win_shape="gaussian",
    win_step_um=400,
    win_scale_um=450,
    win_margin_um=None,
    max_disp_um=None,
    thomas_kw=None,
    xcorr_kw=None,
    raster_kw=default_raster_kw,
    weights_kw=default_weights_kw_ap,
    device=None,
    pbar=True,
    save_full=False,
    precomputed_D_C_maxdisp=None,
):
    """Estimate motion from spikes
    
    Arguments
    ---------
    amps, depths, times : arrays of shape (n_spikes,)
        The amplitudes, depths (microns) and times (seconds) of input
        spike events.
    rigid : bool
        If True, ignore the window args and do rigid registration.
    bin_um, bin_s : numbers
        The size of the bins along depth in microns and along time in seconds.
        The returned object's .displacement array will respect these bins.
    win_shape, win_step_um, win_scale_um, win_margin_um
        Control the shape ("gaussian", "rect"), step/distance between windows,
        their scale/size, and their margin from the border (-1000 means no window
        within 1000um of the edge of the probe)
    max_disp_um : number
        Maximum possible displacement in microns
    thomas_kw, xcorr_kw, raster_kw, weights_kw
        These dictionaries allow setting parameters for fine control over the registration
    
    Returns
    -------
    motion_est : a motion_util.MotionEstimate object
        This has a .displacement attribute which is the displacement estimate in a
        (num_nonrigid_blocks, num_time_bins) array. It also has properties describing
        the time and spatial bins, and methods for getting the displacement at a particular
        time and depth.
    extra : dict
        This has extra info about what happened during registration
    """
    thomas_kw = thomas_kw if thomas_kw is not None else {}
    raster_kw = default_raster_kw | raster_kw
    weights_kw = default_weights_kw_ap | weights_kw
    raster_kw["bin_s"] = bin_s
    raster_kw["bin_um"] = bin_um

    # this will store return values other than the MotionEstimate
    extra = {}

    raster, spatial_bin_edges_um, time_bin_edges_s = spike_raster(
        amps,
        depths_um,
        times_s,
        **raster_kw,
    )
    windows, window_centers = get_windows(
        # pseudo geom to fool spikeinterface
        np.c_[np.zeros_like(spatial_bin_edges_um), spatial_bin_edges_um],
        win_step_um,
        win_scale_um,
        spatial_bin_edges=spatial_bin_edges_um,
        margin_um=-win_scale_um / 2
        if win_margin_um is None
        else win_margin_um,
        win_shape=win_shape,
        zero_threshold=1e-5,
        rigid=rigid,
    )

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
            xcorr_kw=xcorr_kw,
            device=device,
        )
    else:
        Ds, Cs, max_disp_um = precomputed_D_C_maxdisp

    # turn Cs into weights
    Us, wextra = weight_correlation_matrix(
        Ds,
        Cs,
        amps,
        depths_um,
        times_s,
        windows,
        lambda_t=thomas_kw.get("lambda_t", DEFAULT_LAMBDA_T),
        eps=thomas_kw.get("eps", DEFAULT_EPS),
        raster_kw=raster_kw,
        pbar=pbar,
        **weights_kw,
    )
    extra.update({k: wextra[k] for k in wextra if k not in ("S", "U")})
    if save_full:
        extra.update({k: wextra[k] for k in wextra if k in ("S", "U")})

    # solve for P
    # now we can do our tridiag solve
    displacement, textra = thomas_solve(Ds, Us, **thomas_kw)
    me = get_motion_estimate(
        displacement,
        spatial_bin_centers_um=window_centers,
        time_bin_edges_s=time_bin_edges_s,
    )
    if save_full:
        extra.update(textra)

    extra["windows"] = windows
    extra["window_centers"] = window_centers
    extra["max_disp_um"] = max_disp_um
    if save_full:
        extra["D"] = Ds
        extra["C"] = Cs

    return me, extra
