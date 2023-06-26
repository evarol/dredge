import numpy as np
from tqdm.auto import trange
from .motion_util import get_windows, get_motion_estimate
from .dredgelib import xcorr_windows, threshold_correlation_matrix, thomas_solve



def register_online_lfp(
    lfp_recording,
    rigid=True,
    chunk_len_s=10.0,
    max_disp_um=None,
    # nonrigid window construction arguments
    win_shape="gaussian",
    win_step_um=800,
    win_scale_um=850,
    win_margin_um=None,
    # weighting arguments
    mincorr=0.8,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=20,
    soft=False,
    # low-level arguments
    thomas_kw=None,
    xcorr_kw=None,
    # misc
    save_full=False,
    device=None,
    pbar=True,
):
    """Online registration of a preprocessed LFP recording

    Arguments
    ---------
    lfp_recording : spikeinterface recording object
    rigid : boolean, optional
        If True, window-related arguments are ignored and we do rigid registration
    chunk_len_s : float
        Length of chunks (in seconds) that the recording is broken into for online
        registration. The computational speed of the method is a function of the
        number of samples this corresponds to, and things can get slow if it is
        set high enough that the number of samples per chunk is bigger than ~10,000.
        But, it can't be set too low or the algorithm doesn't have enough data
        to work with. The default is set assuming sampling rate of 250Hz, leading
        to 2500 samples per chunk.
    max_disp_um : number, optional
        This is the ceiling on the possible displacement estimates. It should be
        set to a number which is larger than the allowed displacement in a single
        chunk. Setting it as small as possible (while following that rule) can speed
        things up.
    win_shape, win_step_um, win_scale_um, win_margin_um
        Nonrigid window-related arguments
        The depth domain will be broken up into windows with shape controlled by win_shape,
        spaced by win_step_um at a margin of win_margin_um from the boundary, and with
        width controlled by win_scale_um.
    mincorr : float in [0,1]
        Minimum maximal correlation between time bins such that they will be included
        in the optimization
    mincorr_percentile, mincorr_percentile_nneighbs
        If mincorr_percentile is set to a number in [0, 100], then mincorr will be replaced
        by this percentile of the correlations of neighbors within mincorr_percentile_nneighbs
        time bins of each other
    device : string or torch.device
        Controls torch device

    Returns
    -------
    me : motion_util.MotionEstimate
        A motion estimate object. me.displacement is the displacement trace, but this object
        includes methods for getting the displacement at different times and depths; see
        the documentation in the motion_util.py file.
    extra : dict
        Dict containing extra info for debugging
    """
    geom = lfp_recording.get_channel_locations()
    fs = lfp_recording.get_sampling_frequency()
    T_total = lfp_recording.get_num_samples()
    T_chunk = min(int(np.floor(fs * chunk_len_s)), T_total)

    # kwarg defaults and handling
    # need lfp-specific defaults
    weights_kw = dict(
        mincorr=mincorr,
        mincorr_percentile_nneighbs=mincorr_percentile_nneighbs,
        soft=soft,
        do_window_weights=False,
        max_dt_s=None,
    )
    xcorr_kw = xcorr_kw if xcorr_kw is not None else {}
    thomas_kw = thomas_kw if thomas_kw is not None else {}
    full_xcorr_kw = dict(
        rigid=rigid,
        bin_um=np.median(np.diff(geom[:, 1])),
        max_disp_um=max_disp_um,
        pbar=False,
        xcorr_kw=xcorr_kw,
        device=device,
    )
    mincorr_percentile = None
    assert "max_dt_s" not in weights_kw or weights_kw["max_dt_s"] is None
    threshold_kw = dict(
        mincorr_percentile_nneighbs=weights_kw["mincorr_percentile_nneighbs"],
        # max_dt_s=weights_kw["max_dt_s"],  # max_dt not implemented for lfp at this point
        # bin_s=1 / fs,  # only relevant for max_dt_s
        in_place=True,
        soft=soft,
    )
    if "mincorr_percentile" in weights_kw:
        mincorr_percentile = weights_kw["mincorr_percentile"]

    # get windows
    windows, window_centers = get_windows(
        geom,
        win_step_um,
        win_scale_um,
        spatial_bin_centers=geom[:, 1],
        margin_um=win_margin_um,
        win_shape=win_shape,
        zero_threshold=1e-5,
        rigid=rigid,
    )
    B = len(windows)
    extra = dict(window_centers=window_centers, windows=windows)

    # -- allocate output and initialize first chunk
    P_online = np.empty((B, T_total), dtype=np.float32)
    # below, t0 is start of prev chunk, t1 start of cur chunk, t2 end of cur
    t0, t1 = 0, T_chunk
    traces0 = lfp_recording.get_traces(start_frame=t0, end_frame=t1)
    Ds0, Cs0, max_disp_um = xcorr_windows(
        traces0.T, windows, geom[:, 1], win_scale_um, **full_xcorr_kw
    )
    full_xcorr_kw["max_disp_um"] = max_disp_um
    Ss0, mincorr0 = threshold_correlation_matrix(
        Cs0, mincorr_percentile=mincorr_percentile, mincorr=mincorr, **threshold_kw
    )
    if save_full:
        extra["D"] = [Ds0]
        extra["C"] = [Cs0]
        extra["S"] = [Ss0]
        extra["D01"] = []
        extra["C01"] = []
        extra["S01"] = []
    extra["mincorrs"] = [mincorr0]
    extra["max_disp_um"] = max_disp_um
    P_online[:, t0:t1], _ = thomas_solve(Ds0, Ss0, **thomas_kw)

    # -- loop through chunks
    chunk_starts = range(T_chunk, T_total, T_chunk)
    if pbar:
        chunk_starts = trange(T_chunk, T_total, T_chunk, desc=f"Online chunks [{chunk_len_s}s each]")
    for t1 in chunk_starts:
        t2 = min(T_total, t1 + T_chunk)
        traces1 = lfp_recording.get_traces(start_frame=t1, end_frame=t2)

        # cross-correlations between prev/cur chunks
        Ds10, Cs10, _ = xcorr_windows(
            traces1.T,
            windows,
            geom[:, 1],
            win_scale_um,
            raster_b=traces0.T,
            **full_xcorr_kw,
        )

        # cross-correlation in current chunk
        Ds1, Cs1, _ = xcorr_windows(
            traces1.T, windows, geom[:, 1], win_scale_um, **full_xcorr_kw
        )
        Ss1, mincorr1 = threshold_correlation_matrix(
            Cs1, mincorr_percentile=mincorr_percentile, mincorr=mincorr, **threshold_kw
        )
        Ss10, _ = threshold_correlation_matrix(Cs10, mincorr=mincorr1, **threshold_kw)
        extra["mincorrs"].append(mincorr1)

        if save_full:
            extra["D"].append(Ds1)
            extra["C"].append(Cs1)
            extra["S"].append(Ss1)
            extra["D01"].append(Ds10)
            extra["C01"].append(Cs10)
            extra["S01"].append(Ss10)

        # solve online problem
        P_online[:, t1:t2], _ = thomas_solve(
            Ds1,
            Ss1,
            P_prev=P_online[:, t0:t1],
            Ds_curprev=Ds10,
            Us_curprev=Ss10,
            Ds_prevcur=-Ds10.transpose(0, 2, 1),
            Us_prevcur=Ss10.transpose(0, 2, 1),
            **thomas_kw,
        )

        # update loop vars
        t0, t1 = t1, t2
        traces0 = traces1

    # -- convert to motion estimate and return
    me = get_motion_estimate(
        P_online,
        time_bin_centers_s=lfp_recording.get_times(0),
        spatial_bin_centers_um=window_centers,
    )
    return me, extra
