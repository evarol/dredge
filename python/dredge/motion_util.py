"""motion_utils

This library has 3 sections:
 - Helper classes (subclasses of MotionEstimate) for managing
   the output of registration. Make one with the `get_motion_estimate`
   function
 - Helper functions for plotting motion estimates
 - Helper functions for making nonrigid windows / binning space+time
   / computing rasters

The main registration APIs in dredge_ap and drege_lfp use these helpers.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d

# -- motion estimate helper classes


class MotionEstimate:
    """MotionEstimate

    This class won't be instantiated, users interact with subclasses below
    which are usually instantiated by the `get_motion_estimate()` function.

    MotionEstimate and its subclasses manage your displacement estimate
    and its temporal (and optionally spatial, if nonrigid) domain(s),
    and they can smartly give you the displacement at a given time (and
    optionally depth, if nonrigid).

    You can use the `disp_at_s(...)` method on subclasses to get the
    displacement at a time (and, optionally, depth if nonrigid).
    You can use the `correct_s(...)` method to compute the drift-corrected
    position at a given time (and depth).
    """

    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        spatial_bin_edges_um=None,
        time_bin_centers_s=None,
        spatial_bin_centers_um=None,
    ):
        self.displacement = displacement
        self.time_bin_edges_s = time_bin_edges_s
        self.spatial_bin_edges_um = spatial_bin_edges_um

        self.time_bin_centers_s = time_bin_centers_s
        if time_bin_edges_s is not None:
            if time_bin_centers_s is None:
                self.time_bin_centers_s = 0.5 * (
                    time_bin_edges_s[1:] + time_bin_edges_s[:-1]
                )

        self.spatial_bin_centers_um = spatial_bin_centers_um
        if spatial_bin_edges_um is not None:
            if spatial_bin_centers_um is None:
                self.spatial_bin_centers_um = 0.5 * (
                    spatial_bin_edges_um[1:] + spatial_bin_edges_um[:-1]
                )

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        """Get the displacement at time t_s and depth depth_um

        ! This must be implemented by subclasses!

        Arguments
        ---------
        t_s, depth_um : floats or np.arrays
            These should be numbers or arrays of the same shape corresponding to times
            (in seconds) and depths (in microns)
        grid : boolean, optional
            If true, treat t_s and depth_um as x/y coordinates of a 2d rectangular grid.
            Then, if t_s and depth_um have `n` and `m` elements, this computes displacements
            on the `n x m` grid.

        Returns
        -------
        An array of displacements in microns with the same shape as depth_um (when grid=False).
        """
        raise NotImplementedError

    def correct_s(self, t_s, depth_um, grid=False):
        """Return the registered depths for events at times `t_s` and depths `depth_um`

        Arguments
        ---------
        t_s, depth_um : floats or np.arrays
            These should be numbers or arrays of the same shape corresponding to times
            (in seconds) and depths (in microns)
        grid : boolean, optional
            If true, treat t_s and depth_um as x/y coordinates of a 2d rectangular grid.
            Then, if t_s and depth_um have `n` and `m` elements, this applies displacements
            on the `n x m` grid.

        Returns
        -------
        An array of motion-corrected depth positions in microns with the same shape as
        depth_um (when grid=False).
        """
        return np.asarray(depth_um) - self.disp_at_s(t_s, depth_um, grid=grid)


class RigidMotionEstimate(MotionEstimate):
    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        time_bin_centers_s=None,
    ):
        displacement = np.atleast_1d(np.asarray(displacement).squeeze())

        assert displacement.ndim == 1
        if time_bin_edges_s is not None:
            assert (1 + displacement.shape[0],) == time_bin_edges_s.shape
        else:
            assert time_bin_centers_s is not None
            assert time_bin_centers_s.shape == displacement.shape

        super().__init__(
            displacement,
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
        )

        self.lerp = interp1d(
            self.time_bin_centers_s,
            self.displacement,
            bounds_error=False,
            fill_value=tuple(self.displacement[[0, -1]]),
        )

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        """Get the displacement at times `t_s` and depths `depth_um`

        Arguments
        ---------
        t_s : float or np.array
            A float or array of floats with times in seconds at which you want to
            get the estimated displacement
        depth_um : optional, float or np.array
            Since the motion estimate is rigid, the displacement does not depend
            on the depth, and this is ignored. It's just here to make a uniform
            interface across MotionEstimate classes, so that users don't need to
            worry about which one they have.
        grid : boolean, optional
            If true, treat t_s and depth_um as x/y coordinates of a 2d rectangular grid.
            Then, if t_s and depth_um have `n` and `m` elements, this computes displacements
            on the `n x m` grid.

        Returns
        -------
        An array of displacements in microns with the same shape as t_s (when grid=False).
        """
        if depth_um is not None and np.asarray(depth_um).shape != np.asarray(t_s).shape:
            assert grid
        disp = self.lerp(np.asarray(t_s))
        if grid:
            disp = disp[None]
            if depth_um is not None:
                disp = np.broadcast_to(disp, (*np.atleast_1d(depth_um).shape, *np.atleast_1d(t_s).shape))
        return disp


class NonrigidMotionEstimate(MotionEstimate):
    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        time_bin_centers_s=None,
        spatial_bin_edges_um=None,
        spatial_bin_centers_um=None,
    ):
        assert displacement.ndim == 2
        if time_bin_edges_s is not None:
            time_bin_edges_s
            assert (1 + displacement.shape[1],) == time_bin_edges_s.shape
        else:
            assert time_bin_centers_s is not None
            assert (displacement.shape[1],) == time_bin_centers_s.shape
        if spatial_bin_edges_um is not None:
            assert (1 + displacement.shape[0],) == spatial_bin_edges_um.shape
        else:
            assert spatial_bin_centers_um is not None
            assert (displacement.shape[0],) == spatial_bin_centers_um.shape

        super().__init__(
            displacement,
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
            spatial_bin_edges_um=spatial_bin_edges_um,
            spatial_bin_centers_um=spatial_bin_centers_um,
        )

        # used below to disable RectBivariateSpline's extrapolation behavior
        # we'd rather fill in with the boundary value than make some line
        # going who knows where
        self.t_low = self.time_bin_centers_s.min()
        self.t_high = self.time_bin_centers_s.max()
        self.d_low = self.spatial_bin_centers_um.min()
        self.d_high = self.spatial_bin_centers_um.max()

        self.lerp = RegularGridInterpolator(
            (self.spatial_bin_centers_um, self.time_bin_centers_s),
            self.displacement,
        )

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        """Get the displacement at times `t_s` and depths `depth_um`

        Arguments
        ---------
        t_s : float or np.array
            A float or array of floats with times in seconds at which you want to
            get the estimated displacement
        depth_um : optional, float or np.array
            Since the motion estimate is nonrigid, the displacement depends on depth.
            This should be an array of the same shape as t_s giving the depths for each
            of the times in t_s.
        grid : boolean, optional
            If true, treat t_s and depth_um as x/y coordinates of a 2d rectangular grid.
            Then, if t_s and depth_um have `n` and `m` elements, this computes displacements
            on the `n x m` grid.

        Returns
        -------
        An array of displacements in microns with the same shape as t_s (when grid=False).
        """
        if np.asarray(depth_um).shape != np.asarray(t_s).shape:
            assert grid
        if grid:
            depth_um, t_s = np.meshgrid(depth_um, t_s, indexing="ij")
        points = np.c_[
            np.clip(depth_um, self.d_low, self.d_high).ravel(),
            np.clip(t_s, self.t_low, self.t_high).ravel(),
        ]
        return self.lerp(points).reshape(np.asarray(t_s).shape)


class IdentityMotionEstimate(RigidMotionEstimate):
    """The motion estimate with no motion."""

    def __init__(self):
        super().__init__(np.array([0.0]), time_bin_centers_s=np.array([0.0]))


class ComposeMotionEstimates(MotionEstimate):
    """Compose motion estimates, applying them in forward order (not reverse!)."""

    def __init__(self, *motion_estimates):
        super().__init__(None)
        self.motion_estimates = motion_estimates
        self.time_bin_edges_s = motion_estimates[-1].time_bin_edges_s
        self.time_bin_centers_s = motion_estimates[-1].time_bin_centers_s

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        assert not grid

        disp = np.zeros_like(t_s)
        if depth_um is None:
            depth_um = np.zeros_like(t_s)

        for me in self.motion_estimates:
            disp += me.disp_at_s(t_s, depth_um + disp)

        return disp


def get_motion_estimate(
    displacement,
    time_bin_edges_s=None,
    time_bin_centers_s=None,
    spatial_bin_edges_um=None,
    spatial_bin_centers_um=None,
    windows=None,
    window_weights=None,
    upsample_by_windows=False,
):
    """Helper function for constructing MotionEstimates

    This would be the suggested way to instantiate RigidMotionEstimates
    and NonrigidMotionEstimates, since it handles both cases equally.

    Returns: an instance of a MotionEstimate subclass.
    """
    displacement = np.asarray(displacement).squeeze()
    assert displacement.ndim <= 2
    assert any(a is not None for a in (time_bin_edges_s, time_bin_centers_s))

    # rigid case
    if displacement.ndim == 1:
        return RigidMotionEstimate(
            displacement,
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
        )
    assert any(
        a is not None for a in (spatial_bin_edges_um, spatial_bin_centers_um)
    )

    # linear interpolation nonrigid
    if not upsample_by_windows:
        return NonrigidMotionEstimate(
            displacement,
            time_bin_edges_s=time_bin_edges_s,
            time_bin_centers_s=time_bin_centers_s,
            spatial_bin_edges_um=spatial_bin_edges_um,
            spatial_bin_centers_um=spatial_bin_centers_um,
        )

    # upsample using the windows to spatial_bin_centers space
    if spatial_bin_centers_um is not None:
        D = spatial_bin_centers_um.shape[0]
    else:
        D = spatial_bin_edges_um.shape[0] - 1
    assert windows.shape == (displacement.shape[0], D)
    if window_weights is None:
        window_weights = np.ones_like(displacement)
    assert window_weights.shape == displacement.shape
    # precision weighted average
    normalizer = windows.T @ window_weights
    displacement_upsampled = (
        windows.T @ (displacement * window_weights)
    ) / normalizer

    return NonrigidMotionEstimate(
        displacement_upsampled,
        time_bin_edges_s=time_bin_edges_s,
        time_bin_centers_s=time_bin_centers_s,
        spatial_bin_edges_um=spatial_bin_edges_um,
        spatial_bin_centers_um=spatial_bin_centers_um,
    )


def get_interpolated_recording(
    motion_est, recording, border_mode="remove_channels"
):
    """Use spikeinterface to interpolate a recording to correct for the motion in motion_est

    This handles internally translation between the sample times of recording
    and motion_est. So, you can use this function with a motion_est computed from 250Hz
    LFP to correct 250Hz LFP or 2500Hz LFP or 30kHz AP equally.

    Arguments
    ---------
    motion_est : a MotionEstimate object
    recording : a SpikeInterface recording
        These two objects should be trimmed to the same time domain (different
        sample times is okay, but they should have the same temporal extent in
        seconds)

    Returns
    -------
    rec_interpolated : spikeinterface InterpolateMotionRecording object
    """
    # we need to make a copy of the recording which has no times stored
    # this is not something spikeinterface supports so we are doing it manually
    from copy import copy
    from spikeinterface.sortingcomponents.motion_interpolation import (
        interpolate_motion,
    )

    rec = copy(recording)
    rec._recording_segments[0] = copy(rec._recording_segments[0])
    rec._recording_segments[0].t_start = None
    rec._recording_segments[0].time_vector = None

    # fake the temporal bins
    # we have been maintaining the metadata about time bins in our motion estimate,
    # but again, spikeinterface is not doing this so we have to throw that info away
    # before calling spikeinterface functions
    dt = np.diff(motion_est.time_bin_centers_s).min()
    temporal_bins = (
        motion_est.time_bin_centers_s
        - motion_est.time_bin_centers_s[0]
        + dt / 2
    )

    # the other issue is that spikeinterface doesn't understand rigid interpolation for now
    # so, we have to turn our rigid estimate into a nonrigid estimate
    spatial_bins = motion_est.spatial_bin_centers_um
    displacement = motion_est.displacement
    if spatial_bins is None:
        # we had a rigid motion
        # spatial bins can be the edges of the probe?
        geom_y = rec.get_channel_locations()[:, 1]
        spatial_bins = [geom_y.min(), geom_y.max()]
        # make 2 copies of our displacement so that they appear to correspond to these spatial bins
        displacement = np.stack((displacement, displacement), axis=0)
        assert displacement.ndim == 2 and displacement.shape[0] == 2

    # now we can use correct_motion
    rec_interpolated = interpolate_motion(
        rec,
        displacement.T,
        temporal_bins,
        spatial_bins,
        border_mode=border_mode,
    )
    return rec_interpolated


def speed_limit_filter(me, speed_limit_um_per_s=5000.0):
    """Interpolate away outrageously huge jumps."""
    displacement = np.atleast_2d(me.displacement)
    speed = np.abs(np.gradient(displacement, me.time_bin_centers_s, axis=1))
    valid = speed <= speed_limit_um_per_s
    valid[[0, -1]] = True
    print(f"{valid.mean()=}")
    if valid.all():
        return me
    valid_lerp = [
        interp1d(me.time_bin_centers_s[v], d[v])
        for v, d in zip(valid, displacement)
    ]
    filtered_displacement = [vl(me.time_bin_centers_s) for vl in valid_lerp]

    return get_motion_estimate(
        filtered_displacement,
        time_bin_edges_s=me.time_bin_edges_s,
        time_bin_centers_s=me.time_bin_centers_s,
        spatial_bin_edges_um=me.spatial_bin_edges_um,
        spatial_bin_centers_um=me.spatial_bin_centers_um,
    )


def resample_to_new_time_bins(me, new_time_bin_centers_s=None):
    """Take a MotionEstimate and use its interpolation to"""
    displacement_up = me.disp_at_s(
        new_time_bin_centers_s, me.spatial_bin_centers_um, grid=True
    )
    return get_motion_estimate(
        displacement_up,
        time_bin_centers_s=new_time_bin_centers_s,
        spatial_bin_centers_um=me.spatial_bin_centers_um,
    )


# -- preprocessing helpers


def fill_gaps_along_depth(recording):
    """A naive method for filling missing channels in (especially LFP) recordings."""
    import spikeinterface.preprocessing as sppx

    # figure out where the gaps are and how big they are relative
    # to the rest of the channel spacings
    geom = recording.get_channel_locations()
    geom_y = geom[:, 1]
    dy = np.diff(geom_y)
    dy_unique = np.unique(dy)
    if dy_unique.size <= 1:
        # nothing to do, all channel spacings are the same
        return recording
    pitch = dy_unique.min()

    # figure out where the new channels should go
    new_locs = []
    for j in np.flatnonzero(dy > pitch):
        dyj = dy[j]
        n_add = int(dyj // pitch - 1)
        at_j = geom[geom_y == geom_y[j]].mean(0)
        at_next = geom[geom_y == geom_y[j] + dyj].mean(0)
        center_x = 0.5 * (at_j[0] + at_next[0])
        for k in range(n_add):
            new_locs.append([center_x, at_j[1] + (k + 1) * dyj])

    # use a spikeinterface preprocessing chain to interpolate
    # into these gaps
    rec = sppx.add_fake_channels(recording, len(new_locs), new_locs)
    rec = sppx.depth_order(rec)
    bad_channels = [id for id in rec.channel_ids if id.startswith("FakeChan")]
    return sppx.interpolate_bad_channels(rec, bad_channels)


# -- plotting


def show_raster(
    raster, spatial_bin_edges_um, time_bin_edges_s, ax, **imshow_kwargs
):
    """Display a spike activity raster as created with `spike_raster` below"""
    return ax.imshow(
        raster,
        extent=(*time_bin_edges_s[[0, -1]], *spatial_bin_edges_um[[0, -1]]),
        origin="lower",
        **imshow_kwargs,
    )


def show_spike_raster(amps, depths, times, ax, **imshow_kwargs):
    """Display a spike activity raster as created with `spike_raster` below"""
    raster, spatial_bin_edges_um, time_bin_edges_s = spike_raster(
        amps, depths, times
    )
    return ax.imshow(
        raster,
        extent=(*time_bin_edges_s[[0, -1]], *spatial_bin_edges_um[[0, -1]]),
        origin="lower",
        **imshow_kwargs,
    )


def plot_me_traces(
    me,
    ax,
    offset=0,
    depths_um=None,
    label=False,
    zero_times=False,
    t_start=None,
    t_end=None,
    **plot_kwargs,
):
    """Plot the displacement estimates for the MotionEstimate me as lines."""
    if depths_um is None:
        depths_um = me.spatial_bin_centers_um
    if depths_um is None:
        depths_um = [sum(ax.get_ylim()) / 2]

    times = me.time_bin_centers_s
    t_offset = times[0] if zero_times else 0
    if t_start is not None:
        times = times[times >= t_start]
    if t_end is not None:
        times = times[times <= t_end]

    lines = []
    for b, depth in enumerate(depths_um):
        disp = me.disp_at_s(times, depth_um=depth, grid=True)
        disp = disp.squeeze()
        if isinstance(label, str):
            lab = label
        else:
            lab = f"bin {b}" if label else None
        l = ax.plot(
            times - t_offset,
            depth + offset + disp,
            label=lab,
            **plot_kwargs,
        )
        lines.extend(l)
    return lines


def show_registered_raster(me, amps, depths, times, ax, **imshow_kwargs):
    """Plot a registered raster for the MotionEstimate me."""
    depths_reg = me.correct_s(times, depths)
    raster, spatial_bin_edges_um, time_bin_edges_s = spike_raster(
        amps, depths_reg, times
    )
    ax.imshow(
        raster,
        extent=(*time_bin_edges_s[[0, -1]], *spatial_bin_edges_um[[0, -1]]),
        origin="lower",
        **imshow_kwargs,
    )


def show_displacement_heatmap(
    me, ax, spatial_bin_centers_um=None, **imshow_kwargs
):
    """Plot a spatiotemporal heatmap of displacement for the MotionEstimate me."""
    if spatial_bin_centers_um is None:
        spatial_bin_centers_um = me.spatial_bin_centers_um

    displacement_heatmap = me.disp_at_s(
        me.time_bin_centers_s, spatial_bin_centers_um, grid=True
    )
    ax.imshow(
        displacement_heatmap,
        extent=(
            *me.time_bin_centers_s[[0, -1]],
            *spatial_bin_centers_um[[0, -1]],
        ),
        origin="lower",
        **imshow_kwargs,
    )


# lfp plotting helpers


def show_lfp_image(
    lfp_recording,
    start_sample,
    end_sample,
    ax,
    volts=False,
    seconds=True,
    microns=False,
    aspect="auto",
    batched_mode=False,
    traces=None,
    origin="lower",
    **imshow_kwargs,
):
    if traces is None and batched_mode:
        traces = np.concatenate(
            [
                lfp_recording.get_traces(
                    0, t0, min(end_sample, t0 + 10), return_scaled=volts
                )
                for t0 in range(start_sample, end_sample, 10)
            ]
        )
    elif traces is None:
        traces = lfp_recording.get_traces(
            0, start_sample, end_sample, return_scaled=volts
        )

    if seconds:
        times = lfp_recording.get_times(0)[start_sample:end_sample]
        extent_t = times[[0, -1]]
        xlabel = "time (seconds)"
    else:
        extent_t = start_sample, end_sample
        xlabel = "time (samples)"

    if microns:
        geom_y = lfp_recording.get_channel_locations()[:, 1]
        extent_y = geom_y.max(), geom_y.min()
        ylabel = "depth (microns)"
    else:
        extent_y = 0, lfp_recording.get_num_channels()
        ylabel = "channels"

    if origin == "lower":
        extent_y = extent_y[1], extent_y[0]

    extent = [*extent_t, *extent_y]
    im = ax.imshow(traces.T, extent=extent, aspect=aspect, origin=origin, **imshow_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return im


def show_lfp_me_traces(
    me,
    start_sample,
    end_sample,
    ax,
    label=False,
    seconds=True,
    depths_um=None,
    **plot_kwargs,
):
    times_s = me.time_bin_centers_s[start_sample:end_sample]
    times = times_s
    if not seconds:
        times = np.arange(start_sample, end_sample)

    if depths_um is None:
        depths_um = me.spatial_bin_centers_um
    if depths_um is None:
        depths_um = [sum(ax.get_ylim()) / 2]

    for b, depth in enumerate(depths_um):
        disp = me.disp_at_s(times_s, depth_um=depth)
        if isinstance(label, str):
            lab = label
        else:
            lab = f"bin {b}" if label else None
        (l,) = ax.plot(
            times,
            depth + disp,
            label=lab,
            **plot_kwargs,
        )

    return l


# -- bins / windows / rasters

def get_bins(x, bin_h):
    return np.arange(
        np.floor(x.min()),
        np.ceil(x.max()) + bin_h,
        bin_h,
    )


def get_windows(
    geom,
    win_step_um,
    win_sigma_um,
    spatial_bin_edges=None,
    spatial_bin_centers=None,
    margin_um=0,
    win_shape="rect",
    zero_threshold=1e-5,
    rigid=False,
):
    """Helper wrapper around si_get_windows below."""
    if win_shape == "gaussian":
        win_sigma_um = win_sigma_um / 2
    if margin_um is None:
        margin_um = -win_sigma_um

    windows, locs = si_get_windows(
        rigid=rigid,
        contact_pos=geom,
        spatial_bin_edges=spatial_bin_edges,
        spatial_bin_centers=spatial_bin_centers,
        margin_um=margin_um,
        win_step_um=win_step_um,
        win_sigma_um=win_sigma_um,
        win_shape=win_shape,
    )
    if windows.ndim == 1:
        windows = windows[None, :]

    windows /= windows.sum(axis=1, keepdims=True)
    windows[windows < zero_threshold] = 0
    windows /= windows.sum(axis=1, keepdims=True)

    return windows, locs


# spikeinterface's get_windows function, modified
# this has been copied here for the time being
# certain modifications are necessary:
#  - no error on small windows (or, smaller gaussian sigm)
#  - support input of bin centers rather than edges for LFP use case
#  - does not actually need bin_um, and this may not always exist under non-uniform bin spacing (i.e. lfp could be from probe with holes)
#  - should return arrays
#  - contact_pos[:, 1] in bin pos calc
# also, not implemented here, but shouldn't the margin logic be based
# on the spatial bins rather than the geometry? the way it is now,
# it makes nonrigid registration after rigid registration of an insertion
# recording impossible (or generally any iterated idea)
# from spikeinterface.sortingcomponents.motion_estimation import (
#     get_windows as si_get_windows,
# )


def si_get_windows(
    rigid,
    contact_pos,
    spatial_bin_edges=None,
    margin_um=0,
    win_step_um=400,
    win_sigma_um=450,
    win_shape="gaussian",
    spatial_bin_centers=None,
):
    """
    Generate spatial windows (taper) for non-rigid motion.
    For rigid motion, this is equivalent to have one unique rectangular window that covers the entire probe.
    The windowing can be gaussian or rectangular.

    Parameters
    ----------
    rigid : bool
        If True, returns a single rectangular window
    bin_um : float
        Spatial bin size in um
    contact_pos : np.ndarray
        Position of electrodes (num_channels, 2)
    spatial_bin_edges : np.array
        The pre-computed spatial bin edges
    margin_um : float
        The margin to extend (if positive) or shrink (if negative) the probe dimension to compute windows.=
    win_step_um : float
        The steps at which windows are defined
    win_sigma_um : float
        Sigma of gaussian window (if win_shape is gaussian)
    win_shape : float
        "gaussian" | "rect"

    Returns
    -------
    non_rigid_windows : list of 1D arrays
        The scaling for each window. Each element has num_spatial_bins values
    non_rigid_window_centers: 1D np.array
        The center of each window

    Notes
    -----
    Note that kilosort2.5 uses overlaping rectangular windows.
    Here by default we use gaussian window.

    """
    if spatial_bin_centers is None:
        spatial_bin_centers = 0.5 * (
            spatial_bin_edges[1:] + spatial_bin_edges[:-1]
        )
    n = spatial_bin_centers.size

    if rigid:
        # win_shape = 'rect' is forced
        non_rigid_windows = [np.ones(n, dtype="float64")]
        middle = (spatial_bin_centers[0] + spatial_bin_centers[-1]) / 2.0
        non_rigid_window_centers = np.array([middle])
    else:
        min_ = np.min(contact_pos[:, 1]) - margin_um
        max_ = np.max(contact_pos[:, 1]) + margin_um
        num_non_rigid_windows = int((max_ - min_) // win_step_um)
        border = ((max_ - min_) % win_step_um) / 2
        non_rigid_window_centers = (
            np.arange(num_non_rigid_windows + 1) * win_step_um + min_ + border
        )
        non_rigid_windows = []

        for win_center in non_rigid_window_centers:
            if win_shape == "gaussian":
                win = np.exp(
                    -((spatial_bin_centers - win_center) ** 2)
                    / (2 * win_sigma_um**2)
                )
            elif win_shape == "rect":
                win = np.abs(spatial_bin_centers - win_center) < (
                    win_sigma_um / 2.0
                )
                win = win.astype("float64")
            elif win_shape == "triangle":
                center_dist = np.abs(spatial_bin_centers - win_center)
                in_window = center_dist <= (win_sigma_um / 2.0)
                win = -center_dist
                win[~in_window] = 0
                win[in_window] -= win[in_window].min()
                win[in_window] /= win[in_window].max()

            non_rigid_windows.append(win)

    return np.array(non_rigid_windows), np.array(non_rigid_window_centers)


def get_window_domains(windows):
    """Array of windows -> list of slices where window > 0."""
    slices = []
    for w in windows:
        in_window = np.flatnonzero(w)
        slices.append(slice(in_window[0], in_window[-1] + 1))
    return slices


def spike_raster(
    amps,
    depths,
    times,
    bin_um=1.0,
    bin_s=1.0,
    spatial_bin_edges_um=None,
    time_bin_edges_s=None,
    amp_scale_fn=None,
    gaussian_smoothing_sigma_um=0,
    gaussian_smoothing_sigma_s=0,
    avg_in_bin=True,
    post_transform=None,
):
    """Create an image representation of spike activity

    Arguments
    ---------
    amps, depths, times : 1D np.arrays all of same length
        Amplitudes, depths (microns), times (seconds) of spike events
    bin_um : float
        Spatial bin size (microns)
    bin_s : float
        Temporal bin size (seconds)
    spatial_bin_edges_um, time_bin_edges_um : 1D arrays
        Optional. If you already know what bins you want to use, supply
        them here. Otherwise, bins are determined based on bin_um/bin_s
        and the depths and times.
    amp_scale_fn : None or a function
        Apply this function to amplitudes. None (default) means leave them as
        they are. For instance, you could use np.log1p to scale the amplitudes
        logarithmically. Or, np.ones_like to replace amplitudes with 1s.
    gaussian_smoothing_sigma_um, gaussian_smoothing_sigma_s
        Bandwidth for Gaussian smoothing of raster in space and time.
    avg_in_bin : bool
        If true, average the amplitudes in each bin. If false, sum them.
    post_transform : None or a function
        If supplied, this function is applied to the final raster before
        reurning.

    Returns
    -------
    raster : 2D np.array
        The n_depth_bins x n_time_bins image
    spatial_bin_edges_um : 1D np.array
        The n_depth_bins + 1 spatial bin edges
    time_bin_edges_s : 1D np.array
        The n_time_bins_bins + 1 temporal bin edges
    """
    assert amps.shape == depths.shape == times.shape
    assert amps.ndim == 1

    if spatial_bin_edges_um is None:
        spatial_bin_edges_um = get_bins(depths, bin_um)
    if time_bin_edges_s is None:
        time_bin_edges_s = get_bins(times, bin_s)

    if amp_scale_fn is None:
        weights = amps
    else:
        weights = amp_scale_fn(amps)

    if gaussian_smoothing_sigma_um:
        spatial_bin_edges_um_1um = np.arange(
            spatial_bin_edges_um[0],
            spatial_bin_edges_um[-1] + 1,
            1,
        )
        spatial_bin_centers_um_1um = 0.5 * (
            spatial_bin_edges_um_1um[1:] + spatial_bin_edges_um_1um[:-1]
        )
        r_up = np.histogram2d(
            depths,
            times,
            bins=(spatial_bin_edges_um_1um, time_bin_edges_s),
            weights=weights,
        )[0]
        if avg_in_bin:
            r_up /= np.maximum(
                1,
                np.histogram2d(
                    depths,
                    times,
                    bins=(spatial_bin_edges_um_1um, time_bin_edges_s),
                )[0],
            )

        r_up = gaussian_filter1d(r_up, gaussian_smoothing_sigma_um, axis=0)
        r = np.empty(
            (spatial_bin_edges_um.size - 1, time_bin_edges_s.size - 1),
            dtype=r_up.dtype,
        )
        for i, (bin_start, bin_end) in enumerate(
            zip(spatial_bin_edges_um, spatial_bin_edges_um[1:])
        ):
            in_bin = np.flatnonzero(
                (bin_start <= spatial_bin_centers_um_1um)
                & (bin_end > spatial_bin_centers_um_1um)
            )
            r[i] = r_up[in_bin].sum(0) / (in_bin.size if avg_in_bin else 1)
    else:
        r = np.histogram2d(
            depths,
            times,
            bins=(spatial_bin_edges_um, time_bin_edges_s),
            weights=weights,
        )[0]
        if avg_in_bin:
            r /= np.maximum(
                1,
                np.histogram2d(
                    depths,
                    times,
                    bins=(spatial_bin_edges_um, time_bin_edges_s),
                )[0],
            )

    if post_transform is not None:
        r = post_transform(r)

    if gaussian_smoothing_sigma_s:
        r = gaussian_filter1d(r, gaussian_smoothing_sigma_s / bin_s, axis=1)

    return r, spatial_bin_edges_um, time_bin_edges_s
