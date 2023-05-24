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
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d


# -- motion estimate helper classes


class MotionEstimate:
    """MotionEstimate

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
        raise NotImplementedError

    def correct_s(self, t_s, depth_um, grid=False):
        return depth_um - self.disp_at_s(t_s, depth_um, grid=grid)


class RigidMotionEstimate(MotionEstimate):
    def __init__(
        self,
        displacement,
        time_bin_edges_s=None,
        time_bin_centers_s=None,
    ):
        displacement = np.asarray(displacement).squeeze()

        assert displacement.ndim == 1
        if time_bin_edges_s is not None:
            assert (1 + displacement.shape[0],) == time_bin_edges_s.shape
        else:
            assert time_bin_centers_s is not None
            assert time_bin_centers_s.shape == displacement.shape

        super().__init__(
            displacement.squeeze(),
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
        assert not grid
        return self.lerp(t_s)


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

        self.lerp = RectBivariateSpline(
            self.spatial_bin_centers_um,
            self.time_bin_centers_s,
            self.displacement,
            kx=1,
            ky=1,
        )

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        return self.lerp(
            np.clip(depth_um, self.d_low, self.d_high),
            np.clip(t_s, self.t_low, self.t_high),
            grid=grid,
        )


class IdentityMotionEstimate(MotionEstimate):
    """The motion estimate with no motion."""

    def __init__(self):
        super().__init__(None)

    def disp_at_s(self, t_s, depth_um=None, grid=False):
        return np.zeros_like(t_s)


class ComposeMotionEstimates(MotionEstimate):
    """Compose motion estimates, if each was estimated from the previous' corrections"""

    def __init__(self, *motion_estimates):
        super().__init__(None)
        self.motion_estimates = motion_estimates
        self.time_bin_edges_s = motion_estimates[0].time_bin_edges_s
        self.time_bin_centers_s = motion_estimates[0].time_bin_centers_s

    def disp_at_s(self, t_s, depth_um=None, grid=False):
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
    """Helper function for construction MotionEstimates.

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
    assert any(a is not None for a in (spatial_bin_edges_um, spatial_bin_centers_um))

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
    displacement_upsampled = (windows.T @ (displacement * window_weights)) / normalizer

    return NonrigidMotionEstimate(
        displacement_upsampled,
        time_bin_edges_s=time_bin_edges_s,
        time_bin_centers_s=time_bin_centers_s,
        spatial_bin_edges_um=spatial_bin_edges_um,
        spatial_bin_centers_um=spatial_bin_centers_um,
    )


def speed_limit_filter(me, speed_limit_um_per_s=5000.0):
    """Interpolate away outrageously huge jumps."""
    displacement = np.atleast_2d(me.displacement)
    speed = np.abs(np.gradient(displacement, me.time_bin_centers_s, axis=1))
    valid = speed <= speed_limit_um_per_s
    valid[[0, -1]] = True
    print(f"{valid.mean()=}")
    if valid.all():
        return me
    valid_lerp = [interp1d(me.time_bin_centers_s[v], d[v]) for v, d in zip(valid, displacement)]
    filtered_displacement = [vl(me.time_bin_centers_s) for vl in valid_lerp]

    return get_motion_estimate(
        filtered_displacement,
        time_bin_edges_s=me.time_bin_edges_s,
        time_bin_centers_s=me.time_bin_centers_s,
        spatial_bin_edges_um=me.spatial_bin_edges_um,
        spatial_bin_centers_um=me.spatial_bin_centers_um,
    )


# -- plotting


def show_raster(raster, spatial_bin_edges_um, time_bin_edges_s, ax, **imshow_kwargs):
    """Display a spike activity raster as created with `spike_raster` below"""
    ax.imshow(
        raster,
        extent=(*time_bin_edges_s[[0, -1]], *spatial_bin_edges_um[[0, -1]]),
        origin="lower",
        **imshow_kwargs,
    )


def plot_me_traces(me, ax, offset=0, depths_um=None, label=False, zero_times=False, **plot_kwargs):
    """Plot the displacement estimates for the MotionEstimate me as lines."""
    if depths_um is None:
        depths_um = me.spatial_bin_centers_um
    if depths_um is None:
        depths_um = [sum(ax.get_ylim()) / 2]

    t_offset = me.time_bin_centers_s[0] if zero_times else 0

    for b, depth in enumerate(depths_um):
        disp = me.disp_at_s(me.time_bin_centers_s, depth_um=depth)
        if isinstance(label, str):
            lab = label
        else:
            lab = f"bin {b}" if label else None
        ax.plot(
            me.time_bin_centers_s - t_offset,
            depth + offset + disp,
            label=lab,
            **plot_kwargs,
        )


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


def show_displacement_heatmap(me, ax, spatial_bin_centers_um=None, **imshow_kwargs):
    """Plot a spatiotemporal heatmap of displacement for the MotionEstimate me."""
    if spatial_bin_centers_um is None:
        spatial_bin_centers_um = me.spatial_bin_centers_um

    displacement_heatmap = me.disp_at_s(me.time_bin_centers_s, spatial_bin_centers_um, grid=True)
    ax.imshow(
        displacement_heatmap,
        extent=(
            *me.time_bin_centers_s[[0, -1]],
            *spatial_bin_centers_um[[0, -1]],
        ),
        origin="lower",
        **imshow_kwargs,
    )


# -- bins / windows / rasters


def get_bins(depths, times, bin_um, bin_s):
    spatial_bin_edges_um = np.arange(
        np.floor(depths.min()),
        np.ceil(depths.max()) + bin_um,
        bin_um,
    )
    time_bin_edges_s = np.arange(
        np.floor(times.min()),
        np.ceil(times.max()) + bin_s,
        bin_s,
    )
    return spatial_bin_edges_um, time_bin_edges_s


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
        spatial_bin_centers = 0.5 * (spatial_bin_edges[1:] + spatial_bin_edges[:-1])
    n = spatial_bin_centers.size

    if rigid:
        # win_shape = 'rect' is forced
        non_rigid_windows = [np.ones(n, dtype="float64")]
        middle = (spatial_bin_centers[0] + spatial_bin_centers[-1]) / 2.0
        non_rigid_window_centers = np.array([middle])
    else:
        min_ = np.min(contact_pos) - margin_um
        max_ = np.max(contact_pos) + margin_um
        num_non_rigid_windows = int((max_ - min_) // win_step_um)
        border = ((max_ - min_) % win_step_um) / 2
        non_rigid_window_centers = (
            np.arange(num_non_rigid_windows + 1) * win_step_um + min_ + border
        )
        non_rigid_windows = []

        for win_center in non_rigid_window_centers:
            if win_shape == "gaussian":
                win = np.exp(
                    -((spatial_bin_centers - win_center) ** 2) / (2 * win_sigma_um**2)
                )
            elif win_shape == "rect":
                win = np.abs(spatial_bin_centers - win_center) < (win_sigma_um / 2.0)
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

    if (spatial_bin_edges_um is None) or (time_bin_edges_s is None):
        _spatial_bin_edges_um, _time_bin_edges_s = get_bins(
            depths, times, bin_um, bin_s
        )
        if spatial_bin_edges_um is None:
            spatial_bin_edges_um = _spatial_bin_edges_um
        if time_bin_edges_s is None:
            time_bin_edges_s = _time_bin_edges_s

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
        spatial_bin_centers_um_1um = 0.5 * (spatial_bin_edges_um_1um[1:] + spatial_bin_edges_um_1um[:-1])
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
        r = np.empty((spatial_bin_edges_um.size - 1, time_bin_edges_s.size - 1), dtype=r_up.dtype)
        for i, (bin_start, bin_end) in enumerate(zip(spatial_bin_edges_um, spatial_bin_edges_um[1:])):
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
