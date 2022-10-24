import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import colorcet as cc
from sklearn.metrics.pairwise import cosine_similarity

from scipy.io import loadmat
from collections import namedtuple

from . import ibme, ibme_corr

DSMeta = namedtuple(
    "DSMeta", ["name", "has_csd", "t_start", "t_end", "ins_depth"]
)


def load_chanmap(chanmap_mat):
    try:
        h = loadmat(chanmap_mat)
    except ValueError:
        with h5py.File(chanmap_mat) as h5:
            h = {k: h5[k][...] for k in h5}
    return np.c_[h["xcoords"].squeeze(), h["ycoords"].squeeze()]


def template_based_registration(
    y, t, a, n_iter=10, disp=None, batch_size=32, device=None
):
    p = 0

    # keep initial depths around
    y0 = y

    for i in range(n_iter):
        # get the motion raster and its mean over time
        r, dd, tt = ibme.fast_raster(a, y, t)
        r_mean = r.mean(axis=1)

        # we pad with a time dimension of length 1 here so that
        # we can use calc_corr_decent_pair
        r_mean = r_mean[:, None]

        # find best displacements from the template for all time bins
        # and add to our total displacement estimate
        D, C = ibme_corr.calc_corr_decent_pair(
            r_mean, r, disp=disp, batch_size=batch_size, device=device
        )
        p = p - D.squeeze()

        # interpolate `p` to displace the underlying points
        y = ibme.warp_rigid(y0, t, tt, p)

    return p, y


def spikes_and_reg_from_rez2(rez2_mat):
    with h5py.File(rez2_mat) as rez2:
        # stores spike times and other attributes
        st0 = rez2["/rez/st0"][:]

        # displacement estimate
        dshift = rez2["/rez/dshift"][:].squeeze()

    samples = st0[0].astype(int)
    t = st0[0] / 30_000
    y = st0[1]
    a = st0[2]
    # zero-index the batching
    batch = st0[4].astype(int) - 1

    return samples, t, y, a, batch, dshift


def show_dc(D, C):
    fig, axes = plt.subplots(ncols=2, figsize=(6, 4), sharey=True)
    aa, ab = axes

    dmax = np.abs(D).max()
    im = aa.imshow(D, vmin=-dmax, vmax=dmax, cmap=plt.cm.seismic)
    plt.colorbar(im, ax=aa, shrink=0.3, label="displacement (um)")

    im = ab.imshow(C, vmin=0, cmap=plt.cm.magma)
    plt.colorbar(im, ax=ab, shrink=0.3, label="correlation")

    return fig, axes


def entropy_1d(y, bin_size_um=1):
    bins = np.arange(
        np.floor(y.min()), np.ceil(y.max()) + bin_size_um, bin_size_um
    )
    hist, *_ = np.histogram(y, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return -np.sum(p * np.log(p))


def total_corr(y, t, a):
    r, dd, tt = ibme.fast_raster(a, y, t)
    # corr = np.corrcoef(r.T)
    corr = cosine_similarity(r.T)
    assert corr.shape == (*tt.shape, *tt.shape)
    return np.abs(corr).sum() / corr.size


def total_std(y, t, a):
    r, dd, tt = ibme.fast_raster(a, y, t)
    return (r.mean(axis=1) * r.std(axis=1)).sum() / (r.mean(axis=1).sum())


def mi(ri, rj, pi, pj, bins):
    joint, *_ = np.histogram2d(ri, rj, bins, density=True)
    pind = pi[:, None] * pj[None, :]
    which = np.nonzero(joint > 0)
    return np.sum(joint[which] * np.log(joint[which] / pind[which]))


def total_mi(y, t, a, nbins=50, dt=1):
    r, dd, tt = ibme.fast_raster(a, y, t / dt)
    bins = np.geomspace(1, a.max(), num=nbins)
    bins[0] = 0
    pis = [
        np.histogram(r[:, i], bins=bins, density=True)[0]
        for i in range(r.shape[1])
    ]
    ijs = [(i, j) for i in range(r.shape[1]) for j in range(i + 1, r.shape[1])]
    mis = Parallel(8)(
        delayed(mi)(r[:, i], r[:, j], pis[i], pis[j], bins)
        for i, j in tqdm(ijs)
    )
    return np.mean(mis)


def showmetrics(
    t,
    a,
    ys,
    names,
):
    records = []
    for n, y in zip(names, ys):
        records.append(
            dict(
                method=n,
                metric="mean_abs_corr",
                value=total_corr(y, t, a),
            )
        )
        records.append(
            dict(
                method=n,
                metric="h_y",
                value=entropy_1d(y),
            )
        )
        records.append(
            dict(
                method=n,
                metric="total_std",
                value=total_std(y, t, a),
            )
        )

    g = sns.catplot(
        data=pd.DataFrame.from_records(records),
        x="method",
        y="value",
        col="metric",
        kind="point",
        sharey=False,
    )
    return g


def rrasters(ys, names, t, a):
    fig, axes = plt.subplots(
        nrows=len(ys),
        figsize=(7, 3.5 * len(ys)),
        sharex=True,
        gridspec_kw=dict(hspace=0.1),
    )

    for y, name, ax in zip(ys, names, axes.flat):
        print(name, flush=True)
        r, dd, tt = ibme.fast_raster(a, y, t)
        ax.imshow(
            r,
            vmax=15,
            extent=[tt[0], tt[-1], dd[0], dd[-1]],
            aspect="auto",
            pal=gpal["apcm"],
        )
        ax.set_ylabel(f"{name} depth (\\textmu{{}}m)")
    ax.set_xlabel("time (s)")

    return fig, axes


hero_mos = """\
aaaaaaaaaaa....pppppppppppp
aaaaaaaaaaa.b..pppppppppppp
aaaaaaaaaaa.b..pppppppppppp
aaaaaaaaaaa.b..............
aaaaaaaaaaa.b..qqqqqqqqqqqq
aaaaaaaaaaa.b..qqqqqqqqqqqq
aaaaaaaaaaa....qqqqqqqqqqqq
...........................
...............uuuuuggvvvvv
xxxx.yyyy.zzzz.uuuuu..vvvvv
xxxx.yyyy.zzzz.uuuuu..vvvvv
"""


bighpad = 1.5
hpad = 1.0
tinypad = 0.25
wpad = 0.5
tinywpad = 0.25


def hero(
    y,
    y_ks,
    y_csd,
    y_ap,
    t,
    a,
    name,
    dshift,
    p_csd,
    p_ap,
    tt_ks,
    tt_csd,
    tt_ap,
    csd,
    tstarts,
    metrics_df,
):
    rem = plt.rcParams["font.size"]

    r_ks, dd_ks, _ = ibme.fast_raster(a, y_ks, t)
    r_ap, dd_ap, _ = ibme.fast_raster(a, y_ap, t)
    r_csd, dd_csd, _ = ibme.fast_raster(a, y_csd, t)

    p_ap_250 = interp1d(tt_ap, p_ap, fill_value="extrapolate")(tt_csd) / 20 / 2
    p_ks_250 = (
        interp1d(tt_ks, -dshift, fill_value="extrapolate")(tt_csd) / 20 / 2
    )

    left_wratios = np.array(
        [0.5] * 4 + [1] + [0.5] * 4 + [1] + [1.55, 0.1, 0.25, 0.1]
    )
    right_wratios = np.array([1] * 5 + [tinywpad / 2, tinywpad / 2] + [1] * 5)

    fig, axes = plt.subplot_mosaic(
        hero_mos,
        figsize=(7, 4),
        gridspec_kw=dict(
            hspace=0,
            height_ratios=[
                1,
                1,
                hpad,
                tinypad,
                1,
                1,
                hpad,
                hpad,
                0.5,
                1.5,
                hpad,
            ],
            width_ratios=list(12 * left_wratios / left_wratios.sum())
            + [bighpad]
            + list(12 * right_wratios / right_wratios.sum()),
        ),
    )

    # panel A: raster with traces
    r, dd, tt = ibme.fast_raster(a, y, t)
    print(r.shape, tt[-1])
    im = axes["a"].imshow(
        r,
        vmax=15,
        # extent=[tt[0], tt[-1] - 1, dd[0], dd[-1]],
        aspect="auto",
        cmap=gpal["apcm"],
        # origin="lower",
    )
    plt.colorbar(im, cax=axes["b"], label="amplitude (s.u.)", shrink=0.5)
    axes["b"].set_yticks([0, 15])
    axes["b"].yaxis.set_label_coords(2.25, 0.5)
    yticks = axes["a"].get_yticks()
    print(yticks)
    axes["a"].set_yticks([0] + [dd.max()], [0] + [3820])
    axes["a"].set_ylabel("depth (\\textmu{{}}m)", labelpad=-2 * rem)
    axes["a"].set_xticks([0] + tstarts + [tt[-1]])
    axes["a"].set_xlabel("time (s)", labelpad=-rem)
    axes["a"].plot(
        tt_ks,
        800 + (dshift - dshift.mean()),
        color=gpal["aplines"][0],
        label="KS",
        lw=1,
        zorder=2,
    )
    axes["a"].plot(
        tt_csd,
        1600 + 20 * (p_csd - p_csd.mean()),
        color=gpal["aplines"][1],
        label="Ours (CSD)",
        lw=1,
        zorder=2,
    )
    axes["a"].plot(
        tt_ap,
        2400 + (p_ap - p_ap.mean()),
        color=gpal["aplines"][2],
        label="Ours (AP)",
        lw=1,
        zorder=2,
    )
    axes["a"].legend(ncol=3, loc="lower left", fancybox=False, framealpha=1)

    # panel B: corrected rasters
    axes["p"].imshow(
        r_csd,
        vmax=15,
        # extent=[tt[0], tt[-1], dd_csd[0], dd_csd[-1]],
        aspect="auto",
        cmap=gpal["apcm"],
        # origin="lower",
    )
    axes["p"].set_yticks([])
    axes["p"].set_xticks([])
    axes["p"].set_ylabel("Ours (CSD) reg.\\ depth (\\textmu{{}}m)")

    axes["q"].imshow(
        r_ap,
        vmax=15,
        # extent=[tt[0], tt[-1], dd_ap[0], dd_ap[-1]],
        aspect="auto",
        cmap=gpal["apcm"],
        # origin="lower",
    )
    axes["q"].set_yticks([])
    axes["q"].set_ylabel("Ours (AP) reg.\\ depth (\\textmu{{}}m)")
    axes["q"].set_xticks([0] + tstarts + [r_ap.shape[1] - 1])
    axes["q"].set_xlabel("time (s, 1Hz sampling)", labelpad=-rem)

    for k in "apq":
        for t0 in tstarts:
            axes[k].axvline(t0, color=gpal["vline"], ls=":", lw=1)
            axes[k].axvline(t0 + 40, color=gpal["vline"], ls=":", lw=1)

    # panel C: metrics
    names = ["None", "KS", "AP", "CSD"]
    # tcs = [total_corr(y_, t, a) for y_ in [y, y_ks, y_ap, y_csd]]
    # hs = [entropy_1d(y_) for y_ in [y, y_ks, y_ap, y_csd]]
    # tss = [total_std(y_, t, a) for y_ in [y, y_ks, y_ap, y_csd]]
    for k, met in zip(
        "xyz",
        ["total_corr", "total_mi", "total_std"],
    ):
        rows = metrics_df[metrics_df["met"] == met]
        dsets = np.sort(np.unique(rows["dataset"]))
        metname = rows["metric"].values[rows["met"].values == met][0]
        vals = []
        for ds, col in zip(dsets, gpal["metlines"]):
            dsrows = rows[rows["dataset"] == ds]
            nas_vals = [
                (na, dsrows["value"].values[dsrows["method"].values == na][0])
                for na in names
                if (dsrows["method"].values == na).any()
            ]
            thenames, thevals = map(list, zip(*nas_vals))
            vals += thevals
            axes[k].plot(thevals, label=ds, color=col)
        # axes[k].plot(vals, color="k", marker=".")
        axes[k].set_xticks(range(len(names)), names, rotation=90)
        axes[k].set_yticks(
            [min(vals), max(vals)], [f"{min(vals):.2g}", f"{max(vals):.2g}"]
        )
        axes[k].set_ylabel(metname)
        axes[k].yaxis.set_label_coords(-0.15, 0.5)
    axes["y"].legend(fancybox=False, ncol=3, loc=(-0.8, 1.1))

    # panel D: csd chunks with traces
    sl = slice(tstarts[0] * 250, tstarts[0] * 250 + 40 * 250)
    chunk0 = csd[sl]
    vmax = np.percentile(np.abs(chunk0), 99)
    axes["u"].imshow(
        chunk0.T, aspect="auto", cmap=gpal["csdcm"], vmin=-vmax, vmax=vmax
    )
    axes["u"].set_xticks(
        (0, chunk0.shape[0] - 1),
        (tstarts[0], tstarts[0] + 40),
    )
    axes["u"].set_yticks([0, csd.shape[1] - 1], [0, 3820])
    axes["u"].set_ylabel("depth (\\textmu{{}}m)", labelpad=-2 * rem)
    axes["u"].set_xlabel("time (s, 250Hz sampling)", labelpad=-rem)
    axes["u"].plot(
        csd.shape[1] / 4 + p_ks_250[sl] - p_ks_250[sl].mean(),
        color=gpal["aplines"][0],
        lw=0.5,
    )
    axes["u"].plot(
        2 * csd.shape[1] / 4 + p_csd[sl] - p_csd[sl].mean(),
        color=gpal["aplines"][1],
        lw=0.5,
    )
    axes["u"].plot(
        3 * csd.shape[1] / 4 + p_ap_250[sl] - p_ap_250[sl].mean(),
        color=gpal["aplines"][2],
        lw=0.5,
    )

    sl = slice(tstarts[1] * 250, tstarts[1] * 250 + 40 * 250)
    chunk1 = csd[sl]
    vmax = np.percentile(np.abs(chunk1.T), 99)
    axes["v"].imshow(
        chunk1.T, aspect="auto", cmap=gpal["csdcm"], vmin=-vmax, vmax=vmax
    )
    axes["v"].set_xticks(
        (0, chunk1.shape[0] - 1),
        (tstarts[1], tstarts[1] + 40),
    )
    axes["v"].set_yticks([])
    axes["v"].set_xlabel("time (s, 250Hz sampling)", labelpad=-rem)
    axes["v"].plot(
        csd.shape[1] / 4 + p_ks_250[sl] - p_ks_250[sl].mean(),
        color=gpal["aplines"][0],
        lw=0.5,
    )
    axes["v"].plot(
        2 * csd.shape[1] / 4 + p_csd[sl] - p_csd[sl].mean(),
        color=gpal["aplines"][1],
        lw=0.5,
    )
    axes["v"].plot(
        3 * csd.shape[1] / 4 + p_ap_250[sl] - p_ap_250[sl].mean(),
        color=gpal["aplines"][2],
        lw=0.5,
    )

    for k, l, dx, dy in zip(
        # "apxu", "ABCD", [-0.05, -0.025, -0.2, -0.055], [1.1, 1.195, 1.15, 1.18]
        "aaaa",
        "ABCD",
        [-0.05, 1.18, -0.05, 1.18],
        [0.98, 0.98, -0.1, -0.1],
    ):
        axes[k].text(
            dx,
            dy,
            f"$\\mathsf{{{l}}}$",
            transform=axes[k].transAxes,
            fontsize=2 * rem,
            fontweight="bold",
            va="top",
            ha="right",
        )

    axes["g"].axis("off")

    return fig, axes


# Algorithmic multi panel fig:
# A: Timing of online algorithm for CSD. Show quadratic scaling of previous algorithm and linear scaling of online algorithm in units of 1x realtime
# B: Adaptive threshold selection. Plot estimated motion traces on top of CSD chunk for range of parameters + adaptive selected parameter?
# C (?): prior? show it removes some glitches and does not cause shrinkage?

mos_algo_left = """\
aa.lll.mm.w
aa.lll.mm.w
aa.lll.mm.w
...lll....w
bb.lll.nn.w
bb.lll.nn..
bb.lll.nn.x
..........x
.ooooo.qq.x
gooooo.qq.x
gppppp.qq.x
.ppppp.qq.x
"""


def rolling_mean_std(x, w_rad=5):
    mean = []
    std = []
    for i in range(w_rad, len(x) - w_rad):
        win = x[i - w_rad : i + w_rad]
        mean.append(np.mean(win))
        std.append(np.std(win))
    mean = [mean[0]] * w_rad + mean + [mean[-1]] * w_rad
    std = [std[0]] * w_rad + std + [std[-1]] * w_rad
    return np.array(mean), np.array(std)


def ciline(mean, std, ax, color=None, plot_kwargs=dict(lw=1)):
    ax.plot(mean, color=color, **plot_kwargs)
    ax.fill_between(
        np.arange(len(mean)), mean - std, mean + std, color=color, alpha=0.5
    )


pal0 = dict(
    apcm=plt.cm.viridis,
    csdcm=plt.cm.bone,
    aplines=["w", plt.cm.RdPu(0.5), plt.cm.RdPu(0.2)],
    vline="w",
)

pal1 = dict(
    apcm=plt.cm.binary,
    # csdcm=plt.cm.bone,
    csdcm=sns.dark_palette("#9cf", as_cmap=True),
    aplines=["violet", "red", "orange"],
    metlines=np.array(sns.color_palette("husl", 8))[[1, 3, 5]],
    vline="k",
)

gpal = pal1


def algo(
    # panel A
    # panel B
    ap_glitch,
    p_glitch0,
    p_glitch1,
    Dglitch,
    Cglitch,
    p2_full,
    p2_full_nolambd,
    p9_full,
    p9_full_nolambd,
    r,
    c2p_ap,
    c_ad,
    p_ad,
    csdchunk,
    c2p_csd,
    c_ad_csd,
    p_ad_csd,
):
    rem = plt.rcParams["font.size"]
    paw = [0.25, 1.75]
    pbw = [1, 1, 0.5, 0.75, 1, 1]

    # fig = plt.figure(constrained_layout=True, figsize=(7, 4))
    # fig_left, fig_right = fig.subfigures(nrows=1, ncols=2, width_ratios=[5, 2])

    fig, axes_left = plt.subplot_mosaic(
        mos_algo_left,
        gridspec_kw=dict(
            hspace=0,
            wspace=0,
            width_ratios=paw + [wpad] + pbw + [0.75, 3],
            # height_ratios=,
        ),
        figsize=(7, 4),
    )

    # axes_right = fig_right.subplot_mosaic(
    #     mos_algo_right,
    #     gridspec_kw=dict(
    #         height_ratios=[1, 0.1, 1],
    #     )
    # )

    axes = {**axes_left}

    # panel B: glitches, prior, shrinkage
    axes["l"].imshow(
        ap_glitch[100:200],
        aspect="auto",
        vmax=20,
        interpolation="nearest",
        cmap=gpal["apcm"],
    )
    axes["l"].plot(
        ap_glitch[100:200].shape[0] / 2 + p_glitch0,
        lw=1,
        color=gpal["aplines"][1],
        label="$\\lambda\\to\\infty$",
    )
    axes["l"].plot(
        ap_glitch[100:200].shape[0] / 2 + p_glitch1,
        lw=1,
        ls="--",
        color=gpal["aplines"][2],
        label="$\\lambda=1$",
    )
    axes["l"].legend(fancybox=False, loc="upper right", framealpha=1)
    axes["l"].set_yticks([])
    axes["l"].set_ylabel("depth (detail zoom)")
    axes["l"].set_xticks([0, ap_glitch.shape[1] - 1], [0, ap_glitch.shape[1]])
    axes["l"].set_xlabel("time (s, 1Hz sampling)", labelpad=-0.8 * rem)

    im = axes["m"].imshow(Cglitch, vmin=0, cmap=plt.cm.magma)
    cbar = plt.colorbar(im, ax=axes["m"], shrink=0.9, label="correlation")
    cbar.ax.set_yticks([0, 1])
    cbar.ax.yaxis.set_label_coords(3.5, 0.5)
    axes["m"].set_xticks([0, ap_glitch.shape[1] - 1], [0, ap_glitch.shape[1]])
    axes["m"].set_yticks([0, ap_glitch.shape[1] - 1], [0, ap_glitch.shape[1]])
    axes["m"].set_xlabel("time (s)", labelpad=-0.8 * rem)
    axes["m"].set_ylabel("time (s)", labelpad=-1.6 * rem)

    im = axes["n"].imshow(Dglitch, vmin=-50, vmax=50, cmap=plt.cm.seismic)
    cbar = plt.colorbar(
        im, ax=axes["n"], shrink=0.9, label="displacment (\\textmu{{}}m)"
    )
    cbar.ax.set_yticks([-50, 50])
    cbar.ax.yaxis.set_label_coords(3.5, 0.5)
    axes["n"].set_xticks([0, ap_glitch.shape[1] - 1], [0, ap_glitch.shape[1]])
    axes["n"].set_yticks([0, ap_glitch.shape[1] - 1], [0, ap_glitch.shape[1]])
    axes["n"].set_xlabel("time (s)", labelpad=-0.8 * rem)
    axes["n"].set_ylabel("time (s)", labelpad=-1.6 * rem)

    c2 = plt.cm.Greens
    c9 = plt.cm.Purples
    # axes["o"].scatter(np.arange(len(p2_full)), p2_full_nolambd, s=2, marker=".", facecolor=c2(0.5))
    # axes["o"].scatter(np.arange(len(p2_full)), p2_full, s=1, marker=".", facecolor=c2(0.2))
    # axes["p"].scatter(np.arange(len(p9_full)), p9_full_nolambd, s=2, marker=".", facecolor=c9(0.5))
    # axes["p"].scatter(np.arange(len(p9_full)), p9_full, s=1, marker=".", facecolor=c9(0.2))
    axes["o"].plot(
        np.arange(len(p2_full)), -p2_full_nolambd, lw=1, color="red"
    )
    axes["o"].plot(np.arange(len(p2_full)), -p2_full, lw=1, color=c2(0.5))
    axes["p"].plot(
        np.arange(len(p9_full)), -p9_full_nolambd, lw=1, color="red"
    )
    axes["p"].plot(np.arange(len(p9_full)), -p9_full, lw=1, color=c9(0.5))
    axes["o"].set_xticks([])
    axes["p"].set_xticks([])
    axes["p"].set_xlabel("time")
    axes["o"].set_yticks([-20, 0, 20])
    axes["p"].set_yticks([-20, 0, 20])
    axes["g"].axis("off")
    axes["g"].text(-1.75, 0.12, "displacment (\\textmu{{}}m)", rotation=90)
    # axes["g"].set_ylabel("depth (\\textmu{{}}m)")

    axes["q"].scatter(
        np.abs(p2_full_nolambd),
        np.abs(p2_full),
        color=c2(0.5),
        marker="x",
        s=5,
        alpha=0.5,
        label="Pt02",
        rasterized=True,
    )
    axes["q"].scatter(
        np.abs(p9_full_nolambd),
        np.abs(p9_full),
        color=c9(0.5),
        marker="x",
        s=5,
        alpha=0.5,
        label="Pt09",
        rasterized=True,
    )
    axes["q"].plot(
        np.arange(50), np.arange(50), color="k", lw=1, label="$y=x$"
    )
    axes["q"].legend(loc="lower right", fancybox=False)
    axes["q"].set_xticks([0, 49], [0, 50])
    axes["q"].set_xlabel(
        "abs.\\ displacment, $\\lambda\\to\\infty$", labelpad=-0.8 * rem
    )
    axes["q"].set_yticks([0, 49], [0, 50])
    axes["q"].set_ylabel(
        "abs.\\ displacment, $\\lambda=1$", labelpad=-1.2 * rem
    )

    axes["a"].text(0.5, 0.5, "timings in AP (us v KS)", ha="center")
    axes["b"].text(0.5, 0.5, "timings in CSD (online/offline)", ha="center")

    # panel : adaptive mincorr
    r_show = r[250:600, 750:1250]
    axes["w"].imshow(r_show, cmap=gpal["apcm"], aspect="auto", vmax=15)
    axes["w"].set_yticks([0, r_show.shape[0] - 1], [250, 600])
    axes["w"].set_xticks([0, r_show.shape[1] - 1], [750, 1250])
    pal = sns.color_palette("husl", len(c2p_ap) + 1)
    offsets = np.linspace(50, 600 - 250 - 50, num=len(pal))
    for (c, p), col, offset in zip(
        sorted({**c2p_ap, **{c_ad: p_ad}}.items(), key=lambda x: x[0]),
        pal,
        offsets,
    ):
        label = f"${c:0.2f}^*$" if c == c_ad else f"{c:0.2f}"
        axes["w"].plot(
            np.arange(r_show.shape[1]),
            offset + p[750:1250],
            color="w",
            lw=2,
            zorder=1,
        )
        axes["w"].scatter(
            np.arange(r_show.shape[1]),
            offset + p[750:1250],
            color=col,
            s=2,
            marker=".",
            linewidths=0,
            label=label,
            zorder=2,
        )
        # pm, ps = rolling_mean_std(p)
        # ciline(offset + pm[750:1250], 10 * ps[750:1250], ax=axes["w"], plot_kwargs=dict(lw=1, label=label), color=col)
    axes["w"].legend(
        loc="upper left",
        framealpha=1,
        fancybox=False,
        title="min corr.\\ $\\theta$",
    )
    axes["w"].set_ylabel("depth (\\textmu{{}}m)", labelpad=-2 * rem)
    axes["w"].set_xlabel("time (s, 1Hz sampling)", labelpad=-rem)

    csdchunk = csdchunk[20 * 250 :]
    axes["x"].imshow(csdchunk.T, cmap=gpal["csdcm"], aspect="auto")
    axes["x"].set_yticks([0, csdchunk.shape[1] - 1], [0, 3820])
    axes["x"].set_xticks([0, csdchunk.shape[0] - 1], [20, 40])
    pal = sns.color_palette("husl", len(c2p_csd) + 1)
    offsets = np.linspace(40, 192 - 40, num=len(pal))
    for (c, p), col, offset in zip(
        sorted(
            {**c2p_csd, **{c_ad_csd: p_ad_csd}}.items(), key=lambda x: x[0]
        ),
        pal,
        offsets,
    ):
        label = f"${c:0.2f}^*$" if c == c_ad_csd else f"{c:0.2f}"
        axes["x"].plot(
            np.arange(csdchunk.shape[0]),
            offset + p[20 * 250 :],
            color="w",
            lw=2,
            zorder=1,
        )
        axes["x"].scatter(
            np.arange(csdchunk.shape[0]),
            offset + p[20 * 250 :],
            color=col,
            s=2,
            marker=".",
            linewidths=0,
            label=label,
            zorder=2,
        )
        # pm, ps = rolling_mean_std(p)
        # ciline(offset + pm[750:1250], 10 * ps[750:1250], ax=axes["x"], plot_kwargs=dict(lw=1, label=label), color=col)
    axes["x"].legend(
        loc="upper left",
        framealpha=1,
        fancybox=False,
        title="min corr.\\ $\\theta$",
    )
    axes["x"].set_ylabel("depth (\\textmu{{}}m)", labelpad=-2 * rem)
    axes["x"].set_xlabel("time (s, 250Hz sampling)", labelpad=-rem)

    return fig, axes
