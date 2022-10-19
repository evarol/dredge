import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.io import loadmat
from collections import namedtuple

from . import ibme

DSMeta = namedtuple("DSMeta", ["name", "has_csd", "t_start", "t_end", "ins_depth"])

def load_chanmap(chanmap_mat):
    try:
        h = loadmat(chanmap_mat)
    except:
        with h5py.File(chanmap_mat) as h5:
            h = {k: h5[k][...] for k in h5}
    return np.c_[h["xcoords"].squeeze(), h["ycoords"].squeeze()]

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
    im = aa.imshow(D, vmin=-dmax, vmax=dmax, cmap=plt.cm.bwr)
    plt.colorbar(im, ax=aa, shrink=0.3, label="displacement (um)")
    
    im = ab.imshow(C, vmin=0, cmap=plt.cm.magma)
    plt.colorbar(im, ax=ab, shrink=0.3, label="correlation")
    
    return fig, axes

def entropy_1d(y, bin_size_um=1):
    bins = np.arange(np.floor(y.min()), np.ceil(y.max()) + bin_size_um, bin_size_um)
    hist, *_ = np.histogram(y, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return -np.sum(p * np.log(p))

def total_corr(y, t, a):
    r, dd, tt = ibme.fast_raster(a, y, t)
    corr = np.corrcoef(r.T)
    assert corr.shape == (*tt.shape, *tt.shape)
    return np.abs(corr).sum() / corr.size

def total_std(y, t, a):
    r, dd, tt = ibme.fast_raster(a, y, t)
    return r.std(axis=1).mean()

def showmetrics(t, a, ys, names):
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
