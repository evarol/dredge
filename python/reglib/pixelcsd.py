import numpy as np


def pixelcsd(lfp, geom):
    """Takes neuropixels lfp and geometry and computes CSD for each column
       returns the average per-depth

    Args:
        lfp : np.array, depth by time
        geom : n_channels by 2

    Returns:
        CSD [array]: [description]
    """
    if geom.shape[0] != lfp.shape[0]:
        raise ValueError(
            "May need to transpose `lfp`. It should be depth x time."
        )

    x_values = geom[:, 0]
    y_values = geom[:, 1]
    assert all(y_values[1:] >= y_values[:-1]), "Requires depth order."

    x_unique = np.unique(x_values)
    y_unique = np.unique(y_values)

    # init with NaNs:
    csd = np.full(
        (y_unique.size, lfp.shape[1], x_unique.size),
        np.nan,
        dtype=lfp.dtype,
    )

    for i, x in enumerate(x_unique):
        lfp_subset = lfp[x_values == x, :]
        y_subset = y_values[x_values == x]

        # csd as second spatial derivative
        csd_subset = np.gradient(lfp_subset, y_subset, axis=0)
        csd_subset = np.gradient(csd_subset, y_subset, axis=0)
        csd[np.isin(y_unique, y_subset), :, i] = csd_subset

    mean_csd = np.nanmean(csd, axis=2)
    # remove rows that are all NaNs:
    idx = ~np.isnan(mean_csd).all(axis=1)
    return mean_csd[idx], y_unique[idx]
