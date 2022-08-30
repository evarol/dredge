import numpy as np


def pixelCSD(data, geom):
    """Takes neuropixels data and geometry and computes CSD for each column
       returns the average per-depth

    Args:
        data ([type]): [description]
        geom ([type]): [description]

    Returns:
        CSD [array]: [description]
    """
    X_values = geom[:, 0]
    Y_values = geom[:, 1]
    X_unique = np.unique(X_values)

    # init with NaNs:
    CSD = np.full((len(np.unique(Y_values))-2, data.shape[1],
                   len(X_unique)), np.nan, dtype='float32')
    CSD_y = np.unique(Y_values)[1:-1]

    for x in range(len(X_unique)):
        subset = data[X_values == X_unique[x], :]
        y = Y_values[X_values == X_unique[x]]
        y_full = np.tile(np.diff(y), (subset.shape[1], 1)).T

        temp_csd = computeCSD(subset, y_full)

        depth_idx = np.searchsorted(CSD_y, y)
        depth_idx
        CSD[depth_idx[1:-1], :, x] = temp_csd

    mean_CSD = np.nanmean(CSD, 2)
    # remove rows that are all NaNs:
    idx = ~np.isnan(mean_CSD).all(axis=1)
    return mean_CSD[idx], CSD_y[idx]


def computeCSD(data, y_diff):
    csd = (2 * data[1:-1] - data[2:] - data[:-2]) / y_diff[:-1]
    return csd.astype('float32')
