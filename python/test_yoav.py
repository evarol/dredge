# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pixelCSD

import imp

# %%
data_home = "D:/Neuropixel/To_Columbia/pt02"
geom = loadmat(f"{data_home}/pt02_chanMap.mat")
geom = np.stack([geom["xcoords"], geom["ycoords"]], axis=1)[:, :, 0]

raw_data = loadmat(f"{data_home}/pt02_LFP.mat")
# data = raw_data["LFPMatrix"][:, int(5e5):int(8e5)]
data = raw_data["LFPMatrix"]

# %%
imp.reload(pixelCSD)

CSD, CSD_y = pixelCSD.pixelCSD(data, geom)
raw_data.pop("LFPMatrix")
raw_data["CSD"] = CSD
raw_data["CSD_depth"] = CSD_y

savemat(f"{data_home}/pt02_CSD.mat", raw_data)


# %%
