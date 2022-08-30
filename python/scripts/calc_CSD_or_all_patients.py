# %%
import numpy as np
from scipy.io import loadmat, savemat
import pixelCSD
import h5py
# %% pt. 01:
patients = ['pt01', 'pt02', 'pt03', 'pt04', 'pt05', 'pt06']
patients = ['pt06']

for pat in patients:
    print(pat)
    data_home = f"D:/Neuropixel/To_Columbia/{pat}"
    geom = loadmat(f"{data_home}/{pat}_chanMap.mat")
    try:
        ch192 = list(geom["Channel_name"]).index('CH191')
    except:
        ch192 = 191
    geom = np.stack([np.delete(geom["xcoords"], ch192),
                     np.delete(geom["ycoords"], ch192)], axis=1)
    try:
        raw_data = loadmat(f"{data_home}/{pat}_LFP.mat")
    except:
        f = h5py.File(f"{data_home}/{pat}_LFP.mat", 'r')
        raw_data = {k: np.array(f.get(k)).T for k in f.keys()}

    data = np.delete(raw_data["LFPMatrix"], ch192, 0)

    CSD, CSD_y = pixelCSD.pixelCSD(data, geom)
    raw_data.pop("LFPMatrix")
    raw_data["CSD"] = CSD
    raw_data["CSD_depth"] = CSD_y

    savemat(f"{data_home}/{pat}_CSD.mat", raw_data)

# %%
