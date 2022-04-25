# %%
# %load_ext autoreload
# %autoreload 2

from src import APreg
from lib.neuropixelsLFPregistration.python import lfpreg
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
from enum import auto
from scipy.stats import norm, zscore


# %matplotlib inline
%matplotlib


# %%
# Loading data:
DATA_HOME = "D:/Neuropixel/Verified_data/pt02"
LFP_FILE = f"{DATA_HOME}/Raw/file2_g0_t0.imec0.lf.bin"
AP_FILE = f"{DATA_HOME}/Raw/file2_g0_t0.imec0.ap.bin"
GLOBAL_TIMESTAMP_FILE = f"{DATA_HOME}/Processed/Global_timestamp.mat"
CHANNEL_MAP_FILE = f"{DATA_HOME}/Raw/pt02_chanMap.mat"


NUM_OF_CH_IN_RAW = 385
TIME_FRAME = [220, 873]
TIME_FRAME = [220, 865]


params = {"data_home": DATA_HOME,
          "lfp_file": LFP_FILE,
          "ap_file": AP_FILE,
          "timestamps_file": GLOBAL_TIMESTAMP_FILE,
          "time_frame": TIME_FRAME,
          "num_of_raw_channels": NUM_OF_CH_IN_RAW,
          "channel_map_file": CHANNEL_MAP_FILE
          }

data = APreg.load_data(params)

# %% LFP registration:
"""
After first run data is saved to p_csd_all
"""
try:
    params = loadmat(f"{DATA_HOME}/Processed/p_csd_all.mat")
except:
    mdic, params = APreg.lfp_registration_wrapper(params, data)




# %% split maps:
    """
    
for the next step use 
D:\Git_repos\python\Neuropixel_alignment\src\Intarpolate_and_align_AP_2022
until full python implementation
    
    
    """

lfp_data = zscore(data["LFP_raw"], axis=0)
geom = data["geom"]
p_csd_lfp_Fs = params["p_csd_lfp_Fs"][0]

x_val = np.unique(geom[:, 0])
y_val = np.unique(geom[:, 1])
if len(y_val) < 0.55*len(geom):  # parallel map:
    map_ind_1 = np.where((geom[:, 0] == x_val[0]) |
                         (geom[:, 0] == x_val[1]))[0]
    map_ind_2 = np.where((geom[:, 0] == x_val[2]) |
                         (geom[:, 0] == x_val[3]))[0]

    print("starting registration 1")
    lfp_map = lfp_data[map_ind_1, :len(p_csd_lfp_Fs)]
    lfp_map2 = lfp_map[:, :1000000]
    p_csd_lfp_Fs2 = p_csd_lfp_Fs[:1000000]
    lfp_map_r = lfpreg.rigid_registered_raster(lfp_map2, p_csd_lfp_Fs2)

    lfp_map_r = lfpreg.rigid_registered_raster(lfp_map, p_csd_lfp_Fs)

    mdic = {'lfp_map_1': lfp_map_r, "map_ind_1": map_ind_1}
    savemat(f"{DATA_HOME}/Processed/registered_lfp_1.mat")

    print("starting registration 2")
    lfp_map = lfp_data[map_ind_2, :len(p_csd_lfp_Fs)]
    lfp_map_r = lfpreg.rigid_registered_raster(lfp_map, p_csd_lfp_Fs)
    mdic = {'lfp_map_1': lfp_map_r, "map_ind_1": map_ind_1}
    savemat(f"{DATA_HOME}/Processed/registered_lfp_1.mat")


else:
    # TODO:long map
    pass


# rlfp, yloc1 = lfpreg.lfpraster(data["LFP_raw"], data["geom"])


# plt.figure()
# plt.imshow(lfp_map_1[:,::25], aspect="auto")
# plt.title("z-scored lfp")

# plt.plot(lfp_data[150,::25])
# # %%
# plt.plot(data["AP_TTL"])

# # AP_raw = AP_raw[:, :384].astype(np.float32).T

# # %%
# from scipy import signal
# x = signal.welch(AP_raw[265, 220*30000:840*30000], fs=30000, nperseg = 2**15)
# plt.plot(x[0], x[1])

# # %%
# plt.imshow(AP_raw[:, ::50], aspect='auto')
# plt.plot(AP_raw[35, ::10])

# # %%
# x = AP_raw[:, 1:10000] - np.median(AP_raw[:, 1:10000], axis=
