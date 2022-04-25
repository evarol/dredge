# %%
from tqdm.auto import trange, tqdm
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm, zscore
from scipy import sparse
from scipy.signal import decimate
from scipy.interpolate import interp1d
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
import gc
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]
                .joinpath('lib', 'neuropixelsLFPregistration', 'python')))
from pixelCSD import pixelCSD
import lfpreg
from datetime import datetime
#%%

def load_data(params):
    
    #TODO:Get_global_timing(params, data)
    global_timestamps = loadmat(params["timestamps_file"])
    AP_timestamp = global_timestamps["AP_timestamp"][0]
    LFP_timestamp = global_timestamps["LFP_timestamp"][0]
    DAQ_timestamp = global_timestamps["DAQ_timestamp"][0]

    AP_samples = np.where((AP_timestamp >= params["time_frame"][0])
                          & (AP_timestamp <= params["time_frame"][1]))[0]
    LFP_samples = np.where((LFP_timestamp >= params["time_frame"][0])
                           & (LFP_timestamp <= params["time_frame"][1]))[0]
    DAQ_samples = np.where((DAQ_timestamp >= params["time_frame"][0])
                           & (DAQ_timestamp <= params["time_frame"][1]))[0]

    LFP_raw = np.memmap(
        params["lfp_file"], dtype=np.int16).reshape(-1, params["num_of_raw_channels"])
    if params["num_of_raw_channels"] == 385:
        LFP_TTL = LFP_raw[LFP_samples, -1].T
    LFP_raw = LFP_raw[LFP_samples, :384].astype(np.float32).T

    AP_raw = np.memmap(
        params["ap_file"], dtype=np.int16).reshape(-1, params["num_of_raw_channels"])
    if params["num_of_raw_channels"] == 385:
        AP_TTL = AP_raw[AP_samples, -1].T
    AP_raw = AP_raw[AP_samples, :384].T

    data = {"AP_raw": AP_raw,
            "LFP_raw": LFP_raw,
            "AP_timestamp": AP_timestamp[AP_samples],
            "LFP_timestamp": LFP_timestamp[LFP_samples],
            "DAQ_timestamp": DAQ_timestamp[DAQ_samples],
            }
    try:
        data["AP_TTL"] = AP_TTL
        data["LFP_TTL"] = LFP_TTL
    except:
        pass
    
    if 'channel_map_file' in params.keys():
        geom = loadmat(params["channel_map_file"])
        geom = np.stack([geom["xcoords"], geom["ycoords"]], axis=1)[:, :, 0]
        data["geom"] = geom
    return data

def lfp_registration_wrapper(params, data, DOWN_SAMPLING_FACTOR = 25, plot = True):
    rcsd, yloc1 = lfpreg.lfpraster(data["LFP_raw"], data["geom"],  csd=True) 
    rcsd = decimate(rcsd.astype('float64'), DOWN_SAMPLING_FACTOR, axis=1).astype('float32')

    #main step of the algorithm:
    p_csd = lfpreg.register_rigid(rcsd)


    # interpolate back into LFP and AP sampling:
    x_LFP = np.arange(0, len(p_csd), 1/DOWN_SAMPLING_FACTOR)
    p_csd_lfpFit = interp1d(np.arange(0, len(p_csd)), p_csd, kind=1)
    p_csd_lfp_Fs = p_csd_lfpFit(x_LFP[:-DOWN_SAMPLING_FACTOR])
    
    
    LFP_timestamp = data["LFP_timestamp"][0:len(p_csd_lfp_Fs)]
    AP_timestamp = data["AP_timestamp"]
    AP_timestamp = AP_timestamp[(AP_timestamp>LFP_timestamp[0]) & (AP_timestamp<LFP_timestamp[-1])]

    p_csd_AP_fit = interp1d(LFP_timestamp, p_csd_lfp_Fs, kind=1)
    p_csd_AP_Fs = p_csd_AP_fit(AP_timestamp)
    

    # mdic = {'p_csd': p_csd, "p_csd_lfp_Fs": p_csd_lfp_Fs, "p_csd_AP_Fs": p_csd_AP_Fs,
    #         "AP_timestamp": AP_timestamp, "LFP_timestamp": LFP_timestamp}
    # savemat(f"{params["data_home"]}/Processed/p_csd_all.mat", mdic)

    mdic = {'p_csd': p_csd, "p_csd_lfp_Fs": p_csd_lfp_Fs, "p_csd_AP_Fs": p_csd_AP_Fs,
            "AP_timestamp": AP_timestamp, "LFP_timestamp": LFP_timestamp,
            "rcsd": rcsd, "yloc1": yloc1, "DOWN_SAMPLING_FACTOR": DOWN_SAMPLING_FACTOR}
    params["p_file"] = "{data_home}/Processed/p_csd_all.mat".format(**params)
    params["script_running_time"] = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    mdic.update(params)
    savemat(params["p_file"] , mdic)
    
    if plot is True:
        fig, (aa) = plt.subplots(1, 1, figsize=(5, 6))
        aa.imshow(rcsd, aspect="auto")
        aa.scatter(np.arange(len(p_csd_lfp_Fs[::DOWN_SAMPLING_FACTOR])), 95 + p_csd_lfp_Fs[::DOWN_SAMPLING_FACTOR], c="w", s=0.5)
        aa.set_title("csd registration")
    
    
    
    return mdic, params

def Get_global_timing(params, data):
    # TODO: compute global timestamps for open ephys if required:
    #       D:\git_toolbox\open-ephys-python-tools\open_ephys\analysis\recording.py
    #       compute_global_timestamps(self)
    # TODO: for spikeGLX use to generate global_timestamps file:
    #       lib\global_timing\Get_Global_timestamps_All_patients.m
    #       lib\global_timing\align_TTL_timing.m 
    
    
    pass

# -- library functions
def register_rigid_AP(
    raster,
    mincorr=0.7,
    disp=None,
    batch_size=32,
    step_size=1,
):
    pass

def register_rigid_LFP(
    raster,
    mincorr=0.7,
    disp=None,
    batch_size=32,
    step_size=1,
):
    pass

def normalize_AP(AP_file):
    pass

    raw_a = np.memmap(AP_file, dtype=np.int16).reshape(-1, NUM_OF_CH_IN_RAW)
