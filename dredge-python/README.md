# Python library for DREDge

## Introduction

This library contains Python code for motion estimation and correction for high density LFP and spiking data. Demo notebooks are available to see how to use this code:
 - [Spike registration](notebook/ap_registration.ipynb)
 - [LFP-based registration and interpolation](notebook/lfp_registration_and_interpolation_demo.ipynb)
 - [Experimental demo of chronic multi-session registration](notebook/ap_registration.ipynb)

:point_right: For spike-based workflows (not LFP), SpikeInterface implements DREDge under the name "decentralized", and this is very easy to combine with motion-correction interpolation for input into spike sorters like Kilosort. We suggest to combine this with the "monopolar_triangulation" localization method (Boussard et al., Neurips). The "nonrigid_accurate" preset for their `correct_motion()` function combines this localization method with DREDge and an interpolator. See https://spikeinterface.readthedocs.io/en/latest/modules/motion_correction.html for info!

## Installation with Conda (or Mamba)

### Setting up your conda environment

If you are new to conda, I recommend using [mamba][mamba]'s Mambaforge distribution. Install instructions are at that link -- look for an installation script under the heading Mambaforge.

Conda/mamba lets us have multiple Python environments on our system so that package versions don't conflict across your different projects. Here's how to set up an example environment that will support everything DREDge needs.

```
# make the environment (feel free to use another name)
# and, of course, feel free to add other packages that you like to use here!
# note, I'm adding some of the spikeinterface dependencies here so that mamba manages them
$ mamba create -n dredge python=3.10 numpy scipy h5py tqdm jupyterlab seaborn numba scikit-learn

# activate this environment!
$ mamba activate dredge
# now, you should see (dredge) in your terminal prompt

# install pytorch: go to pytorch.org and follow the installation instructions there
# you can use the Conda option on that site when install with mamba
# [!!!] if you are using mamba, be sure to replace `conda` with `mamba` in the pytorch.org instructions!
```

### Installing DREDge, probeinterface, and spikeinterface

**(This part is only needed for LFP users.)** Since this repo and its relationship with `spikeinterface` are currently in active development, we'll need to install `spikeinterface` and `probeinterface` from their github versions, and we'll even need to use specific branches in some cases. The current install steps are:

```
# make sure you are in a folder where you want to store a bunch of different code
# get probeinterface installed 
(dredge) $ git clone git@github.com:SpikeInterface/probeinterface.git
(dredge) $ cd probeinterface
(dredge) $ pip install -e .
# leave this repo's folder
(dredge) $ cd ..

# install spikeinterface
(dredge) $ git clone git@github.com:SpikeInterface/spikeinterface.git
(dredge) $ cd spikeinterface
# now, we can pip install. do a -e (editable) install so that we can work off of
# different branches later without needing to reinstall
(dredge) $ pip install -e .
# leave the folder
(dredge) $ cd ..
```

**(This part is for all users.)** Finally, we are ready to install `dredge`!

```
(reglib) $ git clone git@github.com:evarol/dredge.git
(reglib) $ cd dredge/dredge-python
(reglib) $ pip install -e .
```

## Trying out the demo

After installing as above, you can open Jupyter Lab and checkout the demo notebook:

```
# make sure we're in the right folder:
(reglib) $ pwd
# => .../dredge/dredge-python
(reglib) $ jupyter lab
```

This should open your browser. Use the file explorer on the left to navigate to `notebook/demo.ipynb`.


## Related information

We're also porting the AP version of this code into [spikeinterface][spikeinterface]. Check out the demo: https://github.com/catalystneuro/spike-sorting-hackathon/blob/main/projects/motion-correction-in-si/motion_estimation_and_correction_demo.ipynb. (Older version on SI reports, which has outputs saved: https://spikeinterface.github.io/blog/spikeinterface-motion-estimation/)


## Code structure

 - `dredgelib.py`
   - Library code (cross-correlations and linear algebra to solve batch and online centralization problems)
 - `motion_util.py`
   - `MotionEstimate` objects. The function `get_motion_estimate` takes the matrix of displacement vectors and the corresponding time and spatial bin edges or centers, and returns a `RigidMotionEstimate` or `NonrigidMotionEstimate`. These classes implement `disp_at_s()` and `correct_s` which take in times and depths and return the corresponding estimated displacements or motion-corrected positions
   - `spike_raster`, which takes spike times, depths, and amplitudes, and returns a 2d raster map
   - Binning and window helpers
   - Plotting helpers (e.g., `show_spike_raster`, `plot_me_traces`, ...)
 - `dredge_ap`: `register()` function registers spike data
 - `dredge_lfp`: `register_online_lfp()` function implements online algorithm for LFP data


[mamba]: https://github.com/conda-forge/miniforge#mambaforge
[spikeinterface]: https://github.com/SpikeInterface/spikeinterface
