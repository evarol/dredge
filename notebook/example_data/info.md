### dartsort_dataset1_p2_tstart750_tend1500.npz

This is a .npz file (NumPy compressed archive) containing spike times (in
seconds), neural-net denoised amplitudes, and localization features (depths
along the probe) extracted by the DARTsort pipeline from the p2 recording
in dataset1 of the Imposed motion datasets of Steinmetz et al., Science
2021, with raw data shared here:
https://figshare.com/articles/dataset/_Imposed_motion_datasets_from_Steinmetz_et_al_Science_2021/14024495.
To shrink the file, only spikes between t=750-1500s have been retained,
which roughly aligns with the period during which zig-zag motion was
deliberately applied for drift algorithm testing purposes.

To run the pipeline which extracts these spikes, visit notebook/dartsort_dredge_demo.ipynb
in the DARTsort repository: https://github.com/cwindolf/dartsort. That repository
includes further instructions for installation, et cetera.
