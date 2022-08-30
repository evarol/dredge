# Python library for LFP registration of high-density ephys data

## Introduction

This library extends the registration code in our [WIP spike sorting library][wipsorter] to work with LFP data. A [demo notebook](notebook/demo.ipynb) is available to see how to use this code.

This library only depends on the following files from the spike sorting repository: `ap_filter.py` (a preprocessing module which relies on Olivier Winter's [`ibl-neuropixel` package](https://pypi.org/project/ibl-neuropixel/)), and the image based motion estimation libraries `ibme.py` and `ibme_corr.py`. 

## Installation with Conda

If you are new to conda, I recommend using [mamba][mamba]'s Mambaforge distribution. We'll need to install some dependencies and also the package from the [sorter repo][wipsorter]. Here are demo installation steps, although the specific versions are not so important (the most important is pytorch and any recent-ish version will work):

```
# make the environment (feel free to use another name)
$ conda create -n reglib -c conda-forge python=3.9 numpy seaborn scikit-image scikit-learn scipy h5py tqdm jupyterlab pyfftw cython
$ conda activate reglib
# install pytorch: if you have a GPU, go to pytorch.org and follow the instructions there. for CPU,
(reglib) $ conda install pytorch torchvision torchaudio -c pytorch
# install pip dependencies
(reglib) $ pip install -r requirements.txt
```

Now, install the spike sorting repo:

```
(reglib) $ git clone git@github.com:cwindolf/spike-psvae
(reglib) $ cd spike-psvae
(reglib) $ pip install -e .
```

Finally, install this repo:

```
# back to original directory
(reglib) $ cd ..
(reglib) $ git@github.com:evarol/neuropixelsLFPregistration.git
(reglib) $ cd neuropixelsLFPregistration/python
(reglib) $ pip install -e .
```

## Trying out the demo

After installing the above, you can run Jupyter Lab and checkout the demo notebook:

```
# make sure we're in the right folder:
(reglib) $ pwd
# => .../neuropixelsLFPregistration/python
(reglib) $ jupyter lab
```

This should open your browser. Use the file explorer on the left to navigate to `notebook/demo.ipynb`.


## Related information

We're also porting the AP version of this code into [spikeinterface][spikeinterface]. You won't need to install our WIP sorter to use that one. Check out the demo: https://github.com/catalystneuro/spike-sorting-hackathon/blob/main/projects/motion-correction-in-si/motion_estimation_and_correction_demo.ipynb. (Older version on SI reports, which has outputs saved: https://spikeinterface.github.io/blog/spikeinterface-motion-estimation/)


[wipsorter]: (https://github.com/cwindolf/spike-psvae)
[mamba]: https://github.com/conda-forge/miniforge#mambaforge
[spikeinterface]: https://github.com/SpikeInterface/spikeinterface