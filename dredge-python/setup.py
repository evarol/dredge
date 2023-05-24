#!/usr/bin/env python
from setuptools import setup

try:
    from Cython.Build import cythonize

    ext_modules = cythonize(["dredge/ibme_fast_raster.pyx"])
except ImportError:
    ext_modules = None

setup(
    name="dredge",
    version="0.1",
    packages=["dredge"],
    ext_modules=ext_modules,
)
