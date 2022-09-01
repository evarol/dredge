#!/usr/bin/env python
from setuptools import setup

try:
    from Cython.Build import cythonize

    ext_modules = cythonize(["reglib/ibme_fast_raster.pyx"])
except ImportError:
    ext_modules = None

setup(
    name="reglib",
    version="0.1",
    packages=["reglib"],
    ext_modules=ext_modules,
)
