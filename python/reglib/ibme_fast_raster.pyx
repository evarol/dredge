#!python
#cython: language_level=3
import numpy as np
# cimport cython


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
def raster(
    const double[::1] times,
    double[::1] depths,
    double[::1] amps,
    double[:, :] M,
    double[:, :] N,
):
    cdef int j
    cdef Py_ssize_t S = times.shape[0]

    for j in range(S):
        idj = <int> depths[j]
        itj = <int> times[j]
        M[idj, itj] += amps[j]
        N[idj, itj] += 1
