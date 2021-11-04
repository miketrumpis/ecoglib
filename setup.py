#!/usr/bin/env python
from setuptools import setup, Extension
import numpy


try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()


header_dirs = list(numpy_include_dirs)

if __name__=='__main__':
    cknn_graph = Extension(
        'ecoglib.graph.cknn_graph',
        ['ecoglib/graph/cknn_graph.pyx'],
        include_dirs=header_dirs,
        extra_compile_args=['-O3']
    )

    bispectrum = Extension(
        'ecoglib.estimation._bispectrum',
        ['ecoglib/estimation/_bispectrum.pyx'],
        include_dirs=header_dirs,
        extra_compile_args=['-O3']
    )

    semivariance = Extension(
        'ecoglib.estimation.spatial_variance._semivariance',
        ['ecoglib/estimation/spatial_variance/_semivariance.pyx'],
        include_dirs=header_dirs,
        extra_compile_args=['-O3']
    )
    setup(
        ext_modules = [cknn_graph,
                       bispectrum,
                       semivariance],
    )
