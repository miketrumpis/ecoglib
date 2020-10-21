import os
from setuptools import setup, Extension, find_packages
from numpy.distutils.command import build_src
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = True
from Cython.Distutils import build_ext
import numpy

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()


dirs = list(numpy_include_dirs)

cknn_graph = Extension(
    'ecoglib.graph.cknn_graph',
    ['ecoglib/graph/cknn_graph.pyx'], 
    include_dirs = dirs,
    extra_compile_args=['-O3']
    )

bispectrum = Extension(
    'ecoglib.estimation._bispectrum',
    ['ecoglib/estimation/_bispectrum.pyx'], 
    include_dirs = dirs, 
    extra_compile_args=['-O3']
    )

semivariance = Extension(
    'ecoglib.estimation.spatial_variance._semivariance',
    ['ecoglib/estimation/spatial_variance/_semivariance.pyx'],
    include_dirs = dirs, 
    extra_compile_args=['-O3']
    )

with open('requirements.txt') as f:
    reqs = f.readlines()
    reqs = list(map(str.strip, reqs))
    def _del_soft_reqs(s):
        if 'mayavi' in s.lower():
            return False
        if 'pyside' in s.lower():
            return False
        return True
    reqs = list(filter(_del_soft_reqs, reqs))

if __name__=='__main__':
    setup(
        name = 'ecoglib',
        version = '0.1',
        packages = find_packages(exclude=['demos', 'notes_n_demos', 'build']),
        ext_modules = [cknn_graph,
                       bispectrum,
                       semivariance],
        cmdclass = {'build_ext': build_ext},
        install_requires=reqs
    )
