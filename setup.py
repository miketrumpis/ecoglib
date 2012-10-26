from distutils.core import setup, Extension
from numpy.distutils.command import build_src
import Cython
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = True
from Cython.Distutils import build_ext
import Cython
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

if __name__=='__main__':
    setup(
        name = 'ecoglib',
        version = '1.0',
        packages = ['vis', 'graph'],
        ext_modules = [ cknn_graph ],
        cmdclass = {'build_ext': build_ext}
    )