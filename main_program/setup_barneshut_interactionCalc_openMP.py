from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "interactionCalc_openMP",
        ["barneshut_interactionCalc_openMP.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

setup(name="interactionCalc_openMP",
      ext_modules=cythonize(ext_modules))

