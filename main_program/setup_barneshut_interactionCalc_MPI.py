from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

ext_modules = [
    Extension(
        "interactionCalc_MPI",
        ["barneshut_interactionCalc_MPI.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(name="interactionCalc_MPI",
      ext_modules=cythonize(ext_modules))

