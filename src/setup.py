from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    ext_modules = cythonize("cython_core.pyx", annotate=False),
    include_dirs=[numpy.get_include()]
)
