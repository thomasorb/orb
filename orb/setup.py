from distutils.core import setup
from Cython.Build import cythonize

setup(name="Cutils for ORB",
      ext_modules=cythonize('cutils.pyx'))
