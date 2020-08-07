from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import io
import codecs
import os
import sys

import orb
import numpy

packages = find_packages(where=".")

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
extensions = [
 Extension(
     "orb.cgvar",
     [
         "orb/cgvar.pyx"
     ],
     include_dirs=[numpy.get_include()]
 ),
 Extension(
     "orb.cutils",
     [
         "orb/cutils.pyx"
     ],
     include_dirs=[numpy.get_include()]
 )

]

setup(
    name='orb-kernel',
    ext_modules=cythonize(extensions),
    version='3.0',
    url='https://github.com/thomasorb/orb',
    license='GPLv3+',
    author='Thomas Martin',
    author_email='thomas.martin.1@ulaval.ca',
    maintainer='Thomas Martin',
    maintainer_email='thomas.martin.1@ulaval.ca',
    setup_requires=['cython', 'numpy', 'gvar'],
    description='Kernel module for the reduction and analysis of SITELLE data',
    long_description=long_description,
    packages=packages,
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        '':['LICENSE.txt', '*.rst', '*.txt', 'docs/*', '*.pyx', 'README.md'],
        'orb':['data/*', '*.pyx']},
    exclude_package_data={
        '': ['*~', '*.so', '*.pyc', '*.c', 'orb/cgvar.c'],
        'orb':['*~', '*.so', '*.pyc', '*.c']},
    platforms='any',
    scripts=[
        'scripts/orb-header',
        'scripts/orb-extract',
        'scripts/orb-convert'],
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent' ],
)
