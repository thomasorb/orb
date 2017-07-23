from __future__ import print_function

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
import io
import codecs
import os
import sys

import orb
import orb.version
import numpy

packages = find_packages(where=".")

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = ''#read('README.rst')

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
    name='orb',
    ext_modules=cythonize(extensions),
    version=orb.version.__version__,
    url='https://github.com/thomasorb/orb',
    license='GPLv3+',
    author='Thomas Martin',
    author_email='thomas.martin.1@ulaval.ca',
    maintainer='Thomas Martin',
    maintainer_email='thomas.martin.1@ulaval.ca',
    install_requires=requirements,
    description='Kernel module for the reduction and analysis of SITELLE data',
    long_description=long_description,
    packages=packages,
    package_dir={"": "."},
    #include_package_data=True,
    package_data={
        '':['COPYING', '*.rst', '*.txt'],
        'orb':['data/*'],
        'docs':['*']},
    platforms='any',
    scripts=[
        'scripts/orb-dstack',
        'scripts/orb-viewer',
        'scripts/orb-bin-cube',
        'scripts/orb-subtractv',
        'scripts/orb-combine',
        'scripts/orb-header',
        'scripts/orb-viewer3d',
        'scripts/orb-extract',
        'scripts/orb-convert',
        'scripts/orb-unstack',
        'scripts/orb-reduce'],
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent' ],
)
