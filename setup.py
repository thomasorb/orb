from distutils.core import setup
from Cython.Build import cythonize
from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

import orb

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

long_description = read('README.txt', 'CHANGES.txt')
long_description = 'Long description'

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='orb',
    ext_modules=cythonize('orb/cgvar.pyx', 'orb/cutils.pyx'),
    version=orb.__version__,
    url='https://github.com/thomasorb/orb',
    license='GPL-3.0',
    author='Thomas Martin',
    tests_require=['pytest'],
    install_requires=requirements,
    cmdclass={'test': PyTest},
    author_email='thomas.martin.1@ulaval.ca',
    description='ORB',
    long_description=long_description,
    packages=['orb'],
    include_package_data=True,
    platforms='any',
    #test_suite='sandman.test.test_sandman',
    classifiers = [
       'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: GPL-3.0                ',
        'Operating System :: OS Independent' ],
    extras_require={
        'testing': ['pytest'],
    }
)
