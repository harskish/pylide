#!/usr/bin/env python

from distutils.core import setup
from pathlib import Path

import sys
if 'develop' in sys.argv:
    print("WARNING: 'python setup.py develop' won't install \
        dependencies correctly, please use 'pip install -e .' instead.")

setup(name='pylide',
    version='0.0.1',
    description='Custom PyTorch operations with Halide',
    author='Erik Härkönen',
    author_email='erik.harkonen@hotmail.com',
    url='https://github.com/harskish/pylide',
    # Single-file modules to install (excluding .py suffix)
    # Will become importable: `import <name>`
    py_modules=[],
    # Proper packages with __init__.py
    # Not recursive, must list subpackages explicitly
    packages=[
        'pylide'
    ],
)