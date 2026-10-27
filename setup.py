"""Build script for the ``c_utils`` Cython extension.

Metadata lives in ``pyproject.toml``; only the extension module is declared here,
which is the usual arrangement for Cython projects. The extension is required by
:mod:`cw_hgt.walk.causal` (backdoor adjustment of the transition probabilities).

    python setup.py build_ext --inplace

A prebuilt Windows/CPython-3.11 binary is shipped in the repository so that the
common case needs no compiler.
"""

from setuptools import Extension, setup

import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - fall back to the generated C file
    cythonize = None

extensions = [Extension("c_utils", ["c_utils.pyx"], include_dirs=[np.get_include()])]

if cythonize is not None:
    extensions = cythonize(extensions, language_level="3")

setup(ext_modules=extensions)
