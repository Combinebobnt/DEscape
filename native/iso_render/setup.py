"""Builds iso_kernel.pyx into a native extension for the Phase 0 benchmark.

Not wired into requirements.txt/CI -- see this directory's README.md. Run
from this directory with the project's own venv:

    ../../.venv/bin/python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# -O3 explicitly, since setuptools/sysconfig's default CFLAGS only give -O2 --
# a free, zero-effort win worth including before comparing against numpy
# (anything past this, e.g. -march=native or OpenMP prange, is real added
# complexity and out of Phase 0's "days, not weeks" scope).
ext = Extension(
    "iso_kernel",
    ["iso_kernel.pyx"],
    extra_compile_args=["-O3"],
)

setup(
    name="iso_render_bench",
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
    include_dirs=[numpy.get_include()],
)
