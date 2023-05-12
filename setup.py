from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup (
    # Code content ; Pybind11Extension handles finding the lib wherever it has been installed
    ext_modules = [
        Pybind11Extension (
            'apt.eigen_wrapper',
            sources = ['apt/eigen_wrapper.cpp'],
            include_dirs = ['include/'],
        )
    ],
    cmdclass = { 'build_ext': build_ext } # Auto use latest C++ https://pybind11.readthedocs.io/en/stable/compiling.html
)
