from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup (
    # Code content ; Pybind11Extension handles finding the lib wherever it has been installed
    ext_modules = [
        Pybind11Extension (
            "apt.eigen_wrapper",
            sources = ["apt/eigen_wrapper.cpp"],
            include_dirs = ["include/"],
        )
    ],
    packages = ["apt"],
)
