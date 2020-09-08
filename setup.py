from setuptools import setup, Extension

setup (
    # Base info
    name = "python_eigen_wrapper",
    version = "0.1.0",
    author = "François GINDRAUD",
    author_email = "francois.gindraud@gmail.com",

    # Code content
    ext_modules = [
        Extension (
            "python_eigen_wrapper",
            sources = ["src/wrapper.cpp"],
            include_dirs = ["include/"],
            libraries = ["gmp"]
        )
    ],

    # Metadata
    description = "Wraps eigenvalue computation functions",
    url = "https://github.com/lereldarion/python_eigen_wrapper",
    license = "MIT",
)
