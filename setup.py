from setuptools import setup, Extension

setup (
    # Base info
    name = "anthony_project_tools",
    version = "0.2.0",
    author = "François GINDRAUD",
    author_email = "francois.gindraud@gmail.com",

    # Code content
    ext_modules = [
        Extension (
            "eigen_wrapper",
            sources = ["src/wrapper.cpp"],
            include_dirs = ["include/"],
        )
    ],

    # Metadata
    description = "Tools for Anthony's project",
    long_description = (
        "Wrapped Eigen3's eigsy function."
        "Spherical Kmeans with kmeans++ init."
    ),
    url = "https://github.com/lereldarion/anthony_project_tools",
    license = "MIT",
)
