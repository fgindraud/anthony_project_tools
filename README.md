# Tools for Anthony's project #

- Wraps Eigen3's eigsy function (C++) in Python
- Spherical Kmeans with kmeans++ init

# Install #

This project uses the PEP517 `pyproject.toml` file, and should be compatible with the new python build standards.

It is recommended to use `pip` to install into the current environment:
```bash
# At the git root
pip install .
```

`pyproject.toml` is configured to install a version of pybind11 in the compilation environnement, instead of pre-including it like before `v0.3`.
During tests, this failed when an outdated system install of pybind11 was present in addition to the temporary one ; maybe remove it if not used.

If the installation shows `UNKNOWN` as package name, it seems to be a similar problem but with an outdated system setuptools (which does not retrieve the metadata in pyproject.toml).
The easiest solution seems to be to use a `virtualenv` to isolate from system libs, and use updated tools (`pip install --upgrade pip setuptools`).

# Usage #

After an **install** in the current environment, the modules can be loaded in any python interpreter or script:
```python
import apt.eigen_wrapper # For eigsy
import apt.kmeans # For kmeans

# Print module info, including list of functions
help(apt.eigen_wrapper)
help(apt.kmeans)

# Usage
import numpy
m = numpy.eye(10) # Id_10
(values, vectors) = apt.eigen_wrapper.eigsy(m)

# Kmeans
apt.kmeans.spherical_kmeans(points, nb_clusters, max_iter)
```

# Dev #

Dependencies :
 * pybind11 : automatically installed by pip if needed
 * Eigen, v3.3.7 : header only, local copy