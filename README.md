Wraps C++ eigenvalue computations functions in Python.

# Install #

Requires the system library `gmp`, with headers.
On Ubuntu/Debian, install the `libgmp-dev` package using `apt-get`.

If using `pip`, standalone or with `anaconda`, the following will install into the current environment:
```bash
# At the git root
pip install .
```

If not using any framework, an install in a _user local_ directory (`$HOME/.local/lib/` on my machine) can be done by running:
```bash
# At the git root
python setup.py install --user
```

# Usage #

After an **install** in the current environment, the module can be loaded in any python interpreter or script:
```python
import python_eigen_wrapper

# Print module info, including list of functions
help(python_eigen_wrapper)

# Usage
import numpy
m = numpy.eye(10) # Id_10
(values, vectors) = python_eigen_wrapper.eig_sy_eigen(m)
(values, vectors) = python_eigen_wrapper.eig_sy_eigen_internal_gmp(m)
```

# Dev #

Header only dependencies used, copied into this repository for ease of use:
 * pybind11, v2.5.0
 * Eigen, v3.3.7