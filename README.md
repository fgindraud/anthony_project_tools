Wraps C++ eigenvalue computations functions in Python.

# Install #

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
import eigen_wrapper

# Print module info, including list of functions
help(eigen_wrapper)

# Usage
import numpy
m = numpy.eye(10) # Id_10
(values, vectors) = eigen_wrapper.eigsy(m)
```

# Dev #

Header only dependencies used, copied into this repository for ease of use:
 * pybind11, v2.5.0
 * Eigen, v3.3.7