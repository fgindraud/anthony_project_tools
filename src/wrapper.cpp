#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(python_eigen_wrapper, m) {
    m.doc() = "Wrapper for eigenvalue computation functions";
}