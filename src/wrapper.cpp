#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

Eigen::VectorXd eig_sy_eigen(const Eigen::Ref<const Eigen::MatrixXd> & input_sym) {
    return input_sym.selfadjointView<Eigen::Lower>().eigenvalues();
}

PYBIND11_MODULE(python_eigen_wrapper, m) {
    m.doc() = "Wrapper for eigenvalue computation functions";

    m.def(
        "eig_sy_eigen",
        &eig_sy_eigen,
        "Computes eigenvalues using the default eigenvalue solver of Eigen C++ lib\n"
        "Uses a QR iterative algorithm in O(n^3), according to Eigen's doc\n"
        "\n"
        "input_sym: symmetric matrix of double (numpy.float64)\n"
        "output: vector of eigenvalues",
        py::arg("input_sym"));
}