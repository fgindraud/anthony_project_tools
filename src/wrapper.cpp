#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace py = pybind11;

// double + Eigen

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> eig_sy_eigen(const Eigen::Ref<const Eigen::MatrixXd> & input_sym) {
    auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(input_sym, Eigen::ComputeEigenvectors);
    return {solver.eigenvalues(), solver.eigenvectors()};
}

PYBIND11_MODULE(python_eigen_wrapper, m) {
    m.doc() = "Wrapper for eigenvalue computation functions";

    m.def(
        "eig_sy_eigen",
        &eig_sy_eigen,
        "Computes eigenvalues using the default eigenvalue solver of Eigen C++ lib\n"
        "Uses a QR iterative algorithm in O(n^3), according to Eigen's doc\n"
        "Manipulates double precision only\n"
        "\n"
        "input_sym: symmetric matrix of double (numpy.float64)\n"
        "output = (E,Q):\n"
        "   E = vector of eigenvalues\n"
        "   Q = matrix with eigenvectors as columns",
        py::arg("input_sym"));
}