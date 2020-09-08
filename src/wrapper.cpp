#include <Eigen/Dense>
#include <gmpxx.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace py = pybind11;

// double + Eigen

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> eig_sy_eigen(const Eigen::Ref<const Eigen::MatrixXd> & input_sym) {
    auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(input_sym, Eigen::ComputeEigenvectors);
    return {solver.eigenvalues(), solver.eigenvectors()};
}

// gmp + Eigen

namespace Eigen {
template <> struct NumTraits<mpf_class> : GenericNumTraits<mpf_class> {
    using Real = mpf_class;
    using NonInteger = mpf_class;
    using Nested = mpf_class;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,

        ReadCost = HugeCost,
        AddCost = HugeCost,
        MulCost = HugeCost,
    };

    static inline Real epsilon() { return 0; }
    static inline Real dummy_precision() { return 0; }
    static int digits10() { return 0; }
};
} // namespace Eigen

using EigenMpMatrix = Eigen::Matrix<mpf_class, Eigen::Dynamic, Eigen::Dynamic>;
using EigenMpVector = Eigen::Matrix<mpf_class, Eigen::Dynamic, 1>;

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> eig_sy_eigen_internal_gmp(
    const Eigen::Ref<const Eigen::MatrixXd> & input_sym) {
    auto solver = Eigen::SelfAdjointEigenSolver<EigenMpMatrix>(input_sym.cast<mpf_class>(), Eigen::ComputeEigenvectors);
    auto to_double = [](const mpf_class & mp) -> double { return mp.get_d(); };
    return {solver.eigenvalues().unaryExpr(to_double), solver.eigenvectors().unaryExpr(to_double)};
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

    m.def(
        "eig_sy_eigen_internal_gmp",
        &eig_sy_eigen_internal_gmp,
        "Computes eigenvalues using the default eigenvalue solver of Eigen C++ lib\n"
        "Uses a QR iterative algorithm in O(n^3), according to Eigen's doc\n"
        "Takes and return doubles, but uses multiprecision internally\n"
        "\n"
        "input_sym: symmetric matrix of double (numpy.float64)\n"
        "output = (E,Q):\n"
        "   E = vector of eigenvalues\n"
        "   Q = matrix with eigenvectors as columns",
        py::arg("input_sym"));
}