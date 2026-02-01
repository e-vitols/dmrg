/* 
 * Adapted from Python to C++ with help from ChatGPT.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static py::array_t<double> parity_local(int local_dim = 4) {
    if (local_dim != 4) {
        throw py::value_error("Only implemented for local dimension 4");
    }

    py::array_t<double> arr({4, 4});
    auto a = arr.mutable_unchecked<2>();

    // zero fill
    for (py::ssize_t i = 0; i < 4; ++i)
        for (py::ssize_t j = 0; j < 4; ++j)
            a(i, j) = 0.0;

    a(0, 0) =  1.0;
    a(1, 1) = -1.0;
    a(2, 2) = -1.0;
    a(3, 3) =  1.0;

    return arr;
}

PYBIND11_MODULE(_parity, m) {
    m.doc() = "C++ parity_local(local_dim=4) via pybind11";
    m.def("parity_local", &parity_local, py::arg("local_dim") = 4,
          "Return diag(1,-1,-1,1) if local_dim==4, else raise ValueError.");
}

