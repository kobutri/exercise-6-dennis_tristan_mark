#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "mpi_comm_wrapper.h"

PYBIND11_MODULE(pmsc, mod)
{
    mod.doc() = "PMSC python module"; 

    namespace py = pybind11;

    py::class_<MpiCommWrapper>(mod, "MpiComm")
        .def("rank", &MpiCommWrapper::rank)
        .def("size", &MpiCommWrapper::size)
        ;

    mod.def("mpi_comm_world", [](){ return MpiCommWrapper(MPI_COMM_WORLD); });
    mod.def("mpi_comm_self", [](){ return MpiCommWrapper(MPI_COMM_SELF); });

    // TODO: add other exports here
}
