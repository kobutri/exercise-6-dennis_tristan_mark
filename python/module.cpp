#include "../discretization/poisson.h"
#include "../discretization/error.h"
#include "../grid/io.h"
#include "../solvers/cg.h"
#include "../solvers/gauss_seidel_iteration.h"
#include "../solvers/jacobi_iteration.h"
#include "../solvers/richardson.h"
#include <algorithm>
#include <initializer_list>
#include <mpi.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

#include "mpi_comm_wrapper.h"

class PySolver : public Solver<scalar_t>
{
public:
    using Solver<scalar_t>::Solver;
    using Solver<scalar_t>::last_stop_reason;

    void solve(Vector<scalar_t>& x, const Vector<scalar_t>& b) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            Solver<scalar_t>,
            solve,
            x, b);
    }

    void set_operator(const SparseMatrix<scalar_t>& A) override
    {
        PYBIND11_OVERLOAD(void, Solver<scalar_t>, set_operator, A);
    }

    void setup() override
    {
        PYBIND11_OVERLOAD(void, Solver<scalar_t>, setup);
    }
};

class PyIterativeSolver : public IterativeSolver<scalar_t>
{
public:
    using IterativeSolver<scalar_t>::IterativeSolver;
    using IterativeSolver<scalar_t>::set_preconditioner;
    using IterativeSolver<scalar_t>::max_iterations;
    using IterativeSolver<scalar_t>::absolute_tolerance;
    using IterativeSolver<scalar_t>::relative_tolerance;
    using IterativeSolver<scalar_t>::last_iterations;
    using IterativeSolver<scalar_t>::last_residual_norm;
    using IterativeSolver<scalar_t>::last_stop_reason;

    void set_operator(const SparseMatrix<scalar_t>& A) override
    {
        PYBIND11_OVERLOAD(void, IterativeSolver<scalar_t>, set_operator, A);
    }

    void setup() override
    {
        PYBIND11_OVERLOAD(void, IterativeSolver<scalar_t>, setup);
    }

    void solve(Vector<scalar_t>& x, const Vector<scalar_t>& b) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            Solver<scalar_t>,
            solve,
            x, b);
    }
};

class PyPreconditioner : public Preconditioner<scalar_t>
{
public:
    using Preconditioner<scalar_t>::Preconditioner;
    using Preconditioner<scalar_t>::set_operator;
    using Preconditioner<scalar_t>::setup;

    void apply(Vector<scalar_t>& x, const Vector<scalar_t>& b) override
    {
        PYBIND11_OVERLOAD_PURE(void, Preconditioner<scalar_t>, apply, x, b);
    }

    void set_operator(const SparseMatrix<scalar_t>& A) override
    {
        PYBIND11_OVERLOAD(void, Preconditioner<scalar_t>, set_operator, A);
    }
};

PYBIND11_MODULE(pmsc, mod)
{
    mod.doc() = "PMSC python module";

    namespace py = pybind11;

    py::class_<MpiCommWrapper>(mod, "MpiComm")
        .def("rank", &MpiCommWrapper::rank)
        .def("size", &MpiCommWrapper::size);

    mod.def("mpi_comm_world", []() { return MpiCommWrapper(MPI_COMM_WORLD); });
    mod.def("mpi_comm_self", []() { return MpiCommWrapper(MPI_COMM_SELF); });

    mod.def(
        "MPI_Init", []() { MPI_Init(nullptr, nullptr); }, "Initializes MPI");
    mod.def(
        "MPI_Finalize", []() { MPI_Finalize(); }, "Finalizies MPI");

    py::class_<Vector<scalar_t>>(mod, "Vector")
        .def(py::init<>())
        .def(py::init<const Vector<scalar_t>&>(), py::arg("other"))
        .def(py::init<int>(), py::arg("size"))
        .def(py::init<const std::vector<scalar_t>&>(), py::arg("values"))
        .def(py::init([](MpiCommWrapper comm, const std::vector<scalar_t>& values) {
                 return Vector<scalar_t>(comm, values);
             }),
             py::arg("communicator"), py::arg("values"))
        .def("__getitem__", static_cast<scalar_t& (Vector<scalar_t>::*)(int)>(&Vector<scalar_t>::operator[]),
             py::is_operator(), py::return_value_policy::reference_internal)
        .def(
            "__setitem__", [](Vector<scalar_t>& vec, int i, scalar_t value) { vec[i] = value; }, py::arg("i"),
            py::arg("value"))
        .def("size", &Vector<scalar_t>::size);

    py::class_<SparseMatrix<scalar_t>>(mod, "SparseMatrix")
        .def(py::init<>())
        .def(py::init<const SparseMatrix<scalar_t>&>(), py::arg("other"))
        .def(py::init<int, int, std::function<int(int)>>(), py::arg("rows"), py::arg("global_columns"),
             py::arg("nz_per_row"))
        .def(py::init<int, int, const std::vector<SparseMatrix<scalar_t>::triplet_type>&>(), py::arg("rows"),
             py::arg("columns"), py::arg("entries"))
        .def(py::init([](MpiCommWrapper comm, int local_rows, int global_columns, std::function<int(int)> nz_per_row) {
                 return SparseMatrix<scalar_t>(comm, local_rows, global_columns,
                                               std::move(nz_per_row));
             }),
             py::arg("communicator"),
             py::arg("local_rows"), py::arg("global_columns"), py::arg("nz_per_row"))
        .def(py::init([](MpiCommWrapper comm, int local_rows, int global_columns,
                         const std::vector<SparseMatrix<scalar_t>::triplet_type>& entries) {
                 return SparseMatrix<scalar_t>(comm, local_rows, global_columns,
                                               entries);
             }),
             py::arg("communicator"),
             py::arg("local_rows"), py::arg("global_columns"), py::arg("entries"))
        .def("rows", &SparseMatrix<scalar_t>::rows)
        .def("columns", &SparseMatrix<scalar_t>::columns)
        .def("row_nz_size", &SparseMatrix<scalar_t>::row_nz_size, py::arg("r"))
        .def("row_nz_index", py::overload_cast<int, int>(&SparseMatrix<scalar_t>::row_nz_index), py::arg("i"),
             py::arg("nz_i"))
        .def(
            "row_nz_size",
            [](SparseMatrix<scalar_t>& mat, int i, int nz_i, int value) { mat.row_nz_index(i, nz_i) = value; },
            py::arg("i"), py::arg("nz_i"), py::arg("value"))
        .def("row_nz_entry", py::overload_cast<int, int>(&SparseMatrix<scalar_t>::row_nz_entry), py::arg("i"),
             py::arg("nz_i"))
        .def(
            "row_nz_entry", [](SparseMatrix<scalar_t>& mat, int i, int nz_i, scalar_t value) {
                mat.row_nz_entry(i, nz_i) = value;
            },
            py::arg("i"), py::arg("nz_i"), py::arg("value"));

    mod.def("equals", static_cast<bool (*)(const Vector<scalar_t>&, const Vector<scalar_t>&)>(&equals),
            py::arg("lhs"), py::arg("rhs"));
    mod.def("assign", &assign<scalar_t>, py::arg("lhs"), py::arg("rhs"));
    mod.def("equals", static_cast<bool (*)(const SparseMatrix<scalar_t>&, const SparseMatrix<scalar_t>&)>(&equals),
            py::arg("lhs"), py::arg("rhs"));
    mod.def("add", &add<scalar_t>, py::arg("result"), py::arg("lhs"), py::arg("rhs"));
    mod.def("subtract", &subtract<scalar_t>, py::arg("result"), py::arg("lhs"), py::arg("rhs"));
    mod.def("dot_product", &dot_product<scalar_t>, py::arg("lhs"), py::arg("rhs"));
    mod.def("multiply",
            static_cast<void (*)(Vector<scalar_t>&, const Vector<scalar_t>&, const scalar_t&)>(&multiply<scalar_t>),
            py::arg("result"), py::arg("lhs"), py::arg("rhs"));
    mod.def("multiply", static_cast<void (*)(Vector<scalar_t>&, const SparseMatrix<scalar_t>&, const Vector<scalar_t>&)>(&multiply<scalar_t>), py::arg("result"),
            py::arg("lhs"), py::arg("rhs"));
    mod.def("norm", &norm<scalar_t>, py::arg("vec"));
    mod.def("calc_res", &calc_res<scalar_t>, py::arg("res"), py::arg("A"), py::arg("x"), py::arg("b"));

    py::class_<Solver<scalar_t>, PySolver>(mod, "Solver")
        .def(py::init<>())
        .def("set_operator", &Solver<scalar_t>::set_operator, py::arg("A"))
        .def("setup", &Solver<scalar_t>::setup)
        .def("solve", &Solver<scalar_t>::solve, py::arg("x"), py::arg("b"))
        .def("last_stop_reason", [](const Solver<scalar_t>& solver) { solver.last_stop_reason(); });

    py::class_<IterativeSolver<scalar_t>, PyIterativeSolver>(mod, "IterativeSolver")
        .def(py::init<>())
        .def("set_operator", &IterativeSolver<scalar_t>::set_operator, py::arg("A"))
        .def("setup", &IterativeSolver<scalar_t>::setup)
        .def("last_stop_reason", &IterativeSolver<scalar_t>::last_stop_reason)
        .def("solve", &IterativeSolver<scalar_t>::solve, py::arg("x"), py::arg("b"))
        .def("set_preconditioner", &IterativeSolver<scalar_t>::set_preconditioner, py::arg("preconditioner"))
        .def("max_iterations", py::overload_cast<std::optional<int>>((void (IterativeSolver<scalar_t>::*)(std::optional<int>)) & IterativeSolver<scalar_t>::max_iterations), py::arg("value"))
        .def("max_iterations", py::overload_cast<>(
                                   (std::optional<int>(IterativeSolver<scalar_t>::*)() const) & IterativeSolver<scalar_t>::max_iterations,
                                   py::const_))
        .def("absolute_tolerance", py::overload_cast<>(
                                       (scalar_t(IterativeSolver<scalar_t>::*)() const) & IterativeSolver<scalar_t>::absolute_tolerance,
                                       py::const_))
        .def("absolute_tolerance", py::overload_cast<scalar_t>((void (IterativeSolver<scalar_t>::*)(scalar_t)) & IterativeSolver<scalar_t>::absolute_tolerance),
             py::arg("value"))
        .def("relative_tolerance", py::overload_cast<std::optional<scalar_t>>((void (IterativeSolver<scalar_t>::*)(
                                                                                  std::optional<scalar_t>)) &
                                                                              IterativeSolver<scalar_t>::relative_tolerance))
        .def("relative_tolerance", py::overload_cast<>(
                                       (std::optional<scalar_t>(IterativeSolver<scalar_t>::*)() const) & IterativeSolver<scalar_t>::relative_tolerance,
                                       py::const_))
        .def("last_iterations", &IterativeSolver<scalar_t>::last_iterations)
        .def("last_residual_norm", &IterativeSolver<scalar_t>::last_stop_reason);

    py::class_<Preconditioner<scalar_t>, PyPreconditioner, std::shared_ptr<Preconditioner<scalar_t>>>(mod, "Preconditioner")
        .def(py::init<>())
        .def("set_operator", &Preconditioner<scalar_t>::set_operator, py::arg("A"))
        .def("setup", &Preconditioner<scalar_t>::setup)
        .def("apply", &Preconditioner<scalar_t>::apply, py::arg("x"), py::arg("b"));

    py::class_<CgSolver<scalar_t>, IterativeSolver<scalar_t>>(mod, "CgSolver")
        .def(py::init<>());

    py::class_<RichardsonSolver<scalar_t>, IterativeSolver<scalar_t>>(mod, "RichardsonSolver")
        .def(py::init<>());

    py::class_<GaussSeidelIteration<scalar_t>, Preconditioner<scalar_t>, std::shared_ptr<GaussSeidelIteration<scalar_t>>>(mod, "GaussSeidelIteration")
        .def(py::init<>());

    py::class_<JacobiIteration<scalar_t>, Preconditioner<scalar_t>, std::shared_ptr<JacobiIteration<scalar_t>>>(mod, "JacobiIteration")
        .def(py::init<>());

    py::class_<MultiIndex>(mod, "MultiIndex")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def("size", &MultiIndex::size)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__getitem__", static_cast<int& (MultiIndex::*)(int)>(&MultiIndex::operator[]), py::is_operator())
        .def(
            "__setitem__", [](MultiIndex& index, int i, int value) { index[i] = value; }, py::is_operator());

    mod.def("from_multi_index", &from_multi_index);
    mod.def("to_multi_index", &to_multi_index);

    py::class_<Point>(mod, "Point")
        .def(py::init<>())
        .def(py::init<scalar_t, scalar_t>())
        .def("size", &Point::size)
        .def("__getitem__", static_cast<scalar_t& (Point::*)(int)>(&Point::operator[]), py::is_operator())
        .def(
            "__setitem__", [](Point& point, int i, scalar_t value) { point[i] = value; }, py::is_operator());

    py::enum_<NeighborSuccession>(mod, "NeighborSuccession")
        .value("predecessor", NeighborSuccession::predecessor)
        .value("successor", NeighborSuccession::successor);

    py::class_<RegularGrid>(mod, "RegularGrid")
        .def(py::init<>())
        .def(py::init<const RegularGrid&>(), py::arg("other"))
        .def(py::init<Point, Point, MultiIndex>(), py::arg("min_corner"), py::arg("max_corner"),
             py::arg("node_count_per_dimension"))
        .def(py::init(
                 [](MpiCommWrapper communicator, Point min_corner, Point max_corner, MultiIndex global_node_count_per_dimension) {
                     return RegularGrid(communicator, min_corner, max_corner, global_node_count_per_dimension);
                 }),
             py::arg("communicator"), py::arg("min_corner"), py::arg("max_corner"),
             py::arg("node_count_per_dimension"))
        .def("node_count_per_dimension", py::overload_cast<>(&RegularGrid::node_count_per_dimension, py::const_))
        .def("node_count_per_dimension", py::overload_cast<int>(&RegularGrid::node_count_per_dimension, py::const_))
        .def("number_of_nodes", &RegularGrid::number_of_nodes)
        .def("number_of_inner_nodes", &RegularGrid::number_of_inner_nodes)
        .def("number_of_boundary_nodes", &RegularGrid::number_of_boundary_nodes)
        .def("number_of_neighbors", &RegularGrid::number_of_neighbors, py::arg("node_index"))
        .def("neighbors_of", &RegularGrid::neighbors_of, py::arg("node_index"), py::arg("neighbors"))
        .def("is_boundary_node", &RegularGrid::is_boundary_node, py::arg("node_index"))
        .def("node_coordinates", &RegularGrid::node_coordinates, py::arg("node_index"))
        .def("node_neighbor_distance", &RegularGrid::node_neighbor_distance, py::arg("node_index"),
             py::arg("neighbor_direction"), py::arg("neighbor_succession"))
        .def("processes_per_dimension", &RegularGrid::processes_per_dimension)
        .def("local_process_coordinates", &RegularGrid::local_process_coordinates)
        .def("global_node_count_per_dimension", &RegularGrid::global_node_count_per_dimension)
        .def("global_multi_to_singleindex", &RegularGrid::global_multi_to_singleindex, py::arg("a"))
        .def("global_single_to_multiindex", &RegularGrid::global_single_to_multiindex, py::arg("i"));

    py::class_<GridFunction<scalar_t>>(mod, "GridFunction")
        .def(py::init<RegularGrid&, scalar_t>())
        .def(py::init<RegularGrid&, const std::function<scalar_t(const Point&)>>())
        .def(py::init<RegularGrid&, const Vector<scalar_t>&>())
        .def("grid", &GridFunction<scalar_t>::grid)
        .def("value", &GridFunction<scalar_t>::value);



    mod.def(
        "write_to_vtk", [](const std::string& path, const GridFunction<scalar_t>& grid_function, const std::string& name) { write_to_vtk(path, grid_function, name); },
        py::arg("file_path"), py::arg("grid_function"), py::arg("name"));

    mod.def("assemble_poisson_matrix", &assemble_poisson_matrix<scalar_t>, py::arg("grid"), py::arg("rhs_function"),
            py::arg("boundary_function"));

     mod.def(
        "assemble_heat_matrix", [](const RegularGrid& grid, const GridFunction<scalar_t>& previous_temperature, const scalar_t t, const scalar_t delta_t, 
            const std::function<scalar_t(const Point&, const scalar_t)>& rhs_function, 
            const std::function<scalar_t(const Point&, const scalar_t)>& boundary_function) 
         { return assemble_heat_matrix(grid, previous_temperature, t, delta_t, rhs_function, boundary_function); }, 
         py::arg("grid"), py::arg("previous_temperature"), py::arg("t"), py::arg("delta_t"), py::arg("rhs_function"),
        py::arg("boundary_function"));

     mod.def("compute_l_infinity_error", &compute_l_infinity_error<scalar_t>, py::arg("grid"), py::arg("computed_solution"), py::arg("analytical_solution"));

     mod.def(
         "to_global_index", [](const Vector<scalar_t>& vector, int local_index) { return (vector.partition()).to_global_index(local_index); },
         py::arg("Vector"), py::arg("local_index"));
}
