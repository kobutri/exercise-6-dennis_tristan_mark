//
// Created by trist on 2/10/2021.
//
#define _USE_MATH_DEFINES
#include <mpi.h>
#include <functional>
#include <cmath>
#include "grid/multiindex.h"
#include "grid/point.h"
#include "grid/grid.h"
#include "discretization/poisson.h"
#include "discretization/error.h"
#include "solvers/solver.h"
#include "solvers/cg.h"

double error_measure(MPI_Comm comm, MultiIndex nodes_per_dim, const std::function<double (const Point&)>& boundary_function, const std::function<double (const Point&)>& poisson_function) {
    Point min_corner(0, 0);
    Point max_corner(1, 1);
    RegularGrid grid(comm, min_corner, max_corner, nodes_per_dim);
    SparseMatrix<double> mat;
    Vector<double> vec;
    std::tie(mat, vec) = assemble_poisson_matrix(grid, poisson_function, boundary_function);
    Vector<double> x(vec.partition());
    CgSolver<double> solver;
    solver.relative_tolerance(1e-15);
    solver.max_iterations(1000);
    solver.absolute_tolerance(0.0);
    solver.set_operator(mat);
    solver.setup();
    std::cout << x.partition().local_size() << std::endl;
    solver.solve(x, vec);
    GridFunction<double> func(grid, x);
    return compute_l_infinity_error(grid, func, boundary_function);
}

double boundary_function(const Point& x) {
    //return sin(M_PI * x[0]) * sin(M_PI * x[1]);
    return x[0]*x[0] + 2 * x[1]*x[1] + 1;
}

double poisson_function([[maybe_unused]] const Point& x) {
    //return 2 * M_PI * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
    return -6;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CgSolver<double> solver;
    solver.relative_tolerance(1e-15);
    solver.max_iterations(1000);
    solver.absolute_tolerance(0.0);

    for (int i = 2; i < 3; ++i) {
        double err = error_measure(MPI_COMM_WORLD, MultiIndex(1 << i, 1 << i), boundary_function, poisson_function);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0) {
            std::cout << err << std::endl;
        }
    }
    MPI_Finalize();
}