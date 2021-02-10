//
// Created by trist on 2/10/2021.
//
#define _USE_MATH_DEFINES
#include <mpi.h>
#include <functional>
#include <cmath>
#include <chrono>
#include <thread>
#include "grid/multiindex.h"
#include "grid/point.h"
#include "grid/grid.h"
#include "discretization/poisson.h"
#include "discretization/error.h"
#include "solvers/solver.h"
#include "solvers/cg.h"
#include "solvers/jacobi_iteration.h"

double error_measure(MPI_Comm comm, MultiIndex nodes_per_dim, const std::function<double (const Point&)>& boundary_function, const std::function<double (const Point&)>& poisson_function) {
    Point min_corner(0, 0);
    Point max_corner(20, 20);
    RegularGrid grid(comm, min_corner, max_corner, nodes_per_dim);
    SparseMatrix<double> mat;
    Vector<double> vec;
    std::tie(mat, vec) = assemble_poisson_matrix(grid, poisson_function, boundary_function);
    //mat.print();
    Vector<double> x(vec.partition());
    CgSolver<double> solver;
    solver.relative_tolerance(1e-15);
    solver.max_iterations(1000);
    solver.absolute_tolerance(0.0);
    solver.set_operator(std::make_shared<SparseMatrix<double>>(mat));
    solver.setup();
    solver.solve(x, vec);
    GridFunction<double> func(grid, x);
    return compute_l_infinity_error(grid, func, boundary_function);
//return 0;
}

double boundary_function(const Point& x) {
    return sin(M_PI * x[0]) * sin(M_PI * x[1]);
}

double poisson_function([[maybe_unused]] const Point& x) {
    return 2 * M_PI * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CgSolver<double> solver;
    solver.relative_tolerance(1e-15);
    solver.max_iterations(10000000);
    solver.absolute_tolerance(0.0);
    solver.set_preconditioner(std::make_shared<JacobiIteration<double>>(JacobiIteration<double>()));

    {
//        using namespace std::chrono_literals;
//        volatile int i = 0;
//        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//        printf("PID %d with rank %d\n", _getpid(), rank);
//        while (i == 0){
//            std::this_thread::sleep_for(100ms);
//        }
    }

    for (int i = 5; i < 12; ++i) {
        auto err = error_measure(MPI_COMM_WORLD, MultiIndex(1 << i, 1 << i), boundary_function, poisson_function);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0) {
            std::cout << err << std::endl;
        }
    }
    MPI_Finalize();
}