//
// Created by trist on 2/5/2021.
//
#include "discretization/poisson.h"
#include "grid/grid.h"
#include "grid/gridfunction.h"
#include "grid/io.h"
#include "grid/point.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/vector.h"
#include "solvers/cg.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <thread>
#include <unistd.h>

scalar_t boundary(const Point& x)
{
    return x[0] * x[0] + 2 * x[1] * x[1] + 1;
}

scalar_t function(const Point& x)
{
    return -6;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    {
        using namespace std::chrono_literals;
        volatile int i = 0;
        printf("PID %d\n", getpid());
        fflush(stdout);
        while(i == 0)
            std::this_thread::sleep_for(100ms);
    }

    RegularGrid grid(MPI_COMM_WORLD, Point(0, 0), Point(1., 1.), MultiIndex(16, 16));
    SparseMatrix<scalar_t> matrix;
    Vector<scalar_t> vector;
    std::tie(matrix, vector) = assemble_poisson_matrix<scalar_t>(grid, function, boundary);
    Vector<scalar_t> x(vector.partition());
    CgSolver<scalar_t> solver;

    solver.set_operator(matrix);
    solver.relative_tolerance(1e-15);
    solver.max_iterations(1000);
    solver.absolute_tolerance(0);
    solver.setup();
    solver.solve(x, vector);
    GridFunction<scalar_t> gridFunction(grid, x);

    write_to_vtk("./poisson_func_2", gridFunction, "poisson_func");
    MPI_Finalize();
}