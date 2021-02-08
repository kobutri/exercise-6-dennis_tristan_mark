#include "../grid/grid.h"
#include "../grid/gridfunction.h"
#include "../linear_algebra/operations.h"
#include <cmath>

template<typename T>
T compute_l_infinity_error(const RegularGrid& grid,
                           const GridFunction<T>& computed_solution,
                           const std::function<T(const Point&)>& analytical_solution) {
    T max = 0;
    for(int i = 0; i < grid.partition().local_size(); ++i)
    {
        T val = std::abs(analytical_solution(grid.node_coordinates(grid.partition().to_global_index(i))) - computed_solution.value(i));
        if(val > max) {
            max = val;
        }
    }

    T global_max = 0;
    MPI_Allreduce(&max, &global_max, 1, convert_to_MPI_TYPE<T>(), MPI_MAX, grid.partition().communicator());
    return global_max;
}