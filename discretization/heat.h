#ifndef HEAT_H
#define HEAT_H

template<typename T>
std::pair<SparseMatrix<T>, Vector<T>>
assemble_heat_matrix(const RegularGrid &grid, const GridFunction<T> &previous_temperature,
                     const scalar_t t, const scalar_t delta_t,
                     const std::function<T(const Point &, const scalar_t)> &rhs_function, //
                     const std::function<T(const Point &, const scalar_t)> &boundary_function) {

    std::function<T(const Point &)> rhs_function_t = [=](const Point &point) { return rhs_function(point, t); }; //since assemble_poisson_matrix is used to create matrix 
    std::function<T(const Point &)> boundary_function_t = [=](const Point &point) {                             //and vector for timeindependent differential equation
        return boundary_function(point, t);                                                                    //the function has to be independent of time t for assemble_poisson_matrix
    };


    ContiguousParallelPartition partition = grid.partition();
    std::pair<SparseMatrix<T>, Vector<T>> solution_poisson = assemble_poisson_matrix(grid, rhs_function_t,
                                                                                     boundary_function_t);
    SparseMatrix<T> Matrix = solution_poisson.first;
    Vector<T> vector = solution_poisson.second;

    for (int i = 0; i < vector.size(); i++)  //                                 change of matrix and vector given by assemble_poisson_matrix 
    {
        int i_global = partition.to_global_index(i);
        if (grid.is_boundary_node(i_global) == false) {
            vector[i] += previous_temperature.value(i) / delta_t;  //i_global
        }
    }
    for (int i = 0; i < Matrix.rows(); i++) {
        int i_global = partition.to_global_index(i);
        for (int j = 0; j < Matrix.row_nz_size(i); j++) {
            if (i_global == Matrix.row_nz_index(i, j) && grid.is_boundary_node(i_global) == false) {
                Matrix.row_nz_entry(i, j) += 1 / delta_t;
            }
        }
    }
    return std::make_pair(Matrix, vector);
}

#endif