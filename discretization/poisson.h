#ifndef POISSON_H
#define POISSON_H

#include "../common/scalar.h"
#include "../grid/grid.h"
#include "../grid/gridfunction.h"
#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"
#include <functional>
#include <tuple>
#include <utility>

template<typename T>
std::pair<SparseMatrix<T>, Vector<T>> assemble_poisson_matrix(const RegularGrid& grid,
                                                              const std::function<T(const Point&)>& rhs_function,
                                                              const std::function<T(
                                                                  const Point&)>& boundary_function)
{
    const int max_neighbor = 2 * space_dimension;
    std::function<int(int)> nnz_per_index = [&grid, max_neighbor](int i) {
        return 1 + (grid.is_boundary_node(i) ? 0 : max_neighbor);
    };
    Vector<T> b(grid.partition());
    SparseMatrix<T> m(grid.partition(), grid.number_of_nodes(), nnz_per_index);
    int entry_counter;
    for(int i = 0; i < m.rows(); ++i)
    {
        int i_global = grid.partition().to_global_index(i);
        MultiIndex a = grid.global_single_to_multiindex(i_global);
        entry_counter = 1;
        if(grid.is_boundary_node(i_global))
        {
            m.row_nz_index(i, 0) = i_global;
            m.row_nz_entry(i, 0) = 1;
            //v.push_back(std::make_tuple(i, i, 1));
            b[i] = boundary_function(grid.node_coordinates(i_global));
            continue;
        }
        b[i] = rhs_function(grid.node_coordinates(grid.partition().to_global_index(i)));
        T uii_sum = 0;
        for(int j = 0; j < space_dimension; ++j)
        {
            T h2 = grid.node_neighbor_distance(i, j, NeighborSuccession::successor);
            h2 *= h2;
            uii_sum += 2 / h2;
            MultiIndex a_ = a;
            a_[j] += 1;
            int i_ = grid.global_multi_to_singleindex(a_);
            m.row_nz_index(i, entry_counter) = i_;
            m.row_nz_entry(i, entry_counter) = (-1) / h2;
            entry_counter += 1;
            a_ = a;
            a_[j] -= 1;
            i_ = grid.global_multi_to_singleindex(a_);
            m.row_nz_index(i, entry_counter) = i_;
            m.row_nz_entry(i, entry_counter) = (-1) / h2;
            entry_counter += 1;
        }
        m.row_nz_index(i, 0) = i_global;
        m.row_nz_entry(i, 0) = uii_sum;
    }

    for(int i = 0; i < m.rows(); ++i)
    {
        int i_global = grid.partition().to_global_index(i);
        if(grid.is_boundary_node(i_global))
        {
            continue;
        }
        std::array<std::pair<int, int>, space_dimension> neighbors;
        grid.neighbors_of(i, neighbors);
        for(int j = 0; j < space_dimension; ++j)
        {
            for(int k = 0; k < 2; ++k)
            {
                int index = k == 0 ? neighbors[j].first : neighbors[j].second;
                if(grid.is_boundary_node(index))
                {
                    for(int l = 0; l < m.row_nz_size(i); ++l)
                    {
                        if(index == m.row_nz_index(i, l))
                        {
                            b[i] -= m.row_nz_entry(i, l) * boundary_function(grid.node_coordinates(index));
                            m.row_nz_entry(i, l) = 0;
                        }
                    }
                }
            }
        }
    }

    return std::make_pair(m, b);
}

#endif //POISPOISSON_H