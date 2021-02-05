#ifndef POISSON_H
#define POISSON_H

#include "../grid/grid.h"
#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"
#include <functional>
#include <tuple>
#include <utility>

template<typename T>
std::pair<SparseMatrix<T>, Vector<T>> assemble_poisson_matrix(const RegularGrid &grid,
                                                              const std::function<T(const Point &)> &rhs_function,
                                                              const std::function<T(
                                                                  const Point &)> &boundary_function) {
    //std::vector<typename SparseMatrix<T>::triplet_type> v;
    const int max_neighbor = 2*space_dimension;
    std::function<int(int)> nnz_per_index = [&grid,max_neighbor](int i)
    {
      return 1+ (grid.is_boundary_node(i) ? 0 : max_neighbor);
    };
    Vector<T> b(grid.number_of_nodes());
    SparseMatrix<T> m(grid.number_of_nodes(), grid.number_of_nodes(), nnz_per_index);
    int entry_counter;
    for (int i = 0; i < grid.number_of_nodes(); ++i) {
        MultiIndex a = single_to_multiindex(i, grid.node_count_per_dimension());
        entry_counter = 1;
        if (grid.is_boundary_node(i)) {
            m.row_nz_index(i,0) = i;
            m.row_nz_entry(i,0) = 1;
            //v.push_back(std::make_tuple(i, i, 1));
            b[i] = boundary_function(grid.node_coordinates(i));
            continue;
        }
        b[i] = rhs_function(grid.node_coordinates(i));
        T uii_sum = 0;
        for (int j = 0; j < space_dimension; ++j) {
            T h2 = grid.node_neighbor_distance(i, j, NeighborSuccession::successor);
            h2 *= h2;
            uii_sum += 2 / h2;
            MultiIndex a_ = a;
            a_[j] += 1;
            int i_ = multi_to_single_index(a_, grid.node_count_per_dimension());
            m.row_nz_index(i,entry_counter) = i_;
            m.row_nz_entry(i,entry_counter) = (-1) / h2;
            entry_counter += 1;
            //v.push_back(std::make_tuple(i, i_, (-1) / h2));
            a_ = a;
            a_[j] -= 1;
            i_ = multi_to_single_index(a_, grid.node_count_per_dimension());
            m.row_nz_index(i,entry_counter) = i_;
            m.row_nz_entry(i,entry_counter) = (-1) / h2;
            entry_counter += 1;
            //v.push_back(std::make_tuple(i, i_, (-1) / h2));
        }
        m.row_nz_index(i,0) = i;
        m.row_nz_entry(i,0) = uii_sum;
        //v.push_back(std::make_tuple(i, i, uii_sum));
    }
    //SparseMatrix<T> m(grid.number_of_nodes(), grid.number_of_nodes(), v);

    for (int i = 0; i < grid.number_of_nodes(); ++i) {
        if (!grid.is_boundary_node(i)) {
            continue;
        }

        std::array<std::pair<int, int>, space_dimension> neighbors;
        grid.neighbors_of(i, neighbors);
        for (int j = 0; j < space_dimension; ++j) {
            for (int k = 0; k < 2; ++k) {
                int index = k == 0 ? neighbors[j].first : neighbors[j].second;
                if (index != -1 && !grid.is_boundary_node(index)) {
                    for (int l = 0; l < m.row_nz_size(index); ++l) {
                        if (i == m.row_nz_index(index, l)) {
                            b[index] -= m.row_nz_entry(index, l) * boundary_function(grid.node_coordinates(i));
                            m.row_nz_entry(index, l) = 0;
                        }
                    }
                }
            }
        }
    }

    return std::make_pair(m, b);
}

#endif //POISPOISSON_H