#ifndef PMSC_GRID_H
#define PMSC_GRID_H

#include <array>
#include <memory>
#include <mpi.h>
#include <utility>
#include <vector>

#include "../common/space_dimension.h"
#include "../linear_algebra/contiguousparallelpartition.h"
#include "multiindex.h"
#include "point.h"

enum class NeighborSuccession {
    predecessor,
    successor
};

class RegularGrid {
public:
    using triplet_type = std::tuple<scalar_t, scalar_t, int>;

    RegularGrid() = default;

    RegularGrid(const RegularGrid &other);

    RegularGrid(RegularGrid &&other) noexcept;

    RegularGrid(MPI_Comm communicator, Point min_corner, Point max_corner, MultiIndex global_node_count_per_dimension);

    //! Creates a regular grid covering a specified domain.
    //!
    //! \param min_corner                The min corner \f$\bm{l}\f$ of the domain \f$\Omega\f$ to be covered by the grid.
    //! \param max_corner                The min corner \f$\bm{u}\f$ of the domain \f$\Omega\f$ to be covered by the grid.
    //! \param node_count_per_dimension  Number of nodes in each space dimension \f$\bm{n}\f$.
    RegularGrid(Point min_corner, Point max_corner, MultiIndex node_count_per_dimension);

    //! Returns the number of nodes in each space dimension \f$\bm{n}\f$.
    MultiIndex node_count_per_dimension() const;

    //! Returns the total number of nodes in the grid.
    int number_of_nodes() const;

    //! Returns the number of nodes that are not on the boundary of the domain.
    int number_of_inner_nodes() const;

    //! Returns the number of nodes that are on the boundary of the domain.
    int number_of_boundary_nodes() const;

    //! Returns the number of neighbors a node identified by its one-dimensional index \f$ i \f$ has.
    int number_of_neighbors(int local_node_index) const;

    //! Computes the indices of the neighbors of a given node.
    //!
    //! \param node_index The one-dimensional index of the node whose neighbors to compute.
    //! \param neighbors  The neighbors of the given node in each space dimension.
    //!                   The first entry in the pair for a given space dimension should be the
    //!                   one-dimensional index of the predecessing neighbor of the given node.
    //!                   The second entry in the pair for a given space dimension should be the
    //!                   one-dimensional index of the successing neighbor of the given node.
    //!                   If the given node does not have a neighbor in any of those directions the returned index
    //!                   should be -1 at that location.
    //!
    //! \returns The number of neighbors the given node has.
    int neighbors_of(int local_node_index, std::array<std::pair<int, int>, space_dimension> &neighbors) const;

    //! Returns true if the node with the given index is located at the boundary of the domain.
    bool is_boundary_node(int global_node_index) const;

    //! Returns the coordinates \f$\bm{x}\f$ of a given node.
    Point node_coordinates(int global_node_index) const;

    //! Returns the distance of a node to one of his neighbors.
    //!
    //! \param node_index           The one-dimensional index of the node to get the neighbor distance of.
    //! \param neighbor_direction   The direction in space in which to return the neighbor distance.
    //! \param neighbor_succession  Indicates whether to return the distance to the predecessor or the successor
    //!                             in the specified space dimension.
    scalar_t
    node_neighbor_distance(int local_node_index, int neighbor_direction, NeighborSuccession neighbor_succession) const;

    const ContiguousParallelPartition &partition() const;

    MultiIndex processes_per_dimension() const;

    MultiIndex local_process_coordinates() const;

    MultiIndex global_node_count_per_dimension() const;

    //MultiIndex node_count_per_dimension() const;

    MultiIndex node_count_per_dimension(int process_rank) const;

    int global_multi_to_singleindex(const MultiIndex &a) const;

    MultiIndex global_single_to_multiindex(int i) const;

private:
    Point min_corner_;
    Point max_corner_;
    MultiIndex node_count_per_dimension_;
    MultiIndex process_per_dim_;
    ContiguousParallelPartition partition_;
};

#endif // PMSC_GRID_H