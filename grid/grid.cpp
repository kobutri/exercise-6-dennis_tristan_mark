#include <cassert>
#include <iostream>


#include "grid.h"


RegularGrid::RegularGrid(const RegularGrid &other) :
        min_corner_(other.min_corner_),
        max_corner_(other.max_corner_),
        node_count_per_dimension_(other.node_count_per_dimension_) {}

RegularGrid::RegularGrid(RegularGrid &&other) noexcept:
        min_corner_(other.min_corner_),
        max_corner_(other.max_corner_),
        node_count_per_dimension_(other.node_count_per_dimension_) {}


RegularGrid::RegularGrid(Point min_corner, Point max_corner, MultiIndex node_count_per_dimension_) :
        min_corner_(min_corner),
        max_corner_(max_corner),
        node_count_per_dimension_(node_count_per_dimension_) {
    for (int i = 0; i < space_dimension; i++) {
        assert(min_corner[i] <= max_corner[i]);
        assert(node_count_per_dimension_[i] >= 0);
    }
}

RegularGrid::RegularGrid(MPI_Comm communicator, Point min_corner, Point max_corner, MultiIndex global_node_count_per_dimension):
   // min_corner_(min_corner),
   // max_corner_(max_corner),
   // node_count_per_dimension_(global_node_count_per_dimension_)
{
    int size = 0;
    MPI_Comm_size(communicator, &size);
    int process_per_dim[space_dimension];
    int periods[space_dimension];
    int nodes_counter;
    int coordinates[space_dimension];
    MPI_Dims_create(size, space_dimension , process_per_dim);
    MPI_Comm new_communicator;
    int MPI_Cart_create(communicator, space_dimension, process_per_dim, periods, true, &new_communicator);
    int number_of_processes = 1;
    for (int i = 0; i< space_dimension; i++)
    {
        number_of_processes *= process_per_dim;
    }
    std::vector<int> partition(number_of_processes);
    for (int i = 0; i<number_of_processes; i++)
    {
        nodes_counter = 1;
        MPI_Cart_coords(new_communicator, i, space_dimension, coordinates);
        for (int j = 0; j < space_dimension; j++)
        {
            if (coordinates[j] < process_per_dim[j]-1)
            {
                nodes_counter *= global_node_count_per_dimension[j]/process_per_dim[j];
            }
        }

    }

}

MultiIndex RegularGrid::node_count_per_dimension() const {
    return node_count_per_dimension_;
}

int RegularGrid::number_of_nodes() const {
    int result = 1;
    for (int i = 0; i < space_dimension; i++) {
        result *= node_count_per_dimension_[i];
    }
    return result;
}

int RegularGrid::number_of_inner_nodes() const {
    int result = 1;
    for (int i = 0; i < space_dimension; i++) {
        result *= node_count_per_dimension_[i] - 2;
    }
    return result;
}

int RegularGrid::number_of_boundary_nodes() const {
    return number_of_nodes() - number_of_inner_nodes();
}

int RegularGrid::number_of_neighbors(int node_index) const {
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    int boundary_type = 0;
    for (int i = 0; i < space_dimension; i++) {
        if (node[i] == 0 || node[i] == node_count_per_dimension_[i] - 1) boundary_type += 1;
    }
    return (space_dimension - boundary_type) * 2 + boundary_type;
}

int RegularGrid::neighbors_of(int node_index, std::array<std::pair<int, int>, space_dimension> &neighbors) const {
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    MultiIndex neighbor_before;
    MultiIndex neighbor_after;
    for (int i = 0; i < space_dimension; i++) {
        neighbor_before = node;
        neighbor_after = node;
        if (node[i] == 0) {
            neighbor_after[i] += 1;
            neighbors[i] = std::make_pair(-1, multi_to_single_index(neighbor_after, node_count_per_dimension_));
        } else if (node[i] == node_count_per_dimension_[i] - 1) {
            neighbor_before[i] -= 1;
            neighbors[i] = std::make_pair(multi_to_single_index(neighbor_before, node_count_per_dimension_), -1);
        } else {
            neighbor_after[i] += 1;
            int dist = multi_to_single_index(neighbor_after, node_count_per_dimension_) - node_index;
            neighbors[i] = std::make_pair(node_index - dist, node_index + dist);
        }
    }
    return number_of_neighbors(node_index);
}

bool RegularGrid::is_boundary_node(int node_index) const {
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    bool result = false;
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    for (int i = 0; i < space_dimension; i++) {
        if (node[i] == 0 || node[i] == node_count_per_dimension_[i] - 1) {
            result = true;
            break;
        }
    }
    return result;
}

Point RegularGrid::node_coordinates(int node_index) const {
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    Point result = Point();
    for (int i = 0; i < space_dimension; i++) {
        result[i] = min_corner_[i] + node[i] * (max_corner_[i] - min_corner_[i]) / (node_count_per_dimension_[i] - 1);
    }
    return result;
}

scalar_t RegularGrid::node_neighbor_distance(int node_index, int neighbor_direction,
                                             NeighborSuccession neighbor_succession) const {
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    assert(neighbor_direction >= 0);
    assert(neighbor_direction < space_dimension);
    int i = (int) neighbor_succession;
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    if (node[neighbor_direction] == 0 + i * node_count_per_dimension_[neighbor_direction])
        return -1;
    else
        return (max_corner_[neighbor_direction] - min_corner_[neighbor_direction]) /
               (node_count_per_dimension_[neighbor_direction] - 1);
}

const ContiguousParallelPartition& RegulrarGrid::partition() const
{
    return partition_;
}

MultiIndex RegularGrid::processes_per_dimension() const
{

}


MultiIndex local_process_coordinates() const;

    MultiIndex global_node_count_per_dimension() const;
    MultiIndex node_count_per_dimension() const;
    MultiIndex node_count_per_dimension(int process_rank) const;
