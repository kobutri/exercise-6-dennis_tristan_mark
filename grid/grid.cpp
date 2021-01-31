#include <cassert>
#include <iostream>

#include "grid.h"

RegularGrid::RegularGrid(const RegularGrid& other) :
    min_corner_(other.min_corner_),
    max_corner_(other.max_corner_),
    node_count_per_dimension_(other.node_count_per_dimension_) {}

RegularGrid::RegularGrid(RegularGrid&& other) noexcept :
    min_corner_(other.min_corner_),
    max_corner_(other.max_corner_),
    node_count_per_dimension_(other.node_count_per_dimension_) {}

RegularGrid::RegularGrid(Point min_corner, Point max_corner, MultiIndex node_count_per_dimension_) :
    min_corner_(min_corner),
    max_corner_(max_corner),
    node_count_per_dimension_(node_count_per_dimension_)
{
    for(int i = 0; i < space_dimension; i++)
    {
        assert(min_corner[i] <= max_corner[i]);
        assert(node_count_per_dimension_[i] >= 0);
    }
}

RegularGrid::RegularGrid(MPI_Comm communicator, Point min_corner, Point max_corner, MultiIndex global_node_count_per_dimension) :
    min_corner_(min_corner), max_corner_(max_corner), node_count_per_dimension_(global_node_count_per_dimension)
{
    int number_of_nodes = 0;

    //array_node_counts(space_dimension) int *dims,
    int array_node_counts[space_dimension];
    
    for(int i = 0; i < global_node_count_per_dimension.size(); i++)
    {
        array_node_counts[i] = global_node_count_per_dimension[i];
    }
    
    MPI_Comm newcomm;
    int periods[space_dimension];
    for(int i = 0; i < space_dimension; i++)
    {
        periods[i] = 1;
    }
    for(int i = 0; i < space_dimension; i++)
    {
        number_of_nodes *= global_node_count_per_dimension[i];
    }

    MPI_Dims_create(number_of_nodes, space_dimension, array_node_counts);

    MPI_Cart_create(communicator, space_dimension, array_node_counts, periods, 1, &newcomm);

    std::vector<int> vector_node_counts(space_dimension);

    for(int i = 0; i < space_dimension; i++)
    {
        vector_node_counts[i] = array_node_counts[i];
        new_node_counts_[i] = array_node_counts[i];
    }

    partition_ = ContiguousParallelPartition(newcomm, vector_node_counts);

}

MultiIndex RegularGrid::global_node_count_per_dimension() const
{
    return node_count_per_dimension_;
}

int RegularGrid::number_of_nodes() const         //there are no changes
{
    int result = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        result *= node_count_per_dimension_[i];
    }
    return result;
}

int RegularGrid::number_of_inner_nodes() const   //there are no changes
{
    int result = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        result *= node_count_per_dimension_[i] - 2;
    }
    return result;
}

int RegularGrid::number_of_boundary_nodes() const    //there are no changes
{
    return number_of_nodes() - number_of_inner_nodes();
}

int RegularGrid::number_of_neighbors(int node_index) const
{
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    node_index = partition_.to_global_index(node_index, partition_.process());
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    int boundary_type = 0;
    for(int i = 0; i < space_dimension; i++)
    {
        if(node[i] == 0 || node[i] == node_count_per_dimension_[i] - 1) boundary_type += 1;
    }
    return (space_dimension - boundary_type) * 2 + boundary_type;
}

int RegularGrid::neighbors_of(int local_node_index, std::array<std::pair<int, int>, space_dimension>& neighbors) const
{
    int node_index = local_node_index;
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());

    node_index = partition_.to_global_index(node_index, partition_.process());
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    MultiIndex neighbor_before;
    MultiIndex neighbor_after;
    for(int i = 0; i < space_dimension; i++)
    {
        neighbor_before = node;
        neighbor_after = node;
        if(node[i] == 0)
        {
            neighbor_after[i] += 1;
            neighbors[i] = std::make_pair(-1, multi_to_single_index(neighbor_after, node_count_per_dimension_));
        }
        else if(node[i] == node_count_per_dimension_[i] - 1)
        {
            neighbor_before[i] -= 1;
            neighbors[i] = std::make_pair(multi_to_single_index(neighbor_before, node_count_per_dimension_), -1);
        }
        else
        {
            neighbor_after[i] += 1;
            int dist = multi_to_single_index(neighbor_after, node_count_per_dimension_) - node_index;
            neighbors[i] = std::make_pair(node_index - dist, node_index + dist);
        }
    }
    return number_of_neighbors(node_index);
}

bool RegularGrid::is_boundary_node(int global_node_index) const
{
    int node_index = global_node_index;
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    bool result = false;
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    for(int i = 0; i < space_dimension; i++)
    {
        if(node[i] == 0 || node[i] == node_count_per_dimension_[i] - 1)
        {
            result = true;
            break;
        }
    }
    return result;
}

Point RegularGrid::node_coordinates(int global_node_index) const
{
    int node_index = global_node_index;
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());

    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    Point result = Point();
    for(int i = 0; i < space_dimension; i++)
    {
        result[i] = min_corner_[i] + node[i] * (max_corner_[i] - min_corner_[i]) / (node_count_per_dimension_[i] - 1);
    }
    return result;
}

scalar_t RegularGrid::node_neighbor_distance(int local_node_index, int neighbor_direction, NeighborSuccession neighbor_succession) const
{
    int node_index = local_node_index;
    assert(node_index >= 0);
    assert(node_index < number_of_nodes());
    assert(neighbor_direction >= 0);
    assert(neighbor_direction < space_dimension);
    int i = (int)neighbor_succession;
    node_index = partition_.to_global_index(node_index, partition_.process());
    MultiIndex node = single_to_multiindex(node_index, node_count_per_dimension_);
    if(node[neighbor_direction] == 0 + i * node_count_per_dimension_[neighbor_direction])
        return -1;
    else
        return (max_corner_[neighbor_direction] - min_corner_[neighbor_direction]) /
               (node_count_per_dimension_[neighbor_direction] - 1);
}

const ContiguousParallelPartition& RegularGrid::partition() const
{
    return partition_;
}

MultiIndex RegularGrid::processes_per_dimension() const     //
{
    MultiIndex coordinates(space_dimension);
    int coords[space_dimension];
    int global_size = partition_.global_size();
    int last_process = partition_.owner_process(global_size - 1);
    MPI_Cart_coords(partition_.communicator(), last_process, space_dimension, coords);
    for(int i = 0; i < space_dimension; i++)
    {
        coordinates[i] = coords[i] + 1;
    }

    return coordinates;
}

MultiIndex RegularGrid::local_process_coordinates() const
{
    int coords[space_dimension];
    MPI_Cart_coords(partition_.communicator(), partition_.process(), space_dimension, coords);
    MultiIndex coordinates(space_dimension);
    for(int i = 0; i < space_dimension; i++)
    {
        coordinates[i] = coords[i];
    }
    return coordinates;
}

    /*MultiIndex RegularGrid::global_node_count_per_dimension() const
{
    return node_count_per_dimension_;
}*/

MultiIndex RegularGrid::node_count_per_dimension() const
{
    int process_rank = partition_.process();
    int local_index = partition_.local_size(process_rank) - 1;
    int global_index = partition_.to_global_index(local_index, process_rank);
    MultiIndex process_max_corner = single_to_multiindex(global_index, node_count_per_dimension_);
    MultiIndex node_count_per_dimension(space_dimension);

    global_index = partition_.to_global_index(0);
    MultiIndex process_min_corner = single_to_multiindex(global_index, node_count_per_dimension_);

    for(int i = 0; i < max_corner_.size(); i++)
    {
        node_count_per_dimension[i] = static_cast<int>((process_max_corner[i] - process_min_corner[i]) * node_count_per_dimension_[i] / (max_corner_[i] - min_corner_[i]));
    }
    return node_count_per_dimension;
    //int coords = local_process_coordinates();
    //MultiIndex node_count_per_dimension();
    //int process = partition_.process();
    //int global_index = to_global_index(local_index);
    /*for(int i = 0; i < max_corner_.size(); i++)
    {
        node_count_per_dimension[i] = //node_count_per_dimension_[i]/(max_corner_[i] - min_corner_[i]);
    }
    return node_count_per_dimension;*/
   // return new_node_counts_;
}

MultiIndex RegularGrid::node_count_per_dimension(int process_rank) const
{
    
    int local_index = partition_.local_size(process_rank)-1;
    int global_index = partition_.to_global_index(local_index, process_rank);
    MultiIndex process_max_corner = single_to_multiindex(global_index, node_count_per_dimension_);
    MultiIndex node_count_per_dimension(space_dimension);
    
    global_index = partition_.to_global_index(0, process_rank);
    MultiIndex process_min_corner = single_to_multiindex(global_index, node_count_per_dimension_);
    
    for(int i = 0; i < max_corner_.size(); i++)
    {
        node_count_per_dimension[i] = static_cast<int>((process_max_corner[i] - process_min_corner[i]) * node_count_per_dimension_[i] / (max_corner_[i] - min_corner_[i]));
    }
    return node_count_per_dimension;
}