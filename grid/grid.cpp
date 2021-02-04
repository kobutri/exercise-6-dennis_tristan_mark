#include <cassert>
#include <iostream>

#include "grid.h"

RegularGrid::RegularGrid(const RegularGrid& other) :
    min_corner_(other.min_corner_),
    max_corner_(other.max_corner_),
    node_count_per_dimension_(other.node_count_per_dimension_),
    partition_(other.partition_),
    local_min_corner_(other.local_min_corner_),
    local_node_count_per_dimension_(other.local_node_count_per_dimension_) {}

RegularGrid::RegularGrid(RegularGrid&& other) noexcept :
    min_corner_(other.min_corner_),
    max_corner_(other.max_corner_),
    node_count_per_dimension_(other.node_count_per_dimension_),
    partition_(other.partition_),
    local_min_corner_(other.local_min_corner_),
    local_node_count_per_dimension_(other.local_node_count_per_dimension_) {}

RegularGrid::RegularGrid(Point min_corner, Point max_corner, MultiIndex node_count_per_dimension_) :
    min_corner_(min_corner),
    max_corner_(max_corner),
    node_count_per_dimension_(node_count_per_dimension_),
    local_min_corner_(MultiIndex(0)),
    local_node_count_per_dimension_(node_count_per_dimension_)
{
    for(int i = 0; i < space_dimension; i++)
    {
        assert(min_corner[i] <= max_corner[i]);
        assert(node_count_per_dimension_[i] >= 0);

    }
    partition_  = ContiguousParallelPartition(MPI_COMM_SELF, {0, number_of_nodes()});
}

RegularGrid::RegularGrid(MPI_Comm communicator, Point min_corner, Point max_corner, MultiIndex global_node_count_per_dimension) :
   min_corner_(min_corner),
   max_corner_(max_corner),
   node_count_per_dimension_(global_node_count_per_dimension)
{
    for(int i = 0; i < space_dimension; i++)
    {
        assert(min_corner[i] <= max_corner[i]);
        assert(node_count_per_dimension_[i] >= 0);

    }
    process_per_dim_ = MultiIndex();
    local_min_corner_ = MultiIndex();
    int size;
    MPI_Comm_size(communicator, &size);
    int process_per_dim[space_dimension] = {0};
    int periods[space_dimension] = {0};
    int nodes_counter;
    int coordinates[space_dimension];
    int nodes_for_first_processes;
    MPI_Dims_create(size, space_dimension , process_per_dim);
    for(int i = 0; i < space_dimension; i++)
    {
        process_per_dim_[i] = process_per_dim[space_dimension-i-1];
    }
    MPI_Comm new_communicator;
    MPI_Cart_create(communicator, space_dimension, process_per_dim, periods, false, &new_communicator);
    int number_of_processes = 1;
    for (int i = 0; i< space_dimension; i++)
    {
        number_of_processes *= process_per_dim[i];
       // std::cout << "process_per_dim " << i << " = " << process_per_dim[i] << "\n" << std::flush;
    }
    std::vector<int> partition(number_of_processes+1);
    partition[0] = 0;
    for (int i = 0; i<number_of_processes; i++)
    {
        nodes_counter = 1;
        MPI_Cart_coords(new_communicator, i, space_dimension, coordinates);
        //change(coordinates);
        for (int j = 0; j < space_dimension; j++)
        {
            nodes_for_first_processes = global_node_count_per_dimension[j]/process_per_dim_[j] +(global_node_count_per_dimension[j]%process_per_dim_[j] == 0 ? 0:1);
            if (coordinates[space_dimension-j-1] < process_per_dim_[j]-1)
            {
                nodes_counter *= nodes_for_first_processes;
            }
            else
            {
                nodes_counter *= global_node_count_per_dimension[j]- (process_per_dim_[j]-1)*nodes_for_first_processes;
            }        
        }
        partition[i+1] = partition[i]+nodes_counter ;
    }
    partition_ = ContiguousParallelPartition(new_communicator, partition);
    MPI_Cart_coords(new_communicator, partition_.process(), space_dimension, coordinates);
    //change(coordinates);
    for(int i = 0; i < space_dimension; i++)
    {
        nodes_for_first_processes = global_node_count_per_dimension[i]/process_per_dim_[i] +(global_node_count_per_dimension[i]%process_per_dim_[i] == 0 ? 0:1);
        local_min_corner_[i] = coordinates[space_dimension-i-1] * nodes_for_first_processes;
    }
    local_node_count_per_dimension_ = node_count_per_dimension(partition_.process());
    //std::cout << "local_node_count_per_dimension[0] = " << local_node_count_per_dimension_[0] << "\n";
    //std::cout << "local_node_count_per_dimension[1] = " << local_node_count_per_dimension_[1] << "\n";
    //std::cout << "local_node_count_per_dimension[2] = " << local_node_count_per_dimension_[2] << "\n";
}

int RegularGrid::number_of_nodes() const         
{
    int result = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        result *= node_count_per_dimension_[i];
    }
    return result;
}

int RegularGrid::number_of_inner_nodes() const   
{
    int result = 1;
    for(int i = 0; i < space_dimension; i++)
    {
        result *= node_count_per_dimension_[i] - 2;
    }
    return result;
}

int RegularGrid::number_of_boundary_nodes() const   
{
    return number_of_nodes() - number_of_inner_nodes();
}

int RegularGrid::number_of_neighbors(int local_node_index) const
{
    assert(local_node_index >= 0);
    assert(local_node_index < partition_.local_size());
    int node_index = partition_.to_global_index(local_node_index);
    MultiIndex node = global_single_to_multiindex(node_index);
    int boundary_type = 0;
    for(int i = 0; i < space_dimension; i++)
    {
        if(node[i] == 0 || node[i] == node_count_per_dimension_[i] - 1) boundary_type += 1;
    }
    return (space_dimension - boundary_type) * 2 + boundary_type;
}

int RegularGrid::neighbors_of(int local_node_index, std::array<std::pair<int, int>, space_dimension>& neighbors) const
{
    assert(local_node_index >= 0);
    assert(local_node_index < partition_.local_size());

    int node_index = partition_.to_global_index(local_node_index);
    MultiIndex node = global_single_to_multiindex(node_index);
    MultiIndex neighbor_before;
    MultiIndex neighbor_after;
    for(int i = 0; i < space_dimension; i++)
    {
        neighbor_before = node;
        neighbor_after = node;
        if(node[i] == 0)
        {
            neighbor_after[i] += 1;
            neighbors[i] = std::make_pair(-1, global_multi_to_singleindex(neighbor_after));
        }
        else if(node[i] == node_count_per_dimension_[i] - 1)
        {
            neighbor_before[i] -= 1;
            neighbors[i] = std::make_pair(global_multi_to_singleindex(neighbor_before), -1);
        }
        else
        {
            
            neighbor_after[i] += 1;
            int node_index_after = global_multi_to_singleindex(neighbor_after);
            neighbor_before[i] -= 1;
            int node_index_before = global_multi_to_singleindex(neighbor_before);
            neighbors[i] = std::make_pair(node_index_before, node_index_after);
            //std::cout << "process: " << partition_.process() << "local_node_index: " << local_node_index << "global_node_index: " << node_index << "global_multi_index: " << node[0] <<node[1]<< "dimensin: "<< i << "neighbour before:" << neighbor_before[0]<<neighbor_before[1] << "single_index: " << node_index_before << "neighbour after:" << neighbor_after[0]<<neighbor_after[1] << "single_index: " << node_index_after<< "\n" << std::flush;
        }
    }
    return number_of_neighbors(local_node_index);
}

bool RegularGrid::is_boundary_node(int global_node_index) const
{
    assert(global_node_index >= 0);
    assert(global_node_index < number_of_nodes());
    bool result = false;
    //std::cout <<"process: " << partition_.process()<< " global_node_index: " << global_node_index << std::flush;
    //std::cout << "global_node_index: " << global_node_index << std::flush;
    MultiIndex global_multi_index = global_single_to_multiindex(global_node_index); 
    //std::cout <<"process: " << partition_.process() <<"global_node_index: " << global_node_index <<global_multi_index[0] << "," << global_multi_index[1]<< ","<< global_multi_index[2] << std::flush;
    //std::cout << "global_node_index: " << global_node_index << std::flush;
    //std::cout << global_multi_index[0] << "," << global_multi_index[1]<< ","<< global_multi_index[2] << std::flush;
    for(int i = 0; i < space_dimension; i++)
    {
        if(global_multi_index[i] == 0 || global_multi_index[i] == node_count_per_dimension_[i]-1)
        {
            result = true;
            break;
        }
    }
    return result;
}

Point RegularGrid::node_coordinates(int global_node_index) const
{
    assert(global_node_index >= 0);
    assert(global_node_index < number_of_nodes());

    MultiIndex global_multi_index = global_single_to_multiindex(global_node_index);
    //std::cout <<"process: " << partition_.process() <<"global_node_index: " << global_node_index <<global_multi_index[0] << "," << global_multi_index[1]<< ","<< global_multi_index[2] << std::flush;
    Point result = Point();
    for(int i = 0; i < space_dimension; i++)
    {
        // std::cout << node[i] << "\n";
        result[i] = min_corner_[i] + global_multi_index[i] * (max_corner_[i] - min_corner_[i]) / (node_count_per_dimension_[i] - 1);
    }
    return result;
}

scalar_t RegularGrid::node_neighbor_distance(int local_node_index, int neighbor_direction, NeighborSuccession neighbor_succession) const
{
    int node_index = local_node_index;
    assert(node_index >= 0);
    assert(node_index < partition_.local_size());
    assert(neighbor_direction >= 0);
    assert(neighbor_direction < space_dimension);
    int i = (int)neighbor_succession;
    node_index = partition_.to_global_index(node_index);
    MultiIndex node = global_single_to_multiindex(node_index);
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

MultiIndex RegularGrid::processes_per_dimension() const     
{
    return process_per_dim_;
}

MultiIndex RegularGrid::local_process_coordinates() const
{
    int coords[space_dimension];
    MPI_Cart_coords(partition_.communicator(), partition_.process(), space_dimension, coords);
    //change(coords);
    MultiIndex coordinates(space_dimension);
    for(int i = 0; i < space_dimension; i++)
    {
        coordinates[i] = coords[space_dimension-i-1];
    }
    return coordinates;
}

MultiIndex RegularGrid::global_node_count_per_dimension() const
{
    return node_count_per_dimension_;
}

MultiIndex RegularGrid::node_count_per_dimension() const
{   
    return local_node_count_per_dimension_;
}

MultiIndex RegularGrid::node_count_per_dimension(int process_rank) const
{
    assert(partition_.owner_process(number_of_nodes()-1) >= process_rank);
    assert(process_rank >= 0);
    int coords[space_dimension];
    int nodes_for_first_processes;
    MPI_Cart_coords(partition_.communicator(), process_rank, space_dimension, coords);
    MultiIndex node_count_per_dimension(space_dimension);
    for (int j = 0; j < space_dimension; j++)
    {
        nodes_for_first_processes = node_count_per_dimension_[j]/process_per_dim_[j] +(node_count_per_dimension_[j]%process_per_dim_[j] == 0 ? 0:1);
        if (coords[space_dimension-j-1] < process_per_dim_[j]-1)
        {
            node_count_per_dimension[j] = nodes_for_first_processes;
        }
        else
        {
            node_count_per_dimension[j ]= node_count_per_dimension_[j]- (process_per_dim_[j]-1)*nodes_for_first_processes;
        } 
    }    
    return node_count_per_dimension;
}

int RegularGrid::global_multi_to_singleindex(const MultiIndex& global_multi_index) const
{
    bool is_local_node = true;
    for (int i = 0; i < space_dimension; i++)
    {
        assert(global_multi_index[i] >= 0);
        assert(global_multi_index[i] < node_count_per_dimension_[i]);
        if (global_multi_index[i] < local_min_corner_[i] || global_multi_index[i] >= local_min_corner_[i]+local_node_count_per_dimension_[i]) 
            is_local_node = false;  
    }
    MultiIndex local_multi_index = MultiIndex();
    if (is_local_node)
    {
        for (int i = 0; i < space_dimension; i++)
        {
            local_multi_index[i] = global_multi_index[i] - local_min_corner_[i];  
        }
        int local_index = multi_to_single_index(local_multi_index, local_node_count_per_dimension_);
        return partition_.to_global_index(local_index);
    }
    else
    {
        int rank;
        int coords[space_dimension];
        for (int i = 0; i < space_dimension; i++)
        {
            coords[space_dimension-i-1] = global_multi_index[i]/(node_count_per_dimension_[i]/process_per_dim_[i]+(node_count_per_dimension_[i]%process_per_dim_[i]==0 ? 0:1));
            local_multi_index[i] = global_multi_index[i] - (coords[space_dimension-i-1])*(node_count_per_dimension_[i]/process_per_dim_[i]+(node_count_per_dimension_[i]%process_per_dim_[i]==0 ? 0:1));
        } 
        //change(coords);
        MPI_Cart_rank(partition_.communicator(), coords, &rank); 
        //std::cout << global_multi_index[0] << global_multi_index[1] << "coords: " << coords[0]<<coords[1]<<"rank: " << rank << "local_multi_index: " << local_multi_index[0]<<local_multi_index[1]<<"\n"<<std::flush;
        int local_index = multi_to_single_index(local_multi_index, node_count_per_dimension(rank));
        return partition_.to_global_index(local_index,rank);
    }
    
}

MultiIndex RegularGrid::global_single_to_multiindex(int global_index) const
{
    assert(global_index >= 0);
    assert(global_index < number_of_nodes());
    if (partition_.is_owned_by_local_process(global_index))
    {
        int local_index = partition_.to_local_index(global_index);
        MultiIndex local_multi_index = single_to_multiindex(local_index, local_node_count_per_dimension_);
        MultiIndex global_multi_index = MultiIndex();
        for (int i = 0; i < space_dimension; i++)
        {
            global_multi_index[i] = local_min_corner_[i] + local_multi_index[i];  
        }
        return global_multi_index;
    }
    else
    {
        //std::cout << "hallo" << std::flush;
        int rank = partition_.owner_process(global_index);
        //std::cout << partition_.process() << "sagt hallo, rank: " << rank <<"\n" <<std::flush;
        auto local_multi_index = single_to_multiindex(global_index - partition_.to_global_index(0,rank),node_count_per_dimension(rank));
        MultiIndex global_multi_index = MultiIndex();
        int coordinates[space_dimension];
        MPI_Cart_coords(partition_.communicator(), rank, space_dimension, coordinates);//local_multi_index[0]<<","<<local_multi_index[1]<<","<<local_multi_index[2]
        //change(coordinates);
        //std::cout<< " globalindex: " << global_index << " owner_process: " << rank << " coordinates: " << coordinates[0]<<","<<coordinates[1]<<","<<coordinates[2]<< "\n"<<std::flush ;
        for(int i = 0; i < space_dimension; i++)
        {
            global_multi_index[i] = coordinates[space_dimension-i-1] * (node_count_per_dimension_[i]/process_per_dim_[i]+(node_count_per_dimension_[i]%process_per_dim_[i]==0 ? 0:1))+ local_multi_index[i];  
        }
        return global_multi_index;
    }    
}


