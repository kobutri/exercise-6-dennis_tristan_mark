#ifndef EXCHANGEPATTERN_H
#define EXCHANGEPATTERN_H

#include "contiguousparallelpartition.h"
#include <algorithm>
#include <iostream>
#include <utility>

class ExchangePattern
{
public:
    ExchangePattern();

    ExchangePattern(std::vector<int> neighboring_processes, std::vector<std::vector<int>> receive_indices,
                    std::vector<std::vector<int>> send_indices);

    ExchangePattern(ExchangePattern&& other) noexcept;

    ExchangePattern(const ExchangePattern& other);

    ExchangePattern& operator=(const ExchangePattern& other);

    ExchangePattern& operator=(ExchangePattern&& other) noexcept;

    const std::vector<int>& neighboring_processes() const;

    const std::vector<std::vector<int>>& receive_indices() const;

    const std::vector<std::vector<int>>& send_indices() const;

private:
    std::vector<int> _neighbor_processes;
    std::vector<std::vector<int>> _receive_indices;
    std::vector<std::vector<int>> _send_indices;
};

bool entry_in_vector(std::vector<int>&, int value);

template<typename T>
class SparseMatrix;

template<typename T>
inline ExchangePattern
create_exchange_pattern(const SparseMatrix<T>& matrix, const ContiguousParallelPartition& column_partition)
{
    int number_of_processes = column_partition.owner_process(matrix.columns() - 1) + 1;
    std::vector<int> neighboring_processes(number_of_processes, 0);
    std::vector<std::vector<int>> receive_indices(number_of_processes);
    std::vector<std::vector<int>> send_indices(number_of_processes);
    
    int process_size = column_partition.local_size();
    int owner_process = 0;
    int global_index;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for(int i = 0; i < process_size; i++)
    {
        global_index = column_partition.to_global_index(i);
        for(int j = 0; j < matrix.row_nz_size(i); j++)
        {
            if(column_partition.is_owned_by_local_process(matrix.row_nz_index(i, j)) == false &&    //checks if the entry with index i, nnz index j is owned by local process
               matrix.row_nz_entry(i, j) != 0)                              
            {
                owner_process = column_partition.owner_process(matrix.row_nz_index(i, j));           //if not the owner_process of this entry becomes a neighbor (if not already)
                neighboring_processes[owner_process] = 1;

                if(entry_in_vector(receive_indices[owner_process], matrix.row_nz_index(i, j)) == false)   //the corresponding vector entry will be send to the neighbor 
                {                                                              
                    receive_indices[owner_process].push_back(matrix.row_nz_index(i, j));
                    send_indices[owner_process].push_back(global_index);
                }
            }
        }
    }
    
    std::vector<int> neighboring_processes_2;                          //to avoid sorting the neighboring_processes vector all entrys are initialized with 0
    for(unsigned int i = 0; i < neighboring_processes.size(); i++)     //one is assigned to all neighbors during upper for-loop
    {                                                                  //this loop create a vector which only contains neighbors
        if(neighboring_processes[i] == 1)
        {
            neighboring_processes_2.push_back(i);
        }
    }
    std::vector<std::vector<int>> receive_indices_2;
    std::vector<std::vector<int>> send_indices_2;
    for(unsigned int i = 0; i < neighboring_processes_2.size(); i++)
    {
        std::sort(receive_indices[neighboring_processes_2[i]].begin(),             //the receive-and send-indices vectors can contain unsorted elements
                  receive_indices[neighboring_processes_2[i]].end());              //here this is cured
        std::sort(send_indices[neighboring_processes_2[i]].begin(), send_indices[neighboring_processes_2[i]].end());
        receive_indices_2.push_back(receive_indices[neighboring_processes_2[i]]);
        send_indices_2.push_back(send_indices[neighboring_processes_2[i]]);
    }
    ExchangePattern exchange_pattern(neighboring_processes_2, receive_indices_2, send_indices_2);
    return exchange_pattern;
}

#endif