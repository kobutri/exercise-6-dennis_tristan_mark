#ifndef EXCHANGEPATTERN_H
#define EXCHANGEPATTERN_H

#include "contiguousparallelpartition.h"
#include <iostream>
#include <utility>
#include <algorithm>

template<typename T>
class SparseMatrix;

class ExchangePattern
{
public:
    ExchangePattern();

    ExchangePattern(std::vector<int> neighboring_processes, std::vector<std::vector<int>> receive_indices, std::vector<std::vector<int>> send_indices);

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

#include "matrix.h"

template<typename T>
inline ExchangePattern
create_exchange_pattern(const SparseMatrix<T>& matrix, const ContiguousParallelPartition& column_partition)
{
    int number_of_processes = column_partition.owner_process(matrix.columns() - 1) + 1;
    std::vector<int> neighboring_processes(number_of_processes);
    std::vector<std::vector<int>> receive_indices(number_of_processes);
    std::vector<std::vector<int>> send_indices(number_of_processes);
    //int process = column_partition.process();
    int process_size = column_partition.local_size();
    int owner_process = 0;
    int global_index;

    for(int i = 0; i < process_size; i++)
    {
        global_index = column_partition.to_global_index(i);
        for(int j = 0; j < matrix.row_nz_size(i); j++)
        {
            if(column_partition.is_owned_by_local_process(matrix.row_nz_index(i, j)) == false)
            {
                owner_process = column_partition.owner_process(matrix.row_nz_index(i, j));
                neighboring_processes[owner_process] = 1;

                if(entry_in_vector(receive_indices[owner_process], matrix.row_nz_index(i, j)) == false)
                {
                    receive_indices[owner_process].push_back(matrix.row_nz_index(i, j));
                    send_indices[owner_process].push_back(global_index);
                }
            }
        }
    }
    std::vector<int> neighboring_processes_2;
    for(unsigned int i = 0; i < neighboring_processes.size(); i++)
    {
        if(neighboring_processes[i] == 1)
        {
            neighboring_processes_2.push_back(i);
        }
    }
    std::vector<std::vector<int>> receive_indices_2;
    std::vector<std::vector<int>> send_indices_2;
    for(unsigned int i = 0; i < neighboring_processes_2.size(); i++)
    {
        std::sort(receive_indices[neighboring_processes_2[i]].begin(), receive_indices[neighboring_processes_2[i]].end());
        std::sort(send_indices[neighboring_processes_2[i]].begin(), send_indices[neighboring_processes_2[i]].end());
        receive_indices_2.push_back(receive_indices[neighboring_processes_2[i]]);
        send_indices_2.push_back(send_indices[neighboring_processes_2[i]]);
    }
    ExchangePattern exchange_pattern(neighboring_processes_2, receive_indices_2, send_indices_2);
    return exchange_pattern;
}

#endif