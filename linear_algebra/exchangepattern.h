#ifndef EXCHANGEPATTERN_H
#define EXCHANGEPATTERN_H

#include "contiguousparallelpartition.h"
#include <algorithm>
#include <iostream>
#include <utility>
#include <map>
#include <set>

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

template<typename T>
class SparseMatrix;


template<typename T>
inline ExchangePattern
create_exchange_pattern(const SparseMatrix<T>& matrix, const ContiguousParallelPartition& column_partition)
{
    std::map<int, std::pair<std::set<int>, std::set<int>>> processes;
    for(int i = 0; i < matrix.rows(); ++i)
    {
        for(int j = 0; j < matrix.row_nz_size(i); ++j)
        {
            int column_index = matrix.row_nz_index(i, j);
            if(matrix.row_nz_entry(i, j) != 0)
            {
                if(!column_partition.is_owned_by_local_process(column_index))
                {
                    processes[column_partition.owner_process(column_index)].first.insert(column_index);
                    processes[column_partition.owner_process(column_index)].second.insert(column_partition.to_global_index(i));
                }
            }
        }
    }

    std::vector<int> neighbors;
    std::vector<std::vector<int>> receive_indices;
    std::vector<std::vector<int>> send_indices;
    for(const auto& [key, value] : processes) {
        neighbors.push_back(key);
        receive_indices.push_back(std::vector<int>(value.first.begin(), value.first.end()));
        send_indices.push_back(std::vector<int>(value.second.begin(), value.second.end()));
    }

    return ExchangePattern(neighbors, receive_indices, send_indices);
}

#endif