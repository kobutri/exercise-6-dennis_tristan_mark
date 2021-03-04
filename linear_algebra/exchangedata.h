#ifndef EXCHANGEDATA_H
#define EXCHANGEDATA_H

#include "exchangepattern.h"
#include "mpi.h"
#include "vector.h"
#include <iostream>
#include <vector>

template<typename T>
class ExchangeData
{
public:
    explicit ExchangeData(const ExchangePattern& exchange_pattern,
                          std::vector<std::vector<T>> data_per_neighboring_process) :
        _exchange_pattern(exchange_pattern),
        _data_per_neighboring_process(data_per_neighboring_process) {}

    const T& get(int owner_rank, int global_index) const
    {
        int i;
        int j;
        for(i = 0; i < static_cast<int>(_data_per_neighboring_process.size()); i++)
        {
            if(_exchange_pattern.neighboring_processes()[i] == owner_rank)
            {
                break;
            }
        }
        for(j = 0; j < static_cast<int>(_data_per_neighboring_process[i].size()); j++)
        {
            if(_exchange_pattern.receive_indices()[i][j] == global_index)
            {
                break;
            }
        }
        return _data_per_neighboring_process[i][j];
    }

private:
    ExchangePattern _exchange_pattern;
    std::vector<std::vector<T>> _data_per_neighboring_process;
};

template<typename T>
MPI_Datatype convert_to_MPI_TYPE();

template<typename T>
void single_exchange(int process, int i, int neighbor_index, const std::vector<std::vector<int>>& receive_indices,
                     const std::vector<std::vector<int>>& send_indices, std::vector<std::vector<T>>& data_per_neighboring_process,
                     std::vector<T>& result, const ContiguousParallelPartition& partition, const Vector<T>& vector)
{
    MPI_Datatype datatype = convert_to_MPI_TYPE<T>();
    MPI_Status status;
    int local_index;
    if(neighbor_index < process)
    {
        result.resize(receive_indices[i].size());
        MPI_Recv(result.data(), static_cast<int>(result.size()), datatype, neighbor_index,
                 neighbor_index, partition.communicator(), &status);
        data_per_neighboring_process[i] = result;
        std::vector<T> send_neighbor;
        for(int j = 0; j < static_cast<int>((send_indices[i]).size()); j++)
        {
            local_index = partition.to_local_index(send_indices[i][j]);
            send_neighbor.push_back(vector[local_index]);
        }
        MPI_Send(send_neighbor.data(), static_cast<int>(send_neighbor.size()), datatype, neighbor_index,
                 process, partition.communicator());
    }
    else
    {
        std::vector<T> send_neighbor;
        for(int j = 0; j < static_cast<int>((send_indices[i]).size()); j++)
        {
            local_index = partition.to_local_index(send_indices[i][j]);
            send_neighbor.push_back(vector[local_index]);
        }
        MPI_Send(send_neighbor.data(), static_cast<int>(send_neighbor.size()), datatype, neighbor_index,
                 process, partition.communicator());
        result.resize(receive_indices[i].size());
        MPI_Recv(result.data(), static_cast<int>(result.size()), datatype, neighbor_index,
                 neighbor_index, partition.communicator(), &status);
        data_per_neighboring_process[i] = result;
    }
    return;
}

template<typename T>
ExchangeData<T> exchange_vector_data(const ExchangePattern& exchange_pattern, const Vector<T>& vector)
{
    ContiguousParallelPartition partition = vector.partition();
    std::vector<std::vector<int>> receive_indices = exchange_pattern.receive_indices();
    std::vector<std::vector<int>> send_indices = exchange_pattern.send_indices();
    std::vector<int> neighboring_processes = exchange_pattern.neighboring_processes();
    int number_of_processes = partition.owner_process(partition.global_size() - 1) + 1;
    int process = partition.process();
    int neighbor_index;
    int neighboring_processes_size = static_cast<int>(neighboring_processes.size());
    std::vector<std::vector<T>> data_per_neighboring_process(neighboring_processes.size());
    std::vector<T> result;
    int i;

    if(number_of_processes % 2 == 1)
    {
        for(int k = 0; k < number_of_processes; k++)
        {
            neighbor_index = (k - process) % number_of_processes;
            neighbor_index = neighbor_index < 0 ? number_of_processes + neighbor_index : neighbor_index;
            for(i = 0; i < neighboring_processes_size; i++)
            {
                if(neighboring_processes[i] == neighbor_index) break;
            }
            if(neighbor_index != process && i < neighboring_processes_size)
            {
                single_exchange(process, i, neighbor_index, receive_indices, send_indices, data_per_neighboring_process, result, partition, vector);
            }
        }
    }
    else
    {
        int idle;
        for(int k = 0; k < number_of_processes - 1; k++)
        {
            idle = ((number_of_processes / 2) * k) % (number_of_processes - 1);
            if(process == number_of_processes - 1)
                neighbor_index = idle;
            else if(process == idle)
                neighbor_index = number_of_processes - 1;
            else
            {
                neighbor_index = (k - process) % (number_of_processes - 1);
                neighbor_index = neighbor_index < 0 ? number_of_processes - 1 + neighbor_index : neighbor_index;
            }
            for(i = 0; i < neighboring_processes_size; i++)
            {
                if(neighboring_processes[i] == neighbor_index) break;
            }
            if(neighbor_index != process && i < neighboring_processes_size)
            {
                single_exchange(process, i, neighbor_index, receive_indices, send_indices, data_per_neighboring_process, result, partition, vector);
            }
        }
    }
    ExchangeData<T> exchange_data(exchange_pattern, data_per_neighboring_process);
    return exchange_data;
}

#endif