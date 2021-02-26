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
ExchangeData<T> exchange_vector_data(const ExchangePattern& exchange_pattern, const Vector<T>& vector)
{
    ContiguousParallelPartition partition = vector.partition();
    std::vector<std::vector<int>> receive_indices = exchange_pattern.receive_indices();
    std::vector<std::vector<int>> send_indices = exchange_pattern.send_indices();
    std::vector<int> neighboring_processes = exchange_pattern.neighboring_processes();
    int process = partition.process();
    int local_index;
    MPI_Status status;
    int neighboring_processes_size = static_cast<int>(neighboring_processes.size());
    std::vector<std::vector<T>> data_per_neighboring_process(neighboring_processes.size());
    MPI_Datatype datatype = convert_to_MPI_TYPE<T>();
    std::vector<T> result;
    int k = 0;

    while(k < neighboring_processes_size)
    {
        if(neighboring_processes[k] < process)
        {
            result.resize(receive_indices[k].size());
            MPI_Recv(result.data(), static_cast<int>(result.size()), datatype, neighboring_processes[k],
                     neighboring_processes[k], partition.communicator(), &status);
            data_per_neighboring_process[k] = result;
            k += 1;
        }
        else
        {
            break;
        }
    }
    for(int i = 0; i < static_cast<int>(send_indices.size()); i++)
    {
        std::vector<T> send_neighbor;

        for(int j = 0; j < static_cast<int>((send_indices[i]).size()); j++)
        {
            local_index = partition.to_local_index(send_indices[i][j]);
            send_neighbor.push_back(vector[local_index]);
        }
        if(neighboring_processes[i] == process)
        {
            data_per_neighboring_process[i] = send_neighbor;
            k += 1;
        }
        else
        {
            MPI_Send(send_neighbor.data(), static_cast<int>(send_neighbor.size()), datatype, neighboring_processes[i],
                     process, partition.communicator());
        }
    }
    while(k < neighboring_processes_size)
    {
        result.resize((receive_indices[k]).size());
        MPI_Recv(result.data(), static_cast<int>(result.size()), datatype, neighboring_processes[k],
                 neighboring_processes[k], partition.communicator(), &status);
        data_per_neighboring_process[k] = result;

        k += 1;
    }
    ExchangeData<T> exchange_data(exchange_pattern, data_per_neighboring_process);
    return exchange_data;
}

#endif