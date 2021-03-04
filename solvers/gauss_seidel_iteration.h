#ifndef GAUSS_SEIDEL_ITERATION_H
#define GAUSS_SEIDEL_ITERATION_H

#include "../linear_algebra/exchangedata.h"
#include "../linear_algebra/exchangepattern.h"
#include "preconditioner.h"
#include <cassert>
#include <map>
#include <set>

template<typename T>
class GaussSeidelIteration : public Preconditioner<T>
{
public:
    void setup() override
    {
        assert(this->set_operator_called == true);
        this->setup_called = true;
    }

    void apply(Vector<T>& x, const Vector<T>& b) override
    {
        // setup exchange pattern
        if(column_partition_ != b.partition())
        {
            exchange_pattern_ = createExchangePattern(*this->A_, b.partition());
            column_partition_ = b.partition();
            for(int i = 0; i < exchange_pattern_.neighboring_processes().size(); ++i)
            {
                receive_.push_back(std::vector<T>(exchange_pattern_.receive_indices()[i].size()));
                send_.push_back(std::vector<T>(exchange_pattern_.send_indices()[i].size()));
            }
        }

        assert(Preconditioner<T>::setup_called == true);

        // receive data from all vector partitions until this one
        for(int i = 0; i < exchange_pattern_.neighboring_processes().size(); ++i)
        {
            if(!exchange_pattern_.receive_indices()[i].empty())
            {
                MPI_Status status;
                MPI_Recv(receive_[i].data(), static_cast<int>(receive_[i].size()), convert_to_MPI_TYPE<T>(), exchange_pattern_.neighboring_processes()[i], 0, column_partition_.communicator(), &status);
            }
        }
        ExchangeData<T> exchange_data = ExchangeData<T>(exchange_pattern_, receive_);

        for(int i = 0; i < x.size(); ++i)
        {
            x[i] = b[i];
            T diag = 0;
            for(int j = 0; j < this->A_->row_nz_size(i); j++)
            {
                if(this->A_->row_partition().to_global_index(i) == this->A_->row_nz_index(i, j))
                {
                    diag = this->A_->row_nz_entry(i, j);
                    break;
                }
            }
            for(int j = 0; j < this->A_->row_nz_size(i); ++j)
            {
                int column_index = this->A_->row_nz_index(i, j);
                if(column_index < this->A_->row_partition().to_global_index(i))
                {
                    if(this->A_->row_nz_entry(i, j) != 0)
                    {
                        if(!column_partition_.is_owned_by_local_process(column_index))
                        {
                            T val = this->A_->row_nz_entry(i, j);
                            val = val;
                            x[i] -= this->A_->row_nz_entry(i, j) * exchange_data.get(column_partition_.owner_process(column_index), column_index);
                        }
                        else
                        {
                            x[i] -= this->A_->row_nz_entry(i, j) * x[column_partition_.to_local_index(column_index)];
                        }
                    }
                }
            }
            if(diag == 0)
            {
                return;
            }
            else
            {
                x[i] /= diag;
            }
        }

        for(int i = 0; i < exchange_pattern_.neighboring_processes().size(); ++i)
        {
            if(!exchange_pattern_.send_indices()[i].empty())
            {
                for(int j = 0; j < exchange_pattern_.send_indices()[i].size(); ++j)
                {
                    send_[i][j] = x[column_partition_.to_local_index(exchange_pattern_.send_indices()[i][j])];
                }
                MPI_Send(send_[i].data(), static_cast<int>(send_[i].size()), convert_to_MPI_TYPE<T>(), exchange_pattern_.neighboring_processes()[i], 0, column_partition_.communicator());
            }
        }
    }

private:
    std::vector<std::vector<T>> receive_;
    std::vector<std::vector<T>> send_;
    ExchangePattern exchange_pattern_;
    ContiguousParallelPartition column_partition_;

    static ExchangePattern createExchangePattern(const SparseMatrix<T>& matrix, const ContiguousParallelPartition& column_partition)
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
                        if(column_index <= column_partition.to_global_index(i))
                        {
                            processes[column_partition.owner_process(column_index)].first.insert(column_index);
                        }
                        if(column_index >= column_partition.to_global_index(i))
                        {
                            processes[column_partition.owner_process(column_index)].second.insert(column_partition.to_global_index(i));
                        }
                    }
                }
            }
        }

        std::vector<int> neighbors;
        std::vector<std::vector<int>> receive_indices;
        std::vector<std::vector<int>> send_indices;
        for(const auto& [key, value] : processes)
        {
            neighbors.push_back(key);
            receive_indices.push_back(std::vector<int>(value.first.begin(), value.first.end()));
            send_indices.push_back(std::vector<int>(value.second.begin(), value.second.end()));
        }

        return ExchangePattern(neighbors, receive_indices, send_indices);
    }
};

#endif