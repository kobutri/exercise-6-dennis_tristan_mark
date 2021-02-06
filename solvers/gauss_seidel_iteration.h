#ifndef GAUSS_SEIDEL_ITERATION_H
#define GAUSS_SEIDEL_ITERATION_H

#include "preconditioner.h"
#include <cassert>

template<typename T>
class GaussSeidelIteration : public Preconditioner<T>
{
public:
    void setup() override
    {
        assert(this->set_operator_called == true);
        receive_ = std::vector<T>(this->A_->row_partition().to_global_index(0));
        send_ = std::vector<T>(this->A_->row_partition().local_size());
        this->setup_called = true;
    }

    void apply(Vector<T>& x, const Vector<T>& b) override
    {
        assert(Preconditioner<T>::setup_called == true);
        ContiguousParallelPartition partition = x.partition();

        // receive data from all vector partitions until this one
        for(int j = 0, k = 0; j < x.partition().to_global_index(0); ++j)
        {
            MPI_Bcast(receive_.data() + k, partition.local_size(j), convert_to_MPI_TYPE<T>(), j, partition.communicator());
            k += partition.local_size(j);
        }
        for(int i = 0; i < x.size(); ++i)
        {
            x[i] = b[i];
            T diag = 0;
            for(int j = 0; j < (*(Preconditioner<T>::A_)).row_nz_size(i); j++)
            {
                if(i == (*(Preconditioner<T>::A_)).row_nz_index(i, j))
                {
                    diag = (*(Preconditioner<T>::A_)).row_nz_entry(i, j);
                    break;
                }
                //                } else if ((*(Preconditioner<T>::A_)).row_nz_index(i, j) < i) {
                //                    x[i] -= x[(*(Preconditioner<T>::A_)).row_nz_index(i, j)] *
                //                            (*(Preconditioner<T>::A_)).row_nz_entry(i, j);
                //                }
            }
            for(int j = 0; j < this->A_->row_nz_size(i); ++j)
            {
                if(this->A_->row_nz_index(i, j) < partition.to_global_index(0))
                {
                    x[i] -= this->A_->row_nz_entry(i, j) * receive_[this->A_->row_nz_index(i, j)];
                }
                else if(this->A_->row_nz_index(i, j) < i)
                {
                    x[i] -= this->A_->row_nz_entry(i, j) * x[this->A_->row_nz_index(i, j)];
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
            send_[i] = x[i];
        }
        MPI_Bcast(send_.data(), static_cast<int>(send_.size()), convert_to_MPI_TYPE<T>(), partition.process(), partition.communicator());
    }

private:
    std::vector<T> receive_;
    std::vector<T> send_;
};

#endif