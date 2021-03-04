#ifndef MATRIX_H
#define MATRIX_H

#include "contiguousparallelpartition.h"
#include "exchangepattern.h"
#include <cassert>
#include <functional>
#include <iomanip>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <vector>

template<typename T>
class SparseMatrix
{
public:
    using triplet_type = std::tuple<int, int, T>;

    SparseMatrix() :
        rows_(0), columns_(0), nnz_(0), A_(nullptr), JA_(nullptr), IA_(nullptr), row_partition_(),
        initialize_exchange_pattern_called(false), exchange_pattern_() {}

    SparseMatrix(const SparseMatrix& other) :
        rows_(other.rows_), columns_(other.columns_), nnz_(other.nnz_), row_partition_(other.row_partition_),
        initialize_exchange_pattern_called(other.initialize_exchange_pattern_called)
    {
        exchange_pattern_ = other.exchange_pattern_;
        //std::copy(other.A_.get(), other.A_.get() + nnz_, A_.get());

        A_ = std::make_unique<T[]>(other.nnz_);
        std::copy(other.A_.get(), other.A_.get() + nnz_, A_.get());

        JA_ = std::make_unique<int[]>(other.nnz_);
        std::copy(other.JA_.get(), other.JA_.get() + nnz_, JA_.get());

        IA_ = std::make_unique<int[]>(other.rows() + 1);
        std::copy(other.IA_.get(), other.IA_.get() + rows_ + 1, IA_.get());
    }

    SparseMatrix(SparseMatrix&& other) noexcept :
        rows_(other.rows()),
        columns_(other.columns()), nnz_(other.nnz_), A_(std::move(other.A_)), JA_(std::move(other.JA_)),
        IA_(std::move(other.IA_)), row_partition_(std::move(other.row_partition_)),
        exchange_pattern_(other.exchange_pattern_),
        initialize_exchange_pattern_called(other.initialize_exchange_pattern_called) {}

    SparseMatrix& operator=(const SparseMatrix& other)
    {
        rows_ = other.rows_;
        columns_ = other.columns_;
        nnz_ = other.nnz_;
        row_partition_ = other.row_partition_;
        initialize_exchange_pattern_called = other.initialize_exchange_pattern_called;
        exchange_pattern_ = other.exchange_pattern_;

        A_ = std::make_unique<T[]>(other.nnz_);
        std::copy(other.A_.get(), other.A_.get() + nnz_, A_.get());

        JA_ = std::make_unique<int[]>(other.nnz_);
        std::copy(other.JA_.get(), other.JA_.get() + nnz_, JA_.get());

        IA_ = std::make_unique<int[]>(other.rows() + 1);
        std::copy(other.IA_.get(), other.IA_.get() + rows_ + 1, IA_.get());

        return *this;
    }

    SparseMatrix& operator=(SparseMatrix&& other) noexcept
    {
        rows_ = other.rows_;
        columns_ = other.columns_;
        nnz_ = other.nnz_;
        initialize_exchange_pattern_called = other.initialize_exchange_pattern_called;
        exchange_pattern_ = other.exchange_pattern_;

        A_ = std::move(other.A_);
        JA_ = std::move(other.JA_);
        IA_ = std::move(other.IA_);
        row_partition_ = std::move(other.row_partition_);

        return *this;
    }

    explicit SparseMatrix(int rows, int global_columns, const std::function<int(int)>& nz_per_row) :

        rows_(rows), columns_(global_columns), row_partition_(MPI_COMM_SELF, {0, rows}),
        initialize_exchange_pattern_called(false), exchange_pattern_()
    {
        assert(columns_ >= 0);
        assert(rows >= 0);
        IA_ = std::make_unique<int[]>(rows_ + 1);
        IA_[0] = 0;
        nnz_ = 0;
        for(int i = 0; i <= rows - 1; i++)
        {
            nnz_ += nz_per_row(row_partition_.to_global_index(i));
            IA_[i + 1] = nnz_;
        }
        A_ = std::make_unique<T[]>(nnz_);
        JA_ = std::make_unique<int[]>(nnz_);
    }

    SparseMatrix(int rows, int columns, const std::vector<triplet_type>& entries) :
        rows_(rows), columns_(columns), nnz_(static_cast<int>(entries.size())),
        row_partition_(MPI_COMM_SELF, {0, rows}),
        initialize_exchange_pattern_called(false), exchange_pattern_()
    {
        assert(rows >= 0);
        assert(columns >= 0);
        A_ = std::make_unique<T[]>(nnz_);
        JA_ = std::make_unique<int[]>(nnz_);
        IA_ = std::make_unique<int[]>(rows_ + 1);
        for(int i = 0; i < nnz_; i++)
        {
            A_[i] = std::get<2>(entries[i]);
            JA_[i] = std::get<1>(entries[i]);
        }

        for(int i = 0, j = 0; i < rows_ + 1; i++)
        {
            IA_[i] = j;
            int row = std::get<0>(entries[std::min(j, nnz_ - 1)]);
            if(i < row) continue;
            while(j < nnz_ && row == std::get<0>(entries[j]))
            {
                j++;
            }
        }
    }

    explicit SparseMatrix(const ContiguousParallelPartition& row_partition, int global_columns,
                          std::function<int(int)> nz_per_row) :
        row_partition_(row_partition),
        columns_(global_columns), rows_(row_partition.local_size()),
        initialize_exchange_pattern_called(false), exchange_pattern_()
    {
        assert(columns_ >= 0);
        IA_ = std::make_unique<int[]>(rows_ + 1);
        IA_[0] = 0;
        nnz_ = 0;
        for(int i = 0; i <= rows_ - 1; i++)
        {
            nnz_ += nz_per_row(row_partition_.to_global_index(i));
            IA_[i + 1] = nnz_;
        }
        A_ = std::make_unique<T[]>(nnz_);
        JA_ = std::make_unique<int[]>(nnz_);
    }

    explicit SparseMatrix(ContiguousParallelPartition row_partition, int global_columns,
                          const std::vector<triplet_type>& entries) :
        row_partition_(row_partition),
        columns_(global_columns), rows_(row_partition.local_size()),
        nnz_(static_cast<int>(entries.size())),
        initialize_exchange_pattern_called(false), exchange_pattern_()
    {
        assert(global_columns >= 0);
        //assert(rows_ * global_columns >= static_cast<int>entries.size());
        A_ = std::make_unique<T[]>(nnz_);
        JA_ = std::make_unique<int[]>(nnz_);
        IA_ = std::make_unique<int[]>(rows_ + 1);
        for(int i = 0; i < nnz_; i++)
        {
            A_[i] = std::get<2>(entries[i]);
            JA_[i] = std::get<1>(entries[i]);
        }
        for(int i = 0, j = 0; i < rows_ + 1; i++)
        {
            IA_[i] = j;
            int row = std::get<0>(entries[std::min(j, nnz_ - 1)]);
            if(i < row) continue;
            while(j < nnz_ && row == std::get<0>(entries[j]))
            {
                j++;
            }
        }
    }

    explicit SparseMatrix(MPI_Comm communicator, int local_rows, int global_columns,
                          std::function<int(int)> nz_per_row) :
        row_partition_(create_partition(communicator, local_rows)),
        columns_(global_columns),
        rows_(local_rows), initialize_exchange_pattern_called(false), exchange_pattern_()
    {
        assert(global_columns >= 0);
        assert(rows_ >= 0);
        IA_ = std::make_unique<int[]>(local_rows + 1);
        IA_[0] = 0;
        nnz_ = 0;
        for(int i = 0; i < rows_; i++)
        {
            nnz_ += nz_per_row(row_partition_.to_global_index(i));
            IA_[i + 1] = nnz_;
        }
        A_ = std::make_unique<T[]>(nnz_);
        JA_ = std::make_unique<int[]>(nnz_);
    }

    explicit SparseMatrix(MPI_Comm communicator, int local_rows, int global_columns,
                          const std::vector<triplet_type>& entries) :
        row_partition_(create_partition(communicator, local_rows)),
        columns_(global_columns), rows_(local_rows),
        nnz_(static_cast<int>(entries.size())), initialize_exchange_pattern_called(false), exchange_pattern_()
    {
        assert(global_columns >= 0);
        //assert(rows_ * global_columns >= static_cast<int>entries.size());
        A_ = std::make_unique<T[]>(nnz_);
        JA_ = std::make_unique<int[]>(nnz_);
        IA_ = std::make_unique<int[]>(rows_ + 1);
        for(int i = 0; i < nnz_; i++)
        {
            A_[i] = std::get<2>(entries[i]);
            JA_[i] = std::get<1>(entries[i]);
        }

        for(int i = 0, j = 0; i < rows_ + 1; i++)
        {
            IA_[i] = j;
            int row = std::get<0>(entries[std::min(j, nnz_ - 1)]);
            if(i < row) continue;
            while(j < nnz_ && row == std::get<0>(entries[j]))
            {
                j++;
            }
        }
    }

    const ContiguousParallelPartition& row_partition() const
    {
        return row_partition_;
    }

    int rows() const
    {
        return rows_;
    }

    int columns() const
    {
        return columns_;
    }

    int non_zero_size() const
    {
        return nnz_;
    }

    int row_nz_size(int r) const
    {
        assert(r >= 0);
        assert(r < rows_);
        return IA_[r + 1] - IA_[r];
    }

    const int& row_nz_index(int r, int nz_i) const
    {
        assert(r >= 0);
        assert(r < rows_);
        assert(nz_i >= 0);
        assert(nz_i < row_nz_size(r));
        return JA_[IA_[r] + nz_i];
    }

    int& row_nz_index(int r, int nz_i)
    {
        assert(r >= 0);
        assert(r < rows_);
        assert(nz_i >= 0);
        assert(nz_i < row_nz_size(r));
        return JA_[IA_[r] + nz_i];
    }

    const T& row_nz_entry(int r, int nz_i) const
    {
        assert(r >= 0);
        assert(r < rows_);
        assert(nz_i >= 0);
        assert(nz_i < row_nz_size(r));
        return A_[IA_[r] + nz_i];
    }

    T& row_nz_entry(int r, int nz_i)
    {
        assert(r >= 0);
        assert(r < rows_);
        assert(nz_i >= 0);
        assert(nz_i < row_nz_size(r));
        return A_[IA_[r] + nz_i];
    }

    void initialize_exchange_pattern(const ContiguousParallelPartition& column_partition)
    {
        exchange_pattern_ = create_exchange_pattern(*this, column_partition);
        initialize_exchange_pattern_called = true;
    }

    const ExchangePattern& exchange_pattern() const
    {
        assert(initialize_exchange_pattern_called == true);
        return exchange_pattern_;
    }

    void print() const
    {
        int rank;
        MPI_Comm_rank(row_partition_.communicator(), &rank);
        int size;
        MPI_Comm_size(row_partition_.communicator(), &size);
        std::vector<std::vector<T>> rows;
        for(int i = 0; i < rows_; ++i)
        {
            std::vector<T> row(columns(), 0.);
            for(int j = 0; j < row_nz_size(i); ++j)
            {
                row[row_nz_index(i, j)] = row_nz_entry(i, j);
            }
            rows.push_back(row);
        }
        std::cout << std::fixed << std::setprecision(2);
        for(int i = 0; i < size; ++i)
        {
            if(rank == i)
            {
                for(int j = 0; j < rows_; ++j)
                {
                    for(int k = 0; k < columns(); ++k)
                    {
                        std::cout << rows[j][k] << "\t";
                    }
                    std::cout << std::endl;
                }
            }
            MPI_Barrier(row_partition().communicator());
        }
    }

private:
    int rows_;
    int columns_;
    int nnz_;
    std::unique_ptr<T[]> A_;
    std::unique_ptr<int[]> JA_;
    std::unique_ptr<int[]> IA_;
    ContiguousParallelPartition row_partition_;
    ExchangePattern exchange_pattern_;
    bool initialize_exchange_pattern_called;
};

#endif