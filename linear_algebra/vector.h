#ifndef VECTOR_H
#define VECTOR_H

#include "contiguousparallelpartition.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>

template<typename T>
class Vector
{
public:
    Vector() :
        partition_(),
        values_(nullptr) {}

    Vector(Vector&& other) noexcept :
        partition_(std::move(other.partition_)),
        values_(std::move(other.values_)) {}

    Vector(const Vector& ot) :
        partition_(ot.partition_)
    {
        values_ = std::make_unique<T[]>(partition_.local_size());
        std::copy(ot.values_.get(), ot.values_.get() + partition_.local_size(), values_.get());
    }

    Vector(int size) :
        partition_(MPI_COMM_SELF, {0, size})
    {
        assert(size >= 0);
        values_ = std::make_unique<T[]>(partition_.local_size());
    }

    Vector(std::initializer_list<T> init) :
        partition_(MPI_COMM_SELF, {0, static_cast<int>(init.size())})
    {
        values_ = std::make_unique<T[]>(partition_.local_size());
        std::copy(init.begin(), init.end(), values_.get());
    }

    Vector(const std::vector<T>& values) :
        partition_(MPI_COMM_SELF, {0, static_cast<int>(values.size())})
    {
        values_ = std::make_unique<T[]>(partition_.local_size());
        std::copy(values.begin(), values.end(), values_.get());
    }

    explicit Vector(const ContiguousParallelPartition& partition) :
        partition_(partition)
    {
        values_ = std::make_unique<T[]>(partition.local_size());
    }

    explicit Vector(MPI_Comm communicator, int local_size) :
        partition_(create_partition(communicator, local_size)),
        values_(std::make_unique<T[]>(local_size)) {}

    explicit Vector(MPI_Comm communicator, std::initializer_list<T> init) :
        partition_(create_partition(communicator, static_cast<int>(init.size())))
    {
        values_ = std::make_unique<T[]>(init.size());
        std::copy(init.begin(), init.end(), values_.get());
    }

    explicit Vector(MPI_Comm communicator, const std::vector<T>& values) :
        partition_(create_partition(communicator, static_cast<int>(values.size())))
    {
        values_ = std::make_unique<T[]>(values.size());
        std::copy(values.begin(), values.end(), values_.get());
    }

    Vector& operator=(const Vector& ot)
    {
        partition_ = ot.partition_;
        values_ = std::make_unique<T[]>(partition_.local_size());
        std::copy(ot.values_.get(), ot.values_.get() + partition_.local_size(), values_.get());
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept
    {
        partition_ = std::move(other.partition_);
        values_ = std::move(other.values_);
        return *this;
    }

    const T& operator[](int i) const
    {
        assert(i >= 0 && i < size());
        return values_[i];
    }

    T& operator[](int i)
    {
        assert(i >= 0 && i <= size());
        return values_[i];
    }

    int size() const
    {
        return partition_.local_size();
    }

    const ContiguousParallelPartition& partition() const
    {
        return partition_;
    }

    void print() const
    {
        int rank;
        MPI_Comm_rank(partition().communicator(), &rank);
        int size_;
        MPI_Comm_size(partition().communicator(), &size_);
        std::cout << std::fixed << std::setprecision(2);
        for(int i = 0; i < size_; ++i)
        {
            if(i == rank)
            {
                for(int j = 0; j < size(); ++j)
                {
                    std::cout << operator[](j) << std::endl;
                }
            }
            MPI_Barrier(partition().communicator());
        }
    }

private:
    ContiguousParallelPartition partition_;
    std::unique_ptr<T[]> values_;
};

#endif