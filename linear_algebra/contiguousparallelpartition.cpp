#include "contiguousparallelpartition.h"
#include <cassert>

ContiguousParallelPartition::ContiguousParallelPartition() :
    comm_size_(0), process_(0), comm_(MPI_COMM_SELF) {}

ContiguousParallelPartition::ContiguousParallelPartition(MPI_Comm communicator, std::vector<int> partition) :
    comm_(communicator)
{
    int process = 0;
    MPI_Comm_rank(communicator, &process);
    process_ = process;

    int size = 0;
    MPI_Comm_size(communicator, &size);
    comm_size_ = size;

    assert(comm_size_ == partition.size() - 1);
    partition_ = std::make_unique<int[]>(partition.size());
    std::copy(partition.begin(), partition.end(), partition_.get());
}

MPI_Comm ContiguousParallelPartition::communicator() const
{
    return comm_;
}

int ContiguousParallelPartition::local_size() const
{
    return partition_[process_ + 1] - partition_[process_];
}

int ContiguousParallelPartition::local_size(int process) const
{
    assert(process < comm_size_);
    return partition_[process + 1] - partition_[process];
}

int ContiguousParallelPartition::global_size() const
{
    return partition_[comm_size_];
}

int ContiguousParallelPartition::owner_process(int global_index) const
{
    assert(global_index < global_size());
    for(int i = 0; i < comm_size_; ++i)
    {
        if(partition_[i] <= global_index && partition_[i + 1] > global_index)
        {
            return i;
        }
    }
    return comm_size_ - 1;
}

bool ContiguousParallelPartition::is_owned_by_local_process(int global_index) const
{
    assert(global_index < global_size());
    return partition_[process_] <= global_index && partition_[process_ + 1] > global_index;
}

bool ContiguousParallelPartition::is_owned_by_process(int global_index, int process) const
{
    assert(process < comm_size_);
    assert(global_index < global_size());
    return partition_[process] <= global_index && partition_[process + 1] > global_index;
}

int ContiguousParallelPartition::to_global_index(int local_index) const
{
    assert(local_index < local_size());
    return partition_[process_] + local_index;
}

int ContiguousParallelPartition::to_local_index(int global_index) const
{
    assert(is_owned_by_local_process(global_index));
    return global_index - partition_[process_];
}

ContiguousParallelPartition::ContiguousParallelPartition(const ContiguousParallelPartition& other) :
    comm_(other.comm_),
    comm_size_(other.comm_size_),
    process_(other.process_)
{
    partition_ = std::make_unique<int[]>(other.comm_size_ + 1);
    std::copy(other.partition_.get(), other.partition_.get() + other.comm_size_ + 1, partition_.get());
}

ContiguousParallelPartition::ContiguousParallelPartition(ContiguousParallelPartition&& other) noexcept :
    comm_size_(other.comm_size_),
    process_(other.process_),
    comm_(other.comm_),
    partition_(std::move(other.partition_)) {}

ContiguousParallelPartition& ContiguousParallelPartition::operator=(const ContiguousParallelPartition& other)
{
    comm_ = other.comm_;
    comm_size_ = other.comm_size_;
    process_ = other.process_;
    partition_ = std::make_unique<int[]>(other.comm_size_ + 1);
    std::copy(other.partition_.get(), other.partition_.get() + other.comm_size_ + 1, partition_.get());
    return *this;
}

ContiguousParallelPartition& ContiguousParallelPartition::operator=(ContiguousParallelPartition&& other) noexcept
{
    comm_ = other.comm_;
    comm_size_ = other.comm_size_;
    process_ = other.process_;
    partition_ = std::move(other.partition_);
    return *this;
}

bool ContiguousParallelPartition::operator==(const ContiguousParallelPartition& rhs) const
{
    if(comm_ != rhs.comm_ || comm_size_ != rhs.comm_size_ || process_ != rhs.process_)
        return false;

    for(int i = 0; i < comm_size_ + 1; ++i)
    {
        if(partition_[i] != rhs.partition_[i])
            return false;
    }
    return true;
}

bool ContiguousParallelPartition::operator!=(const ContiguousParallelPartition& rhs) const
{
    return !(rhs == *this);
}

int ContiguousParallelPartition::process() const
{
    int rank = -1;
    MPI_Comm_rank(communicator(), &rank);
    assert(rank >= 0);
    return rank;
}

int ContiguousParallelPartition::to_global_index(int local_index, int owner_process) const
{
    assert(local_index < local_size(owner_process));
    return partition_[owner_process] + local_index;
}

ContiguousParallelPartition create_partition(MPI_Comm communicator, int local_size)
{
    int local_start = 0;
    MPI_Exscan(&local_size, &local_start, 1, MPI_INT, MPI_SUM, communicator);
    int size = 0;
    MPI_Comm_size(communicator, &size);
    std::vector<int> partition(size);
    MPI_Allgather(&local_start, 1, MPI_INT, partition.data(), 1, MPI_INT, communicator);
    int global_size = 0;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, communicator);
    partition.push_back(global_size);
    return ContiguousParallelPartition(communicator, partition);
}

ContiguousParallelPartition create_uniform_partition(MPI_Comm communicator, int global_size)
{
    int size = 0;
    MPI_Comm_size(communicator, &size);
    std::vector<int> partition(size);
    int local_size = global_size / size;
    for(int i = 0; i < size; ++i)
    {
        partition[i] = i * local_size;
    }
    partition.push_back(global_size);
    return ContiguousParallelPartition(communicator, partition);
}
