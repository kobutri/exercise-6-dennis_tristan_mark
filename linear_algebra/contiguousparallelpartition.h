#ifndef CONTIGUOUSPARALLELPARTITION_H
#define CONTIGUOUSPARALLELPARTITION_H

#include <memory>
#include <mpi.h>
#include <vector>

class ContiguousParallelPartition
{
public:
    explicit ContiguousParallelPartition();
    explicit ContiguousParallelPartition(MPI_Comm communicator, std::vector<int> partition);
    ContiguousParallelPartition(const ContiguousParallelPartition& other);
    ContiguousParallelPartition(ContiguousParallelPartition&& other) noexcept;
    ContiguousParallelPartition& operator=(const ContiguousParallelPartition& other);
    ContiguousParallelPartition& operator=(ContiguousParallelPartition&& other) noexcept;

    MPI_Comm communicator() const;
    int local_size() const;
    bool operator==(const ContiguousParallelPartition& rhs) const;
    bool operator!=(const ContiguousParallelPartition& rhs) const;
    int local_size(int process) const;
    int global_size() const;

    int owner_process(int global_index) const;
    bool is_owned_by_local_process(int global_index) const;
    bool is_owned_by_process(int global_index, int process) const;

    int to_global_index(int local_index) const;
    int to_local_index(int global_index) const;
    int process() const;
    int to_global_index(int local_index, int owner_process) const;

<<<<<<< HEAD

=======
>>>>>>> 66222a7d9f02c12d76db4c9900de89c48eeae539
private:
    MPI_Comm comm_{};
    int comm_size_;
    int process_;
    std::unique_ptr<int[]> partition_{};
};

ContiguousParallelPartition create_partition(MPI_Comm communicator, int local_size);
ContiguousParallelPartition create_uniform_partition(MPI_Comm communicator, int global_size);

#endif