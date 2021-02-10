#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "../common/equals.h"
#include "exchangedata.h"
#include "exchangepattern.h"
#include "matrix.h"
#include "vector.h"
#include <cassert>
#include <cmath>
#include <numeric>

template<typename T>
bool equals(const Vector<T>& lhs, const Vector<T>& rhs)
{
    if(lhs.partition() != rhs.partition()) return false;
    bool local_eq = true;
    for(int i = 0; i < lhs.size(); i++)
    {
        if(!equals(lhs[i], rhs[i])) local_eq = false;
    }
    bool global_eq = false;
    MPI_Allreduce(&local_eq, &global_eq, 1, MPI_C_BOOL, MPI_LAND, lhs.partition().communicator());
    return global_eq;
}

template<typename T>
void assign(Vector<T>& lhs, const T& rhs)
{
    for(int i = 0; i < lhs.size(); i++)
    {
        lhs[i] = rhs;
    }
}

template<typename T>
bool equals(const SparseMatrix<T>& lhs, const SparseMatrix<T>& rhs)
{
    if(lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns() || lhs.non_zero_size() != rhs.non_zero_size())
        return false;

    for(int i = 0; i < lhs.rows(); i++)
    {
        if(lhs.row_nz_size(i) != rhs.row_nz_size(i)) return false;
        for(int j = 0; j < lhs.row_nz_size(i); j++)
        {
            if(lhs.row_nz_entry(i, j) != rhs.row_nz_entry(i, j) || lhs.row_nz_index(i, j) != rhs.row_nz_index(i, j))
                return false;
        }
    }
    return true;
}

template<typename T>
void add(Vector<T>& result, const Vector<T>& lhs, const Vector<T>& rhs)
{
    assert(lhs.partition() == rhs.partition());
    if(result.size() != lhs.size())
    {
        result = Vector<T>(lhs.size());
    }
    for(int i = 0; i < lhs.size(); i++)
    {
        result[i] = lhs[i] + rhs[i];
    }
}

template<typename T>
void subtract(Vector<T>& result, const Vector<T>& lhs, const Vector<T>& rhs)
{
    assert(lhs.partition() == rhs.partition());
    if(result.size() != lhs.size())
    {
        result = Vector<T>(lhs.size());
    }
    for(int i = 0; i < lhs.size(); i++)
    {
        result[i] = lhs[i] - rhs[i];
    }
}

template<typename T>
MPI_Datatype convert_to_MPI_TYPE();

template<typename T>
T dot_product(const Vector<T>& lhs, const Vector<T>& rhs)
{
    assert(lhs.partition() == rhs.partition());
    T res = 0.0;
    for(int i = 0; i < lhs.size(); i++)
    {
        res += lhs[i] * rhs[i];
    }
    T global_res = 0;
    MPI_Allreduce(&res, &global_res, 1, convert_to_MPI_TYPE<T>(), MPI_SUM, lhs.partition().communicator());
    return global_res;
}

template<typename T>
void multiply(Vector<T>& result, const Vector<T>& lhs, const T& rhs)
{
    if(result.partition() != lhs.partition())
    {
        result = Vector<T>(lhs.size());
    }
    for(int i = 0; i < lhs.size(); i++)
    {
        result[i] = lhs[i] * rhs;
    }
}

template<typename T>
void multiply(Vector<T>& result, const SparseMatrix<T>& lhs,
              const Vector<T>& rhs)
{
    //assert(lhs.columns() == rhs.());
    if(lhs.row_partition().local_size() == lhs.row_partition().global_size())
    {
        for(int i = 0; i < lhs.rows(); i++)
        {
            result[i] = 0;
            for(int j = 0; j < lhs.row_nz_size(i); j++)
            {
                result[i] += lhs.row_nz_entry(i, j) * rhs[lhs.row_nz_index(i, j)];
            }
        }
    }
    else
    {
        ExchangePattern ex_pattern = lhs.exchange_pattern();
        ExchangeData<T> ex_data = exchange_vector_data(ex_pattern, rhs);
        ContiguousParallelPartition partition = lhs.row_partition();
        if(result.size() != lhs.rows())
        {
            result = Vector<T>(lhs.rows());
        }
        for(int i = 0; i < lhs.rows(); i++)
        {
            result[i] = 0;
            for(int j = 0; j < lhs.row_nz_size(i); j++)
            {
                if(partition.is_owned_by_local_process(lhs.row_nz_index(i, j)))
                {
                    result[i] += lhs.row_nz_entry(i, j) * rhs[partition.to_local_index(lhs.row_nz_index(i, j))];
                }
                else if(lhs.row_nz_entry(i, j) != 0)
                {
                    result[i] += lhs.row_nz_entry(i, j) * ex_data.get(partition.owner_process(lhs.row_nz_index(i, j)), lhs.row_nz_index(i, j));
                }
            }
        }
    }
}

template<typename T>
T norm(const Vector<T>& vec)
{
    T result = 0;
    for(int i = 0; i < vec.size(); i++)
    {
        result += vec[i] * vec[i];
    }
    T global_res = 0;
    MPI_Allreduce(&result, &global_res, 1, convert_to_MPI_TYPE<T>(), MPI_SUM, vec.partition().communicator());
    return std::sqrt(global_res);
}

template<typename T>
void calc_res(Vector<T>& result, const SparseMatrix<T>& A, const Vector<T>& x, const Vector<T>& b)
{
    multiply(result, A, x);
    subtract(result, b, result);
}

#endif