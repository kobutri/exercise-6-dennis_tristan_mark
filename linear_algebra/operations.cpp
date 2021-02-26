#include "operations.h"

template<>
MPI_Datatype convert_to_MPI_TYPE<int>()
{
    return MPI_INT;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<long>()
{
    return MPI_LONG;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<short>()
{
    return MPI_SHORT;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<unsigned short>()
{
    return MPI_UNSIGNED_SHORT;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<unsigned int>()
{
    return MPI_UNSIGNED;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<unsigned long>()
{
    return MPI_UNSIGNED_LONG;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<float>()
{
    return MPI_FLOAT;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<double>()
{
    return MPI_DOUBLE;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<long double>()
{
    return MPI_LONG_DOUBLE;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<char>()
{
    return MPI_CHAR;
}

template<>
MPI_Datatype convert_to_MPI_TYPE<unsigned char>()
{
    return MPI_BYTE;
}