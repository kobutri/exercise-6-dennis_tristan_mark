#include "multiindex.h"
#include <cassert>

MultiIndex::MultiIndex(int value)
{
    for(int i = 0; i < this->size(); ++i)
    {
        data_[i] = value;
    }
}

MultiIndex::MultiIndex(int i, int j)
{
    assert(this->size() == 2);
    data_[0] = i;
    data_[1] = j;
}

MultiIndex::MultiIndex(int i, int j, int k)
{
    assert(this->size() == 3);
    data_[0] = i;
    data_[1] = j;
    data_[2] = k;
}

int MultiIndex::size() const
{
    return sizeof(data_) / sizeof(data_[0]);
}

bool MultiIndex::operator==(const MultiIndex& other) const
{
    if(this->size() != other.size())
        return false;
    for(int i = 0; i < this->size(); ++i)
    {
        if((*this)[i] != other[i])
            return false;
    }
    return true;
}

bool MultiIndex::operator!=(const MultiIndex& other) const
{
    return !((*this) == other);
}

const int& MultiIndex::operator[](int i) const
{
    assert(i >= 0 && i < this->size());
    return data_[i];
}

int& MultiIndex::operator[](int i)
{
    assert(i >= 0 && i < this->size());
    return data_[i];
}

int multi_to_single_index(const MultiIndex& a, const MultiIndex n)
{
    assert(space_dimension == a.size());
    int ret = 0;
    for(int i = 0; i < a.size(); ++i)
    {
        int temp = a[i];
        for(int j = 0; j < i; ++j)
        {
            temp *= n[j];
        }
        ret += temp;
    }
    return ret;
}

MultiIndex single_to_multiindex(int i, const MultiIndex n)
{
    assert(space_dimension > 0 && space_dimension <= 3);
    if constexpr(space_dimension == 1)
    {
        return MultiIndex(i);
    }
    else if constexpr(space_dimension == 2)
    {
        return MultiIndex(i % n[0], (i - i % n[0]) / n[0]);
    }
    else if constexpr(space_dimension == 3)
    {
        int a2 = i / (n[0] * n[1]);
        int a0 = (i - a2 * n[0] * n[1]) % n[0];
        int a1 = (i - a2 * n[0] * n[1] - a0) / n[0];
        return MultiIndex(a0, a1, a2);
    }
}