#ifndef PMSC_MULTIINDEX_H
#define PMSC_MULTIINDEX_H

#include "../common/space_dimension.h"
#include "../linear_algebra/contiguousparallelpartition.h"

class MultiIndex
{
public:
    static_assert(space_dimension > 0 && space_dimension <= 3, "Invalid space dimension");

    //! Creates a multi index with undefined coordinate values.
    MultiIndex() = default;

    //! Initializes all coordinate values of the multi index with `value`.
    explicit MultiIndex(int value);

    //! Initializes 2-dimensional multi index coordinate values with `i` and `j`.
    //! Works only if `space_dimension` is 2.
    explicit MultiIndex(int i, int j);

    //! Initializes 3-dimensional multi index coordinate values with `i`, `j` and `k`.
    //! Works only if `space_dimension` is 3.
    explicit MultiIndex(int i, int j, int k);

    //! Returns the size of the multi index.
    //! This is equal to `space_dimension`.
    int size() const;

    //! Tests if two multi indices are equal
    bool operator==(const MultiIndex& other) const;

    //! Tests if two multi indices are inequal
    bool operator!=(const MultiIndex& other) const;

    //! Returns the `i`th coordinate value of the multi index.
    const int& operator[](int i) const;

    //! Returns a mutable reference to the `i`th coordinate value of the multi index.
    int& operator[](int i);

private:
    int data_[space_dimension];
};

int multi_to_single_index(const MultiIndex& a, const MultiIndex n);

MultiIndex single_to_multiindex(int i, const MultiIndex n);

#endif // PMSC_MULTIINDEX_H