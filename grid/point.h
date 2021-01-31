#ifndef PMSC_POINT_H
#define PMSC_POINT_H

#include "../common/scalar.h"
#include "../common/space_dimension.h"
#include <assert.h>

class Point
{
public:
    static_assert(space_dimension > 0 && space_dimension <= 3, "Invalid space dimension");

    //! Creates a point with undefined coordinate values.
    Point() = default;

    //! Initializes all coordinate values of the point with `value`.
    explicit Point(scalar_t value);

    //! Initializes 2-dimensional point coordinate values with `x` and `y`.
    //! Works only if `space_dimension` is 2.
    explicit Point(scalar_t x, scalar_t y);

    //! Initializes 3-dimensional point coordinate values with `x`, `y` and `z`.
    //! Works only if `space_dimension` is 3.
    explicit Point(scalar_t x, scalar_t y, scalar_t z);

    //! Returns the size of the point.
    //! This is equal to `space_dimension`.
    int size() const;

    //! Returns the `i`th coordinate value of the point.
    const scalar_t& operator[](int i) const;

    //! Returns a mutable reference to the `i`th coordinate value of the point.
    scalar_t& operator[](int i);

private:
    scalar_t values[space_dimension];
};

bool equals(const Point& lhs, const Point& rhs);

#endif // PMSC_POINT_H