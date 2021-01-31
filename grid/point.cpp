#include "../common/equals.h"

#include "point.h"

Point::Point(scalar_t value)
{
    for(int i = 0; i < space_dimension; i++)
    {
        values[i] = value;
    }
}

Point::Point(scalar_t x, scalar_t y)
{
    assert(this->size() == 2);

    values[0] = x;
    values[1] = y;
}

Point::Point(scalar_t x, scalar_t y, scalar_t z)
{
    assert(this->size() == 3);

    values[0] = x;
    values[1] = y;
    values[2] = z;
}

int Point::size() const
{
    return sizeof(values) / sizeof(values[0]);
}

const scalar_t& Point::operator[](int i) const
{
    assert(i < this->size() && i >= 0);
    return values[i];
}

scalar_t& Point::operator[](int i)
{
    assert(i < this->size() && i >= 0);
    return values[i];
}

bool equals(const Point& lhs, const Point& rhs)
{
    if(lhs.size() != rhs.size())
        return false;
    for(int i = 0; i < lhs.size(); ++i)
    {
        if(!equals(lhs[i], rhs[i])) return false;
    }
    return true;
}