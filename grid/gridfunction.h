#ifndef GRIDFUNCTION_H
#define GRIDFUNCTION_H

#include "../linear_algebra/operations.h"
#include "../linear_algebra/vector.h"
#include "grid.h"
#include <functional>
#include <memory>

template<typename T>
class GridFunction
{
public:
    GridFunction(RegularGrid& grid, T value) :
        grid_(std::make_shared<RegularGrid>(grid)),
        values_(Vector<T>(grid.partition().local_size()))
    {
        assign<T>(values_, value);
    }

    GridFunction(RegularGrid& grid, const std::function<T(const Point&)>& funptr)
    {
        grid_ = std::make_shared<RegularGrid>(grid);
        values_ = Vector<T>(grid.partition().local_size());
        for(int i = 0; i < grid.partition().local_size(); i++)
        {
            Point point = grid.node_coordinates(grid.partition().to_global_index(i));
            values_[i] = funptr(point);
        }
    }

    GridFunction(RegularGrid& grid, const Vector<T>& vector) :
        grid_(std::make_shared<RegularGrid>(grid)),
        values_(vector)
    {
        assert(vector.partition().global_size() == grid.number_of_nodes());
    }

    ~GridFunction() = default;

    const RegularGrid& grid() const
    {
        return *(grid_);
    }

    T value(int node_index) const
    {
        return values_[node_index];
    }

private:
    std::shared_ptr<RegularGrid> grid_;
    Vector<T> values_;
};

#endif