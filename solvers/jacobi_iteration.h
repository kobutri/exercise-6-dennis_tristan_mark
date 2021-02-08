#ifndef JACOBI_ITERATION_H
#define JACOBI_ITERATION_H

#include "../linear_algebra/operations.h"
#include "preconditioner.h"
#include <cassert>

template<typename T>
class JacobiIteration : public Preconditioner<T> {
public:
    virtual void apply(Vector<T> &x, const Vector<T> &b) override {
        assert(Preconditioner<T>::setup_called == true);
        for (int i = 0; i < x.size(); ++i) {
            T diagonalElement = 0;
            for (int j = 0; j < (*(Preconditioner<T>::A_)).row_nz_size(i); ++j) {
                if ((*(Preconditioner<T>::A_)).row_nz_index(i, j) == i) {
                    diagonalElement = (*(Preconditioner<T>::A_)).row_nz_entry(i, j);
                    break;
                }
            }
            if (diagonalElement == 0) {
                continue;
            } else {
                x[i] = b[i] / diagonalElement;
            }
        }
    }
};

#endif