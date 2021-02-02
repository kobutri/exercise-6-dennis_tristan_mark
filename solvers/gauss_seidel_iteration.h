#ifndef GAUSS_SEIDEL_ITERATION_H
#define GAUSS_SEIDEL_ITERATION_H

#include "../linear_algebra/operations.h"
#include "preconditioner.h"
#include <cassert>

template<typename T>
class GaussSeidelIteration : public Preconditioner<T> {
public:
    virtual void apply(Vector<T> &x, const Vector<T> &b) override {
        assert(Preconditioner<T>::setup_called == true);
        for (int i = 0; i < x.size(); ++i) {
            x[i] = b[i];
            T diag = 0;
            for (int j = 0; j < (*(Preconditioner<T>::A_)).row_nz_size(i); j++) {
                if (i == (*(Preconditioner<T>::A_)).row_nz_index(i, j)) {
                    diag = (*(Preconditioner<T>::A_)).row_nz_entry(i, j);
                } else if ((*(Preconditioner<T>::A_)).row_nz_index(i, j) < i) {
                    x[i] -= x[(*(Preconditioner<T>::A_)).row_nz_index(i, j)] *
                            (*(Preconditioner<T>::A_)).row_nz_entry(i, j);
                }
            }
            if (diag == 0) {
                return;
            } else {
                x[i] /= diag;
            }
        }
    }
};

#endif