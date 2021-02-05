#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "../linear_algebra/matrix.h"

template<typename T>
class Preconditioner {
public:
    Preconditioner() :
        A_(nullptr),
        set_operator_called(false),
        setup_called(false) {}

    virtual ~Preconditioner() = default;

    virtual void set_operator(const SparseMatrix<T> &A) {
        A_ = std::make_unique<SparseMatrix<T>>(A);
        set_operator_called = true;
    }

    virtual void setup() {
        assert(set_operator_called == true);
        setup_called = true;
    }

    virtual void apply(Vector<T> &x, const Vector<T> &b) = 0;

protected:
    std::unique_ptr<SparseMatrix<T>> A_;
    bool set_operator_called;
    bool setup_called;
};

#endif