#ifndef PMSCEXERCISE6_PYPRECONDITIONER_H
#define PMSCEXERCISE6_PYPRECONDITIONER_H

class PyPreconditioner : public Preconditioner<scalar_t>
{
public:
    using Preconditioner<scalar_t>::Preconditioner;
    using Preconditioner<scalar_t>::set_operator;
    using Preconditioner<scalar_t>::setup;

    void apply(Vector<scalar_t>& x, const Vector<scalar_t>& b) override
    {
        PYBIND11_OVERLOAD_PURE(void, Preconditioner<scalar_t>, apply, x, b);
    }

    void set_operator(std::shared_ptr<SparseMatrix<scalar_t>> A) override
    {
        PYBIND11_OVERLOAD(void, Preconditioner<scalar_t>, set_operator, A);
    }
};

#endif //PMSCEXERCISE6_PYPRECONDITIONER_H
