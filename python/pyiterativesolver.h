#ifndef PYITERATIVESOLVER_H
#define PYITERATIVESOLVER_H

class PyIterativeSolver : public IterativeSolver<scalar_t>
{
public:
    using IterativeSolver<scalar_t>::IterativeSolver;
    using IterativeSolver<scalar_t>::set_preconditioner;
    using IterativeSolver<scalar_t>::max_iterations;
    using IterativeSolver<scalar_t>::absolute_tolerance;
    using IterativeSolver<scalar_t>::relative_tolerance;
    using IterativeSolver<scalar_t>::last_iterations;
    using IterativeSolver<scalar_t>::last_residual_norm;
    using IterativeSolver<scalar_t>::last_stop_reason;

    void set_operator(const SparseMatrix<scalar_t>& A) override
    {
        PYBIND11_OVERLOAD(void, IterativeSolver<scalar_t>, set_operator, A);
    }

    void setup() override
    {
        PYBIND11_OVERLOAD(void, IterativeSolver<scalar_t>, setup);
    }

    void solve(Vector<scalar_t>& x, const Vector<scalar_t>& b) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            Solver<scalar_t>,
            solve,
            x, b);
    }
};

#endif //PMSCEXERCISE6_PYITERATIVESOLVER_H
