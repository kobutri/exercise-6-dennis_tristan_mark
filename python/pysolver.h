#ifndef PYSOLVER_H
#define PYSOLVER_H

class PySolver : public Solver<scalar_t>
{
public:
    using Solver<scalar_t>::Solver;
    using Solver<scalar_t>::last_stop_reason;

    void solve(Vector<scalar_t>& x, const Vector<scalar_t>& b) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            Solver<scalar_t>,
            solve,
            x, b);
    }

    void set_operator(const SparseMatrix<scalar_t>& A) override
    {
        PYBIND11_OVERLOAD(void, Solver<scalar_t>, set_operator, A);
    }

    void setup() override
    {
        PYBIND11_OVERLOAD(void, Solver<scalar_t>, setup);
    }
};
#endif //PMSCEXERCISE6_PYSOLVER_H
