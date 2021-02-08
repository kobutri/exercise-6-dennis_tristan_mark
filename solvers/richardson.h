#ifndef RICHARDSON_H
#define RICHARDSON_H

#include "../linear_algebra/operations.h"
#include "solver.h"

template<typename T>
class RichardsonSolver : public IterativeSolver<T> {
public:
    virtual void setup() override
    {
        IterativeSolver<T>::setup();
        residuum = Vector<T>(this->A_->row_partition());
        result = Vector<T>(this->A_->row_partition());
    }

    virtual void solve(Vector<T> &x, const Vector<T> &b) override {
        assert(Solver<T>::setup_called == true);
        this->A_.get()->initialize_exchange_pattern(this->A_.get()->row_partition());
        int i = 0;

        calc_res(residuum, *(Solver<T>::A_), x, b);
        T r_0 = 1 / norm(residuum);

        if (r_0 == 0) {
            IterativeSolver<T>::last_iterations_ = 0;
            IterativeSolver<T>::last_res_norm_ = 0;
            Solver<T>::stop_reason = StopReason::converged;
            return;
        }
        T r_k = r_0;
        while (IterativeSolver<T>::abs_tolerance < r_k &&
               IterativeSolver<T>::rel_tolerance < r_k * r_0 &&
               i < IterativeSolver<T>::iteration_limit) {

            if (IterativeSolver<T>::preconditioner_ != nullptr)
                (*(IterativeSolver<T>::preconditioner_)).apply(result, residuum);
            else
                result = residuum;

            add(x, x, result);

            calc_res(residuum, *(Solver<T>::A_), x, b);

            r_k = norm(residuum);
            i += 1;
        }

        IterativeSolver<T>::last_iterations_ = i;
        IterativeSolver<T>::last_res_norm_ = r_k;
        if (IterativeSolver<T>::abs_tolerance >= r_k) {
            Solver<T>::stop_reason = StopReason::converged;
            return;
        }
        if (IterativeSolver<T>::rel_tolerance >= r_k * r_0) {
            Solver<T>::stop_reason = StopReason::converged;
            return;
        }
        Solver<T>::stop_reason = StopReason::reached_iteration_limit;
    }

private:
    Vector<T> residuum ;
    Vector<T> result ;
};

#endif