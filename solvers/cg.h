#ifndef CG_H
#define CG_H

#include "../linear_algebra/operations.h"
#include "solver.h"
#include <cassert>

template<typename T>
class CgSolver : public IterativeSolver<T> {
public:
    virtual void setup() override
    {
        IterativeSolver<T>::setup();
        res = Vector<T>(this->A_->row_partition());
        p = Vector<T>(this->A_->row_partition());
        q = Vector<T>(this->A_->row_partition());
        z = Vector<T>(this->A_->row_partition());
    }

    virtual void solve(Vector<T> &x, const Vector<T> &b) override {
        assert(Solver<T>::setup_called == true);
        T norm_r0, norm_res, a, beta, sc_product_res, sc_product_res2;
        int k = 0;
        calc_res(res, *(Solver<T>::A_), x, b);
        norm_r0 = norm(res);
        norm_res = norm_r0;
        if (norm_r0 == 0) // mayber better < 1e-15 or something
        {
            IterativeSolver<T>::last_iterations_ = 0;
            IterativeSolver<T>::last_res_norm_ = 0;
            Solver<T>::stop_reason = StopReason::converged;
            return;
        }
        norm_r0 = 1 / norm_r0; // avoid division in each itteration
        if (IterativeSolver<T>::preconditioner_ != nullptr)
            (*(IterativeSolver<T>::preconditioner_)).apply(z, res);
        else
            z = res;
        p = z;
        sc_product_res2 = dot_product(z, res);
        while (k < IterativeSolver<T>::iteration_limit &&
               norm_res * norm_r0 > IterativeSolver<T>::rel_tolerance &&
               norm_res > IterativeSolver<T>::abs_tolerance) {
            sc_product_res = sc_product_res2;
            multiply(q, *(Solver<T>::A_), p);
            a = sc_product_res / dot_product(q, p);
            multiply(p, p, a);
            multiply(q, q, a);
            add(x, x, p);
            subtract(res, res, q);
            if (IterativeSolver<T>::preconditioner_ != nullptr)
                (*(IterativeSolver<T>::preconditioner_)).apply(z, res);
            else
                z = res;
            sc_product_res2 = dot_product(z, res);
            beta = sc_product_res2 / sc_product_res;
            multiply(p, p, (beta / a));
            add(p, z, p);
            norm_res = norm(res);
            k += 1;
        }
        IterativeSolver<T>::last_iterations_ = k;
        IterativeSolver<T>::last_res_norm_ = norm_res;
        if (IterativeSolver<T>::abs_tolerance >= norm_res) {
            Solver<T>::stop_reason = StopReason::converged;
            return;
        }
        if (IterativeSolver<T>::rel_tolerance >= norm_res * norm_r0) {
            Solver<T>::stop_reason = StopReason::converged;
            return;
        }
        Solver<T>::stop_reason = StopReason::reached_iteration_limit;
    }
private:
    Vector<T> res;
    Vector<T> p;
    Vector<T> q;
    Vector<T> z;

};

#endif