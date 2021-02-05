#ifndef SOLVER_H
#define SOLVER_H

#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"
#include "preconditioner.h"
#include <cassert>
#include <optional>

enum class StopReason {
    unknown,
    reached_iteration_limit,
    converged
};

template<typename T>
class Solver {
public:
    Solver() :
        A_(nullptr),
        set_operator_called(false),
        setup_called(false),
        stop_reason(StopReason::unknown) {}

    virtual ~Solver() = default;

    virtual void set_operator(const SparseMatrix<T> &A) {
        A_ = std::make_unique<SparseMatrix<T>>(A);
        set_operator_called = true;
    }

    virtual void setup() {
        assert(set_operator_called == true);
        setup_called = true;
    }

    virtual void solve(Vector<T> &x, const Vector<T> &b) = 0;

    StopReason last_stop_reason() const {
        return stop_reason;
    }

protected:
    std::unique_ptr<SparseMatrix<T>> A_;
    bool set_operator_called;
    bool setup_called;
    StopReason stop_reason;
};

template<typename T>
class IterativeSolver : public Solver<T> {
public:
    IterativeSolver() :
        Solver<T>::Solver(),
        preconditioner_(nullptr),
        iteration_limit(std::nullopt),
        abs_tolerance(0),
        rel_tolerance(std::nullopt) {}

    virtual void set_operator(const SparseMatrix<T> &A) override {
        Solver<T>::set_operator(A);
        if (preconditioner_ != nullptr) (*(preconditioner_)).set_operator(A);
    }

    virtual void setup() override {
        Solver<T>::setup();
        if (preconditioner_ != nullptr) (*(preconditioner_)).setup();
    }

    void set_preconditioner(std::shared_ptr<Preconditioner<T>> preconditioner) {
        preconditioner_ = std::move(preconditioner);
        if (Solver<T>::set_operator_called == true) (*(preconditioner_)).set_operator(*(Solver<T>::A_));
    }

    void max_iterations(std::optional<int> value) {
        iteration_limit = value;
    }

    std::optional<int> max_iterations() const {
        return iteration_limit;
    }

    void absolute_tolerance(T value) {
        abs_tolerance = value;
    }

    T absolute_tolerance() const {
        return abs_tolerance;
    }

    void relative_tolerance(std::optional<T> value) {
        rel_tolerance = value;
    }

    std::optional<T> relative_tolerance() const {
        return rel_tolerance;
    }

    int last_iterations() const {
        assert(Solver<T>::stop_reason != StopReason::unknown);
        return last_iterations_;
    }

    T last_residual_norm() const {
        assert(Solver<T>::stop_reason != StopReason::unknown);
        return last_res_norm_;
    }

protected:
    std::shared_ptr<Preconditioner<T>> preconditioner_;
    std::optional<int> iteration_limit;
    T abs_tolerance;
    std::optional<T> rel_tolerance;
    int last_iterations_;
    T last_res_norm_;
};

#endif