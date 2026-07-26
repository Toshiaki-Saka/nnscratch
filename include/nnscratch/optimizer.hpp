// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#ifndef NNSCRATCH_OPTIMIZER_HPP
#define NNSCRATCH_OPTIMIZER_HPP

#include <unordered_map>
#include <vector>

#include "nnscratch/layer.hpp"
#include "nnscratch/tensor.hpp"

namespace nn {

/// Base class for gradient-based parameter updates.
class Optimizer {
public:
    virtual ~Optimizer() = default;
    /// Apply one update step to every (param, grad) pair.
    virtual void step(const std::vector<ParamGrad>& pgs) = 0;
};

/// Plain stochastic gradient descent: p -= lr * g.
class SGD final : public Optimizer {
public:
    explicit SGD(double lr) : lr_(lr) {}
    void step(const std::vector<ParamGrad>& pgs) override;

private:
    double lr_;
};

/// SGD with momentum: carries a velocity that accelerates down consistent
/// gradient directions.
class Momentum final : public Optimizer {
public:
    explicit Momentum(double lr, double mu = 0.9) : lr_(lr), mu_(mu) {}
    void step(const std::vector<ParamGrad>& pgs) override;

private:
    double lr_, mu_;
    std::unordered_map<const Tensor*, Tensor> v_;
};

/// Adam: per-parameter adaptive learning rates via 1st/2nd gradient moments.
class Adam final : public Optimizer {
public:
    explicit Adam(double lr, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : lr_(lr), b1_(b1), b2_(b2), eps_(eps) {}
    void step(const std::vector<ParamGrad>& pgs) override;

private:
    double lr_, b1_, b2_, eps_;
    long t_ = 0;
    std::unordered_map<const Tensor*, Tensor> m_, v_;
};

}  // namespace nn

#endif  // NNSCRATCH_OPTIMIZER_HPP
