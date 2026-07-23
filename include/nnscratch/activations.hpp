// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
#ifndef NNSCRATCH_ACTIVATIONS_HPP
#define NNSCRATCH_ACTIVATIONS_HPP

#include "nnscratch/layer.hpp"
#include "nnscratch/tensor.hpp"

namespace nn {

/// Row-wise softmax of a (N, K) logit tensor into a probability distribution.
/// Subtracts the per-row max for numerical stability (mathematically a no-op).
Tensor softmax(const Tensor& logits);

/// ReLU: max(0, x). The standard hidden-layer non-linearity.
class ReLU final : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

private:
    Tensor mask_;  // 1 where x > 0, else 0
};

/// Hyperbolic tangent activation.
class Tanh final : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

private:
    Tensor out_;
};

/// Logistic sigmoid activation.
class Sigmoid final : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

private:
    Tensor out_;
};

}  // namespace nn

#endif  // NNSCRATCH_ACTIVATIONS_HPP
