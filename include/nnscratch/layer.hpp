// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#ifndef NNSCRATCH_LAYER_HPP
#define NNSCRATCH_LAYER_HPP

#include <vector>

#include "nnscratch/tensor.hpp"

namespace nn {

/// A trainable parameter paired with its gradient.
///
/// `param` is owned by the layer; the optimizer mutates it in place and uses
/// its address as a stable key for per-parameter state (Momentum/Adam),
/// mirroring the `id(p)` keying in the Python reference.
struct ParamGrad {
    Tensor* param;
    Tensor* grad;
};

/// Common interface for every layer in a model.
class Layer {
public:
    virtual ~Layer() = default;

    /// Compute the output for an input batch, caching whatever is needed for
    /// the matching backward pass.
    virtual Tensor forward(const Tensor& x) = 0;

    /// Given the gradient of the loss w.r.t. this layer's output, return the
    /// gradient w.r.t. its input (and stash parameter gradients internally).
    virtual Tensor backward(const Tensor& grad_out) = 0;

    /// Parameters and their gradients, for the optimizer. Layers without
    /// learnable weights (activations, flatten) return an empty list.
    virtual std::vector<ParamGrad> params_and_grads() { return {}; }
};

}  // namespace nn

#endif  // NNSCRATCH_LAYER_HPP
