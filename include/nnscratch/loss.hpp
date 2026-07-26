// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#ifndef NNSCRATCH_LOSS_HPP
#define NNSCRATCH_LOSS_HPP

#include "nnscratch/tensor.hpp"

namespace nn {

/// Softmax activation fused with cross-entropy loss.
///
/// Fusing the two is what collapses the output-layer gradient to the famously
/// simple `(prediction - target) / N`, avoiding a separate (and numerically
/// delicate) softmax-Jacobian step.
class SoftmaxCrossEntropy {
public:
    /// Mean cross-entropy of `logits` against one-hot `targets` (both (N, K)).
    /// Caches the softmax probabilities for the backward pass.
    double forward(const Tensor& logits, const Tensor& targets_onehot);

    /// Gradient of the loss w.r.t. the logits: (p - y) / N.
    Tensor backward() const;

private:
    Tensor probs_;
    Tensor targets_;
};

/// One-hot encode integer labels into a (N, num_classes) tensor.
Tensor one_hot(const std::vector<int>& labels, std::size_t num_classes);

}  // namespace nn

#endif  // NNSCRATCH_LOSS_HPP
