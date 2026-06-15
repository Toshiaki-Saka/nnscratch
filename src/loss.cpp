// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nnscratch contributors
#include "nnscratch/loss.hpp"

#include <cmath>

#include "nnscratch/activations.hpp"

namespace nn {

double SoftmaxCrossEntropy::forward(const Tensor& logits, const Tensor& targets_onehot) {
    probs_ = softmax(logits);
    targets_ = targets_onehot;
    const std::size_t n = logits.rows();
    double loss = 0.0;
    const auto& p = probs_.data();
    const auto& y = targets_onehot.data();
    for (std::size_t i = 0; i < p.size(); ++i) {
        loss -= y[i] * std::log(p[i] + 1e-9);
    }
    return loss / static_cast<double>(n);
}

Tensor SoftmaxCrossEntropy::backward() const {
    const double inv_n = 1.0 / static_cast<double>(probs_.rows());
    return inv_n * (probs_ - targets_);
}

Tensor one_hot(const std::vector<int>& labels, std::size_t num_classes) {
    Tensor out(labels.size(), num_classes);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        out(i, static_cast<std::size_t>(labels[i])) = 1.0;
    }
    return out;
}

}  // namespace nn
