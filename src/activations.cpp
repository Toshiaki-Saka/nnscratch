// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
#include "nnscratch/activations.hpp"

#include <algorithm>
#include <cmath>

namespace nn {

Tensor softmax(const Tensor& logits) {
    const std::size_t n = logits.rows(), k = logits.cols();
    Tensor out(n, k);
    for (std::size_t i = 0; i < n; ++i) {
        double m = logits(i, 0);
        for (std::size_t j = 1; j < k; ++j) m = std::max(m, logits(i, j));
        double sum = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            const double e = std::exp(logits(i, j) - m);
            out(i, j) = e;
            sum += e;
        }
        for (std::size_t j = 0; j < k; ++j) out(i, j) /= sum;
    }
    return out;
}

Tensor ReLU::forward(const Tensor& x) {
    mask_ = x.map([](double v) { return v > 0.0 ? 1.0 : 0.0; });
    return x.map([](double v) { return v > 0.0 ? v : 0.0; });
}

Tensor ReLU::backward(const Tensor& grad_out) { return grad_out * mask_; }

Tensor Tanh::forward(const Tensor& x) {
    out_ = x.map([](double v) { return std::tanh(v); });
    return out_;
}

Tensor Tanh::backward(const Tensor& grad_out) {
    return grad_out * out_.map([](double y) { return 1.0 - y * y; });
}

Tensor Sigmoid::forward(const Tensor& x) {
    out_ = x.map([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
    return out_;
}

Tensor Sigmoid::backward(const Tensor& grad_out) {
    return grad_out * out_.map([](double y) { return y * (1.0 - y); });
}

}  // namespace nn
