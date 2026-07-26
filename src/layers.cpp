// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#include <cmath>

#include "nnscratch/layers.hpp"

namespace nn {

namespace {
double init_std(Init init, std::size_t fan_in, std::size_t fan_out) {
    switch (init) {
        case Init::He:
            return std::sqrt(2.0 / static_cast<double>(fan_in));
        case Init::Xavier:
            return std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
    }
    return 1.0;
}
}  // namespace

Dense::Dense(std::size_t n_in, std::size_t n_out, Rng& rng, Init init)
    : W_(rng.normal({n_in, n_out}, init_std(init, n_in, n_out))),
      b_(1, n_out),
      dW_(n_in, n_out),
      db_(1, n_out) {}

Tensor Dense::forward(const Tensor& x) {
    x_ = x;
    Tensor out = matmul(x, W_);
    out.add_row_vector(b_);
    return out;
}

Tensor Dense::backward(const Tensor& grad_out) {
    dW_ = matmul(x_.transpose(), grad_out);
    db_ = grad_out.sum_rows();
    return matmul(grad_out, W_.transpose());
}

std::vector<ParamGrad> Dense::params_and_grads() {
    return {{&W_, &dW_}, {&b_, &db_}};
}

Tensor Flatten::forward(const Tensor& x) {
    in_shape_ = x.shape();
    return x.flatten_batch();
}

Tensor Flatten::backward(const Tensor& grad_out) {
    return grad_out.reshape(in_shape_);
}

}  // namespace nn
