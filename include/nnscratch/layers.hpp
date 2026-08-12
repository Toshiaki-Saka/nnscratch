// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#ifndef NNSCRATCH_LAYERS_HPP
#define NNSCRATCH_LAYERS_HPP

#include <cstddef>

#include "nnscratch/layer.hpp"
#include "nnscratch/rng.hpp"
#include "nnscratch/tensor.hpp"

namespace nn {

/// Weight-initialisation strategy for a Dense / Conv layer.
enum class Init {
    He,      ///< std = sqrt(2 / fan_in); pairs well with ReLU.
    Xavier,  ///< std = sqrt(2 / (fan_in + fan_out)); activation-neutral.
};

/// Fully-connected layer: out = x . W + b.
class Dense final : public Layer {
public:
    Dense(std::size_t n_in, std::size_t n_out, Rng& rng, Init init = Init::Xavier);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamGrad> params_and_grads() override;

    Tensor& weight() noexcept { return W_; }
    Tensor& bias() noexcept { return b_; }

private:
    Tensor W_, b_;    // parameters
    Tensor dW_, db_;  // gradients
    Tensor x_;        // cached input
};

/// Flattens a rank-4 (N, C, H, W) tensor to rank-2 (N, C*H*W) and restores the
/// original shape on the backward pass.
class Flatten final : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

private:
    std::vector<std::size_t> in_shape_;
};

/// 2-D convolution implemented via im2col, so the heavy lifting reduces to a
/// single matrix multiply (the classic CS231n trick).
class Conv2D final : public Layer {
public:
    Conv2D(std::size_t in_c, std::size_t out_c, std::size_t k, std::size_t stride,
           std::size_t pad, Rng& rng, Init init = Init::He);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamGrad> params_and_grads() override;

    Tensor& weight() noexcept { return W_; }  // (out_c, in_c, k, k)

private:
    std::size_t k_, stride_, pad_;
    Tensor W_, b_;  // (out_c, in_c, k, k), (out_c)
    Tensor dW_, db_;
    Tensor col_;                        // cached im2col matrix
    std::vector<std::size_t> x_shape_;  // cached input shape
    std::size_t oh_{}, ow_{};           // cached output spatial dims
};

}  // namespace nn

#endif  // NNSCRATCH_LAYERS_HPP
