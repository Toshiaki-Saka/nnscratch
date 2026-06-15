// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nnscratch contributors
#ifndef NNSCRATCH_MODEL_HPP
#define NNSCRATCH_MODEL_HPP

#include <memory>
#include <vector>

#include "nnscratch/layer.hpp"
#include "nnscratch/tensor.hpp"

namespace nn {

/// A sequential stack of layers (the C++ analogue of `nn.Sequential` /
/// `keras.Sequential`). Owns its layers.
class Model {
public:
    Model() = default;
    explicit Model(std::vector<std::unique_ptr<Layer>> layers)
        : layers_(std::move(layers)) {}

    /// Append a layer and return a reference to it for fluent construction.
    template <class L, class... Args>
    L& add(Args&&... args) {
        auto layer = std::make_unique<L>(std::forward<Args>(args)...);
        L& ref = *layer;
        layers_.push_back(std::move(layer));
        return ref;
    }

    /// Append an already-constructed layer.
    Layer& push(std::unique_ptr<Layer> layer) {
        Layer& ref = *layer;
        layers_.push_back(std::move(layer));
        return ref;
    }

    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad);
    std::vector<ParamGrad> params_and_grads();

    /// Argmax class predictions for each row of `x`.
    std::vector<int> predict(const Tensor& x);
    /// Fraction of `x` rows whose prediction matches `labels`.
    double accuracy(const Tensor& x, const std::vector<int>& labels);

    Layer& layer(std::size_t i) { return *layers_.at(i); }
    std::size_t size() const noexcept { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace nn

#endif  // NNSCRATCH_MODEL_HPP
