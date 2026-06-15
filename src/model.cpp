// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nnscratch contributors
#include "nnscratch/model.hpp"

namespace nn {

Tensor Model::forward(const Tensor& x) {
    Tensor out = x;
    for (auto& l : layers_) out = l->forward(out);
    return out;
}

void Model::backward(const Tensor& grad) {
    Tensor g = grad;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) g = (*it)->backward(g);
}

std::vector<ParamGrad> Model::params_and_grads() {
    std::vector<ParamGrad> pgs;
    for (auto& l : layers_) {
        auto layer_pgs = l->params_and_grads();
        pgs.insert(pgs.end(), layer_pgs.begin(), layer_pgs.end());
    }
    return pgs;
}

std::vector<int> Model::predict(const Tensor& x) {
    Tensor out = forward(x);
    std::vector<int> preds(out.rows());
    for (std::size_t i = 0; i < out.rows(); ++i) {
        std::size_t best = 0;
        double best_val = out(i, 0);
        for (std::size_t j = 1; j < out.cols(); ++j) {
            if (out(i, j) > best_val) {
                best_val = out(i, j);
                best = j;
            }
        }
        preds[i] = static_cast<int>(best);
    }
    return preds;
}

double Model::accuracy(const Tensor& x, const std::vector<int>& labels) {
    std::vector<int> preds = predict(x);
    std::size_t correct = 0;
    for (std::size_t i = 0; i < preds.size(); ++i) correct += (preds[i] == labels[i]);
    return static_cast<double>(correct) / static_cast<double>(preds.size());
}

}  // namespace nn
