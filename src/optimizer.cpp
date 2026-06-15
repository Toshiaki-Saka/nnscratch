// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nnscratch contributors
#include "nnscratch/optimizer.hpp"

#include <cmath>

namespace nn {

void SGD::step(const std::vector<ParamGrad>& pgs) {
    for (const auto& [p, g] : pgs) p->axpy(-lr_, *g);
}

void Momentum::step(const std::vector<ParamGrad>& pgs) {
    for (const auto& [p, g] : pgs) {
        auto it = v_.find(p);
        if (it == v_.end()) it = v_.emplace(p, Tensor::zeros_like(*p)).first;
        Tensor& v = it->second;
        // v = mu * v - lr * g
        for (std::size_t i = 0; i < v.size(); ++i)
            v.data()[i] = mu_ * v.data()[i] - lr_ * g->data()[i];
        p->axpy(1.0, v);  // p += v
    }
}

void Adam::step(const std::vector<ParamGrad>& pgs) {
    ++t_;
    const double bc1 = 1.0 - std::pow(b1_, static_cast<double>(t_));
    const double bc2 = 1.0 - std::pow(b2_, static_cast<double>(t_));
    for (const auto& [p, g] : pgs) {
        auto mit = m_.find(p);
        if (mit == m_.end()) mit = m_.emplace(p, Tensor::zeros_like(*p)).first;
        auto vit = v_.find(p);
        if (vit == v_.end()) vit = v_.emplace(p, Tensor::zeros_like(*p)).first;
        Tensor& m = mit->second;
        Tensor& v = vit->second;
        for (std::size_t i = 0; i < m.size(); ++i) {
            const double gi = g->data()[i];
            m.data()[i] = b1_ * m.data()[i] + (1.0 - b1_) * gi;
            v.data()[i] = b2_ * v.data()[i] + (1.0 - b2_) * gi * gi;
            const double mh = m.data()[i] / bc1;
            const double vh = v.data()[i] / bc2;
            p->data()[i] -= lr_ * mh / (std::sqrt(vh) + eps_);
        }
    }
}

}  // namespace nn
