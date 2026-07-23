// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
//
// The central correctness test for the whole library: for every learnable
// parameter, the analytic gradient produced by backward() must match a
// finite-difference estimate of d(loss)/d(param). If any layer's backward()
// is wrong, this fails.
#include <algorithm>
#include <cmath>

#include "nnscratch/nnscratch.hpp"
#include "check.hpp"

using namespace nn;

namespace {

double loss_of(Model& m, SoftmaxCrossEntropy& L, const Tensor& X, const Tensor& Y) {
    Tensor logits = m.forward(X);
    return L.forward(logits, Y);
}

void grad_check(Model& m, const Tensor& X, const std::vector<int>& y, std::size_t K) {
    SoftmaxCrossEntropy L;
    const Tensor Y = one_hot(y, K);

    // Analytic gradients.
    Tensor logits = m.forward(X);
    L.forward(logits, Y);
    m.backward(L.backward());
    auto pgs = m.params_and_grads();

    const double eps = 1e-5;
    for (auto& pg : pgs) {
        const std::vector<double> analytic = pg.grad->data();  // snapshot
        Tensor* P = pg.param;
        const std::size_t n = P->size();
        const std::size_t stride = std::max<std::size_t>(1, n / 16);  // sample ~16 elems
        for (std::size_t i = 0; i < n; i += stride) {
            const double orig = P->data()[i];
            P->data()[i] = orig + eps;
            const double lp = loss_of(m, L, X, Y);
            P->data()[i] = orig - eps;
            const double lm = loss_of(m, L, X, Y);
            P->data()[i] = orig;
            const double numeric = (lp - lm) / (2.0 * eps);
            CHECK_CLOSE(numeric, analytic[i], 1e-4 * (1.0 + std::fabs(analytic[i])));
        }
    }
}

}  // namespace

int main() {
    // --- MLP: Dense -> ReLU -> Dense -> Tanh -> Dense -------------------------
    {
        Rng rng(7);
        Model m;
        m.add<Dense>(8, 6, rng, Init::He);
        m.add<ReLU>();
        m.add<Dense>(6, 5, rng, Init::Xavier);
        m.add<Tanh>();
        m.add<Dense>(5, 4, rng, Init::Xavier);

        Tensor X = rng.normal({5, 8}, 1.0);
        std::vector<int> y = {0, 3, 1, 2, 0};
        grad_check(m, X, y, 4);
    }

    // --- Sigmoid path ---------------------------------------------------------
    {
        Rng rng(11);
        Model m;
        m.add<Dense>(6, 5, rng, Init::Xavier);
        m.add<Sigmoid>();
        m.add<Dense>(5, 3, rng, Init::Xavier);

        Tensor X = rng.normal({4, 6}, 1.0);
        std::vector<int> y = {2, 0, 1, 2};
        grad_check(m, X, y, 3);
    }

    // --- CNN: Conv2D -> ReLU -> Flatten -> Dense ------------------------------
    {
        Rng rng(13);
        Model m;
        m.add<Conv2D>(1, 2, 3, 1, 0, rng, Init::He);  // (N,1,5,5) -> (N,2,3,3)
        m.add<ReLU>();
        m.add<Flatten>();
        m.add<Dense>(2 * 3 * 3, 3, rng, Init::Xavier);

        Tensor X = rng.normal({3, 1, 5, 5}, 1.0);
        std::vector<int> y = {0, 2, 1};
        grad_check(m, X, y, 3);
    }

    return nntest::summary("gradcheck");
}
