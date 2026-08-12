// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#include "check.hpp"
#include "nnscratch/nnscratch.hpp"

using namespace nn;

int main() {
    // Adam should drive a single parameter toward the minimum of (w - 3)^2,
    // whose gradient is 2(w - 3).
    {
        Tensor w = Tensor::from({1, 1}, {0.0});
        Tensor g(1, 1);
        Adam opt(0.1);
        for (int step = 0; step < 500; ++step) {
            g.data()[0] = 2.0 * (w.data()[0] - 3.0);
            opt.step({{&w, &g}});
        }
        CHECK_CLOSE(w.data()[0], 3.0, 1e-2);
    }

    // SGD on the same objective.
    {
        Tensor w = Tensor::from({1, 1}, {0.0});
        Tensor g(1, 1);
        SGD opt(0.1);
        for (int step = 0; step < 500; ++step) {
            g.data()[0] = 2.0 * (w.data()[0] - 3.0);
            opt.step({{&w, &g}});
        }
        CHECK_CLOSE(w.data()[0], 3.0, 1e-3);
    }

    // Momentum reaches the minimum too.
    {
        Tensor w = Tensor::from({1, 1}, {0.0});
        Tensor g(1, 1);
        Momentum opt(0.05, 0.9);
        for (int step = 0; step < 500; ++step) {
            g.data()[0] = 2.0 * (w.data()[0] - 3.0);
            opt.step({{&w, &g}});
        }
        CHECK_CLOSE(w.data()[0], 3.0, 1e-2);
    }

    // End-to-end: a tiny MLP should reach high accuracy on a trivially
    // separable 2-class problem within a few epochs.
    {
        Rng rng(1);
        // Class 0: first feature high; class 1: second feature high.
        const std::size_t n = 40;
        Tensor X(n, 2);
        std::vector<int> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            const bool c1 = (i % 2 == 0);
            X(i, 0) = c1 ? 0.1 : 1.0;
            X(i, 1) = c1 ? 1.0 : 0.1;
            y[i] = c1 ? 1 : 0;
        }
        Model m;
        m.add<Dense>(2, 8, rng, Init::He);
        m.add<ReLU>();
        m.add<Dense>(8, 2, rng, Init::Xavier);
        Adam opt(0.05);
        TrainConfig cfg;
        cfg.epochs = 30;
        cfg.batch_size = 8;
        train(m, opt, X, y, X, y, cfg);
        CHECK(m.accuracy(X, y) > 0.95);
    }

    return nntest::summary("optimizer");
}
