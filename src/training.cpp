// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
#include "nnscratch/training.hpp"

#include <algorithm>
#include <cstdio>

#include "nnscratch/loss.hpp"
#include "nnscratch/rng.hpp"

namespace nn {

namespace {

// Gather the rows of `src` listed in `idx` into a new tensor, preserving the
// trailing dimensions (works for both (N,64) and (N,1,8,8)).
Tensor gather_rows(const Tensor& src, const std::vector<std::size_t>& idx) {
    const std::size_t row_size = src.size() / src.dim(0);
    std::vector<std::size_t> shape = src.shape();
    shape[0] = idx.size();
    Tensor out(shape);
    for (std::size_t r = 0; r < idx.size(); ++r) {
        const double* from = src.data().data() + idx[r] * row_size;
        std::copy(from, from + row_size, out.data().data() + r * row_size);
    }
    return out;
}

double batch_accuracy(const Tensor& logits, const std::vector<int>& y) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < logits.rows(); ++i) {
        std::size_t best = 0;
        double best_val = logits(i, 0);
        for (std::size_t j = 1; j < logits.cols(); ++j)
            if (logits(i, j) > best_val) best_val = logits(i, j), best = j;
        correct += (static_cast<int>(best) == y[i]);
    }
    return static_cast<double>(correct) / static_cast<double>(y.size());
}

}  // namespace

History train(Model& model, Optimizer& opt, const Tensor& x_train,
              const std::vector<int>& y_train, const Tensor& x_test,
              const std::vector<int>& y_test, const TrainConfig& cfg) {
    SoftmaxCrossEntropy loss_fn;
    std::size_t num_classes = 1;
    for (int label : y_train) num_classes = std::max(num_classes, static_cast<std::size_t>(label) + 1);
    const Tensor Y_train = one_hot(y_train, num_classes);
    const std::size_t n = x_train.dim(0);
    Rng batch_rng(cfg.batch_seed);

    History hist;
    for (int ep = 0; ep <= cfg.epochs; ++ep) {
        if (ep > 0) {
            const std::vector<std::size_t> order = batch_rng.permutation(n);
            for (std::size_t s = 0; s < n; s += cfg.batch_size) {
                const std::size_t end = std::min(s + cfg.batch_size, n);
                const std::vector<std::size_t> bidx(
                    order.begin() + static_cast<std::ptrdiff_t>(s),
                    order.begin() + static_cast<std::ptrdiff_t>(end));
                Tensor xb = gather_rows(x_train, bidx);
                Tensor yb = gather_rows(Y_train, bidx);
                Tensor logits = model.forward(xb);
                loss_fn.forward(logits, yb);
                model.backward(loss_fn.backward());
                opt.step(model.params_and_grads());
            }
        }
        Tensor train_logits = model.forward(x_train);
        const double train_loss = loss_fn.forward(train_logits, Y_train);
        const double tr_acc = batch_accuracy(train_logits, y_train);
        const double te_acc = model.accuracy(x_test, y_test);
        hist.epoch.push_back(ep);
        hist.loss.push_back(train_loss);
        hist.train_acc.push_back(tr_acc);
        hist.test_acc.push_back(te_acc);
        if (cfg.verbose) {
            std::printf("epoch %3d | loss %.4f | train_acc %5.1f%% | test_acc %5.1f%%\n",
                        ep, train_loss, tr_acc * 100.0, te_acc * 100.0);
        }
    }
    return hist;
}

}  // namespace nn
