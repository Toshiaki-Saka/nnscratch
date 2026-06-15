// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nnscratch contributors
#ifndef NNSCRATCH_TRAINING_HPP
#define NNSCRATCH_TRAINING_HPP

#include <cstdint>
#include <vector>

#include "nnscratch/model.hpp"
#include "nnscratch/optimizer.hpp"
#include "nnscratch/tensor.hpp"

namespace nn {

/// Per-epoch metrics recorded during training.
struct History {
    std::vector<int> epoch;
    std::vector<double> loss;
    std::vector<double> train_acc;
    std::vector<double> test_acc;
};

/// Configuration for a training run.
struct TrainConfig {
    int epochs = 40;
    std::size_t batch_size = 32;
    std::uint64_t batch_seed = 123;  ///< fixes the mini-batch order across runs
    bool verbose = false;
};

/// Mini-batch SGD-style training loop shared by all demos.
///
/// Epoch 0 records the metrics of the *untrained* network before any update,
/// so the history captures the full "random -> trained" trajectory.
History train(Model& model, Optimizer& opt, const Tensor& x_train,
              const std::vector<int>& y_train, const Tensor& x_test,
              const std::vector<int>& y_test, const TrainConfig& cfg);

}  // namespace nn

#endif  // NNSCRATCH_TRAINING_HPP
