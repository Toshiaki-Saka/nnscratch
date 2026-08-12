// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#ifndef NNSCRATCH_DATASET_HPP
#define NNSCRATCH_DATASET_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "nnscratch/tensor.hpp"

namespace nn {

/// The 8x8 handwritten-digits dataset, pre-split into train/test.
///
/// Each split is provided in two views of the *same* samples:
///   * `flat` : (N, 64)        for fully-connected models
///   * `img`  : (N, 1, 8, 8)   for convolutional models
/// Pixel values are normalised to [0, 1].
struct DigitsData {
    struct Split {
        Tensor flat;              // (N, 64)
        Tensor img;               // (N, 1, 8, 8)
        std::vector<int> labels;  // (N)
    };
    Split train;
    Split test;
};

/// Load the digits CSV produced by `data/digits.csv` and split it
/// `train_frac` / (1 - train_frac) using a fixed permutation seed so the
/// split is reproducible across runs and platforms.
DigitsData load_digits(const std::string& csv_path, double train_frac = 0.8,
                       std::uint64_t split_seed = 0);

}  // namespace nn

#endif  // NNSCRATCH_DATASET_HPP
