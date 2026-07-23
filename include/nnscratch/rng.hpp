// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
#ifndef NNSCRATCH_RNG_HPP
#define NNSCRATCH_RNG_HPP

#include <cstdint>
#include <random>

#include "nnscratch/tensor.hpp"

namespace nn {

/// A thin, reproducible PRNG wrapper.
///
/// The original Python code resets a global generator to a fixed seed before
/// constructing each model so that competing configurations start from the
/// *same* random weights. `Rng` makes that intent explicit and local instead
/// of relying on a hidden global, while still being trivial to reseed.
class Rng {
public:
    explicit Rng(std::uint64_t seed = 42) : engine_(seed) {}

    void reseed(std::uint64_t seed) { engine_.seed(seed); }

    /// Draw a single standard-normal sample.
    double normal() {
        std::normal_distribution<double> dist(0.0, 1.0);
        return dist(engine_);
    }

    /// Fill a tensor of `shape` with N(0, std^2) samples.
    Tensor normal(std::vector<std::size_t> shape, double std_dev) {
        Tensor t(std::move(shape));
        std::normal_distribution<double> dist(0.0, std_dev);
        for (double& v : t.data()) v = dist(engine_);
        return t;
    }

    /// A random permutation of {0, 1, ..., n-1}.
    std::vector<std::size_t> permutation(std::size_t n) {
        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        for (std::size_t i = n; i > 1; --i) {
            std::uniform_int_distribution<std::size_t> dist(0, i - 1);
            std::swap(idx[i - 1], idx[dist(engine_)]);
        }
        return idx;
    }

private:
    std::mt19937_64 engine_;
};

}  // namespace nn

#endif  // NNSCRATCH_RNG_HPP
