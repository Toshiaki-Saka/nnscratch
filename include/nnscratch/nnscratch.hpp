// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
//
// nnscratch — a tiny, dependency-free neural network library written from
// scratch in modern C++. Include this single header to pull in the whole
// public API.
#ifndef NNSCRATCH_NNSCRATCH_HPP
#define NNSCRATCH_NNSCRATCH_HPP

#include "nnscratch/activations.hpp"
#include "nnscratch/dataset.hpp"
#include "nnscratch/layer.hpp"
#include "nnscratch/layers.hpp"
#include "nnscratch/loss.hpp"
#include "nnscratch/model.hpp"
#include "nnscratch/optimizer.hpp"
#include "nnscratch/rng.hpp"
#include "nnscratch/tensor.hpp"
#include "nnscratch/training.hpp"

/// Library version, kept in sync with the project() version in CMake.
#define NNSCRATCH_VERSION_MAJOR 0
#define NNSCRATCH_VERSION_MINOR 1
#define NNSCRATCH_VERSION_PATCH 0

#endif  // NNSCRATCH_NNSCRATCH_HPP
