# nnscratch

[![CI](https://github.com/Toshiaki-Saka/nnscratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Toshiaki-Saka/nnscratch/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A tiny neural-network library written **from scratch in modern C++ (C++20)** —
no BLAS, no Eigen, no framework, no third-party runtime dependencies. It is a
faithful port of a numpy teaching project: you can read every line of the
forward pass, the loss, the hand-derived `backward()` of each layer (including
the convolution), and the optimizer update rules.

The goal is pedagogical clarity, not speed: it exists so you can *see* a network
go from random guessing to reading handwritten digits, and so the math of
backpropagation is never a black box.

```
Untrained test accuracy:  ~5–9%   (≈10% = random for 10 classes)
...
Final test accuracy:     ~97–98%
```

Reproduce with `./build/from_scratch` (the dataset is bundled; no flags needed).
The run is deterministic for a given toolchain, but the exact figures vary by a
few tenths of a percent across compilers/OSes — the standard library's random
*distributions* (`std::shuffle`, `std::uniform_real_distribution`) are not
specified to produce identical sequences from the same seed across STL
implementations. A representative MSVC run gives 5.3% → 97.8%.

## Why this exists

Frameworks like PyTorch and TensorFlow replace the most instructive part of a
network — the chain-rule gradient computation — with a single `loss.backward()`
call. This library writes that part out by hand, then maps each piece back to
its framework equivalent (see [the correspondence table](#numpy--c--framework-correspondence)),
so the framework stops being magic.

## Features

- Header + source library, **zero runtime dependencies** (only the C++20 standard library).
- Layers: `Dense`, `Conv2D` (via im2col), `Flatten`, and `ReLU` / `Tanh` / `Sigmoid`.
- Loss: `SoftmaxCrossEntropy` (fused, so the output gradient collapses to `p - y`).
- Optimizers: `SGD`, `Momentum`, `Adam`.
- A `Model` container, a reusable mini-batch training loop, and a reproducible RNG.
- Bundled 8×8 handwritten-digits dataset (`data/digits.csv`) so the demos run **fully offline**.
- A numerical **gradient check** in the test suite that validates every layer's `backward()`.
- Proper CMake packaging: `find_package(nnscratch)` + `nnscratch::nnscratch`.

## Build

Requires a C++20 compiler (GCC ≥ 10, Clang ≥ 12, MSVC ≥ 19.29) and CMake ≥ 3.16.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the demos:

```bash
./build/from_scratch          # Part 1: untrained -> trained
./build/compare               # Part 2: optimizer / activation / architecture comparison
```

On Windows, `run_demo.ps1` (PowerShell 7+) does the whole thing in one command —
build, `ctest`, then the two demos with animated progress bars and ASCII charts:

```powershell
.\run_demo.ps1              # full run
.\run_demo.ps1 -SkipBuild   # skip the build/test step and just run the demos
```

Each demo accepts optional arguments: `from_scratch [digits.csv] [output_dir]`.
They emit a learning-curve CSV and grayscale `.pgm` images of the learned
weights/filters — visualization is intentionally left to the consumer (plot the
CSV with whatever you like) to keep the core dependency-free.

## Use it as a library

```cpp
#include <nnscratch/nnscratch.hpp>
using namespace nn;

Rng rng(42);
Model net;
net.add<Dense>(64, 64, rng, Init::He);
net.add<ReLU>();
net.add<Dense>(64, 32, rng, Init::He);
net.add<ReLU>();
net.add<Dense>(32, 10, rng, Init::He);

DigitsData data = load_digits("data/digits.csv");
SGD opt(0.3);
TrainConfig cfg; cfg.epochs = 60; cfg.verbose = true;
train(net, opt, data.train.flat, data.train.labels,
      data.test.flat, data.test.labels, cfg);
```

Consume it from another CMake project after `cmake --install`:

```cmake
find_package(nnscratch REQUIRED)
target_link_libraries(your_app PRIVATE nnscratch::nnscratch)
```

## What the demos show

**Part 1 (`from_scratch`)** — a 64→64→32→10 MLP starts at ~9% accuracy (random)
and climbs to ~97% as loss falls and the first-layer weights organise into
digit-stroke detectors.

**Part 2 (`compare`)** — three controlled experiments, each varying one axis
while holding the data and initial weights fixed:

| Experiment | Observation (matches the numpy reference) |
|---|---|
| Optimizers | Adam / Momentum converge faster than plain SGD |
| Activations | Sigmoid is slowest to reach 90% (vanishing gradients); ReLU/Tanh are quick |
| Architecture | CNN ≥ deep MLP > shallow (no hidden layer) |

## numpy → C++ → framework correspondence

| This library (C++) | numpy reference | PyTorch | TensorFlow (Keras) |
|---|---|---|---|
| `Dense` | `Dense` | `nn.Linear` | `layers.Dense` |
| `Conv2D` (im2col) | `Conv2D` | `nn.Conv2d` | `layers.Conv2D` |
| `ReLU`/`Tanh`/`Sigmoid` | same | `nn.ReLU` etc. | `activation="relu"` |
| `Flatten` | `Flatten` | `nn.Flatten` | `layers.Flatten` |
| `SoftmaxCrossEntropy` | `SoftmaxCrossEntropy` | `nn.CrossEntropyLoss` | `SparseCategoricalCrossentropy(from_logits=True)` |
| `Model` | `Model` | `nn.Sequential` | `keras.Sequential` |
| **hand-written `backward()`** | **hand-written `backward()`** | **`loss.backward()`** | **`tape.gradient(...)`** |
| `SGD`/`Momentum`/`Adam` | same | `torch.optim.*` | `keras.optimizers.*` |

The single most important row is the bold one: everything the frameworks
automate via autograd is spelled out explicitly here.

## Project layout

```
include/nnscratch/   public headers (Tensor, layers, loss, optimizers, model, ...)
src/                 implementations
apps/                from_scratch.cpp, compare.cpp  (the two demos)
tests/               tensor / gradient-check / optimizer tests (CTest)
data/digits.csv      bundled dataset
reference/           notes on the numpy/PyTorch/TensorFlow correspondence
docs_en/             design & experiment notes (English)
docs_ja/             the same design & experiment notes (Japanese)
run_demo.ps1         Windows demo runner (build + test + animated demos)
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Acknowledgements

Ported from a numpy-only teaching implementation of an MLP/CNN on the 8×8
handwritten-digits dataset (a subset of the UCI optical-recognition dataset, as
shipped with scikit-learn). The C++ port preserves the structure and the
hand-derived gradients of the original.
