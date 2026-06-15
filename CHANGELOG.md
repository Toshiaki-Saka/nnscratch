# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-05

### Added
- Initial release: a dependency-free, from-scratch neural network library in C++20.
- `Tensor` type with the linear-algebra operations needed for MLPs and CNNs.
- Layers: `Dense`, `Conv2D` (im2col), `Flatten`, `ReLU`, `Tanh`, `Sigmoid`.
- `SoftmaxCrossEntropy` loss and `SGD` / `Momentum` / `Adam` optimizers.
- `Model` container and a reusable mini-batch training loop.
- Bundled 8x8 handwritten-digits dataset and two demos (`from_scratch`, `compare`).
- CTest suite including a numerical gradient check covering every layer's backward pass.
- CMake packaging with `find_package(nnscratch)` support.
