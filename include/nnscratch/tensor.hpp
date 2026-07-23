// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
#ifndef NNSCRATCH_TENSOR_HPP
#define NNSCRATCH_TENSOR_HPP

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace nn {

/// A dense, row-major, double-precision tensor.
///
/// The library only ever needs rank-2 tensors `(rows, cols)` for the
/// matrix math, and rank-4 tensors `(N, C, H, W)` for convolutions, so this
/// type deliberately keeps a single flat `std::vector<double>` buffer and a
/// small `shape` vector rather than pulling in a general n-d array library.
/// This mirrors the "numpy only" spirit of the original teaching code while
/// staying dependency-free.
class Tensor {
public:
    Tensor() = default;

    /// Zero-initialised rank-2 tensor.
    Tensor(std::size_t rows, std::size_t cols);

    /// Zero-initialised tensor of arbitrary shape.
    explicit Tensor(std::vector<std::size_t> shape);

    /// Build from an explicit shape and matching flat data buffer.
    static Tensor from(std::vector<std::size_t> shape, std::vector<double> data);

    static Tensor zeros(std::vector<std::size_t> shape) { return Tensor(std::move(shape)); }
    static Tensor zeros_like(const Tensor& t) { return Tensor(t.shape_); }

    // --- shape queries ----------------------------------------------------
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] std::size_t dim(std::size_t i) const { return shape_.at(i); }

    [[nodiscard]] std::size_t rows() const;  ///< rank-2 only
    [[nodiscard]] std::size_t cols() const;  ///< rank-2 only

    // --- element access ---------------------------------------------------
    [[nodiscard]] double& operator()(std::size_t i, std::size_t j);
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const;

    [[nodiscard]] std::vector<double>& data() noexcept { return data_; }
    [[nodiscard]] const std::vector<double>& data() const noexcept { return data_; }

    // --- reshaping (total size must be preserved) -------------------------
    [[nodiscard]] Tensor reshape(std::vector<std::size_t> new_shape) const;
    [[nodiscard]] Tensor flatten_batch() const;  ///< (N, ...) -> (N, prod(rest))

    // --- linear-algebra helpers (rank-2) ----------------------------------
    [[nodiscard]] Tensor transpose() const;

    /// Sum over the batch axis (axis 0): (rows, cols) -> (1, cols).
    [[nodiscard]] Tensor sum_rows() const;

    /// Apply a unary function element-wise, returning a new tensor.
    [[nodiscard]] Tensor map(const std::function<double(double)>& f) const;

    /// Add a row vector `bias` (shape (1, cols) or (cols)) to every row,
    /// in place.
    void add_row_vector(const Tensor& bias);

    // --- in-place SAXPY: *this += alpha * other ---------------------------
    void axpy(double alpha, const Tensor& other);

private:
    std::vector<std::size_t> shape_{};
    std::vector<double> data_{};
};

// --- free-function operators ----------------------------------------------
Tensor matmul(const Tensor& a, const Tensor& b);  ///< (m,k)x(k,n) -> (m,n)

Tensor operator+(const Tensor& a, const Tensor& b);  ///< element-wise
Tensor operator-(const Tensor& a, const Tensor& b);  ///< element-wise
Tensor operator*(const Tensor& a, const Tensor& b);  ///< Hadamard (element-wise)
Tensor operator*(double s, const Tensor& a);         ///< scalar scale

std::ostream& operator<<(std::ostream& os, const Tensor& t);

}  // namespace nn

#endif  // NNSCRATCH_TENSOR_HPP
