// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nnscratch contributors
#include "nnscratch/tensor.hpp"

#include <numeric>
#include <stdexcept>

namespace nn {
namespace {

std::size_t product(const std::vector<std::size_t>& s) {
    return std::accumulate(s.begin(), s.end(), std::size_t{1}, std::multiplies<>{});
}

}  // namespace

Tensor::Tensor(std::size_t rows, std::size_t cols)
    : shape_{rows, cols}, data_(rows * cols, 0.0) {}

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), data_(product(shape_), 0.0) {}

Tensor Tensor::from(std::vector<std::size_t> shape, std::vector<double> data) {
    if (product(shape) != data.size()) {
        throw std::invalid_argument("Tensor::from: shape does not match data size");
    }
    Tensor t;
    t.shape_ = std::move(shape);
    t.data_ = std::move(data);
    return t;
}

std::size_t Tensor::rows() const {
    if (rank() != 2) throw std::logic_error("rows() requires a rank-2 tensor");
    return shape_[0];
}

std::size_t Tensor::cols() const {
    if (rank() != 2) throw std::logic_error("cols() requires a rank-2 tensor");
    return shape_[1];
}

double& Tensor::operator()(std::size_t i, std::size_t j) {
    return data_[i * shape_[1] + j];
}

double Tensor::operator()(std::size_t i, std::size_t j) const {
    return data_[i * shape_[1] + j];
}

Tensor Tensor::reshape(std::vector<std::size_t> new_shape) const {
    if (product(new_shape) != data_.size()) {
        throw std::invalid_argument("reshape: total size mismatch");
    }
    Tensor t = *this;
    t.shape_ = std::move(new_shape);
    return t;
}

Tensor Tensor::flatten_batch() const {
    const std::size_t n = shape_.at(0);
    return reshape({n, data_.size() / n});
}

Tensor Tensor::transpose() const {
    if (rank() != 2) throw std::logic_error("transpose() requires a rank-2 tensor");
    Tensor out(shape_[1], shape_[0]);
    for (std::size_t i = 0; i < shape_[0]; ++i)
        for (std::size_t j = 0; j < shape_[1]; ++j) out(j, i) = (*this)(i, j);
    return out;
}

Tensor Tensor::sum_rows() const {
    if (rank() != 2) throw std::logic_error("sum_rows() requires a rank-2 tensor");
    Tensor out(1, shape_[1]);
    for (std::size_t i = 0; i < shape_[0]; ++i)
        for (std::size_t j = 0; j < shape_[1]; ++j) out(0, j) += (*this)(i, j);
    return out;
}

Tensor Tensor::map(const std::function<double(double)>& f) const {
    Tensor out = *this;
    for (double& v : out.data_) v = f(v);
    return out;
}

void Tensor::add_row_vector(const Tensor& bias) {
    if (rank() != 2) throw std::logic_error("add_row_vector requires a rank-2 tensor");
    if (bias.size() != shape_[1]) {
        throw std::invalid_argument("add_row_vector: bias width mismatch");
    }
    for (std::size_t i = 0; i < shape_[0]; ++i)
        for (std::size_t j = 0; j < shape_[1]; ++j) (*this)(i, j) += bias.data_[j];
}

void Tensor::axpy(double alpha, const Tensor& other) {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument("axpy: size mismatch");
    }
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += alpha * other.data_[i];
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        throw std::logic_error("matmul requires rank-2 tensors");
    }
    if (a.cols() != b.rows()) throw std::invalid_argument("matmul: inner dim mismatch");
    const std::size_t m = a.rows(), k = a.cols(), n = b.cols();
    Tensor out(m, n);
    // ikj loop order keeps the inner accumulation over a contiguous row of b.
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const double aip = a(i, p);
            for (std::size_t j = 0; j < n; ++j) out(i, j) += aip * b(p, j);
        }
    }
    return out;
}

namespace {
template <class Op>
Tensor elementwise(const Tensor& a, const Tensor& b, Op op) {
    if (a.shape() != b.shape()) throw std::invalid_argument("element-wise: shape mismatch");
    Tensor out = a;
    auto& od = out.data();
    const auto& bd = b.data();
    for (std::size_t i = 0; i < od.size(); ++i) od[i] = op(od[i], bd[i]);
    return out;
}
}  // namespace

Tensor operator+(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double x, double y) { return x + y; });
}
Tensor operator-(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double x, double y) { return x - y; });
}
Tensor operator*(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double x, double y) { return x * y; });
}
Tensor operator*(double s, const Tensor& a) {
    Tensor out = a;
    for (double& v : out.data()) v *= s;
    return out;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape=[";
    for (std::size_t i = 0; i < t.shape().size(); ++i) {
        os << t.shape()[i] << (i + 1 < t.shape().size() ? "," : "");
    }
    os << "])";
    return os;
}

}  // namespace nn
