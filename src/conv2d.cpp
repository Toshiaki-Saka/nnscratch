// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#include <cmath>
#include <stdexcept>

#include "nnscratch/layers.hpp"

namespace nn {

namespace {
double conv_init_std(Init init, std::size_t fan_in) {
    // Conv only ever uses He in the reference; Xavier handled for completeness.
    return init == Init::Xavier ? std::sqrt(1.0 / static_cast<double>(fan_in))
                                : std::sqrt(2.0 / static_cast<double>(fan_in));
}
}  // namespace

Conv2D::Conv2D(std::size_t in_c, std::size_t out_c, std::size_t k, std::size_t stride,
               std::size_t pad, Rng& rng, Init init)
    : k_(k),
      stride_(stride),
      pad_(pad),
      W_(rng.normal({out_c, in_c, k, k}, conv_init_std(init, in_c * k * k))),
      b_(1, out_c),
      dW_({out_c, in_c, k, k}),
      db_(1, out_c) {}

Tensor Conv2D::forward(const Tensor& x) {
    if (x.rank() != 4) throw std::logic_error("Conv2D expects a rank-4 (N,C,H,W) input");
    x_shape_ = x.shape();
    const std::size_t N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    const std::size_t out_c = W_.dim(0);
    oh_ = (H + 2 * pad_ - k_) / stride_ + 1;
    ow_ = (W + 2 * pad_ - k_) / stride_ + 1;

    const std::size_t patch = C * k_ * k_;
    col_ = Tensor(N * oh_ * ow_, patch);
    const auto& xd = x.data();
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t i = 0; i < oh_; ++i)
            for (std::size_t j = 0; j < ow_; ++j) {
                const std::size_t row = (n * oh_ + i) * ow_ + j;
                for (std::size_t c = 0; c < C; ++c)
                    for (std::size_t ky = 0; ky < k_; ++ky)
                        for (std::size_t kx = 0; kx < k_; ++kx) {
                            const long h = static_cast<long>(i * stride_ + ky) - static_cast<long>(pad_);
                            const long w = static_cast<long>(j * stride_ + kx) - static_cast<long>(pad_);
                            double v = 0.0;
                            if (h >= 0 && h < static_cast<long>(H) && w >= 0 &&
                                w < static_cast<long>(W)) {
                                const std::size_t hh = static_cast<std::size_t>(h);
                                const std::size_t ww = static_cast<std::size_t>(w);
                                v = xd[((n * C + c) * H + hh) * W + ww];
                            }
                            col_(row, (c * k_ + ky) * k_ + kx) = v;
                        }
            }

    // Wc: (patch, out_c)
    Tensor Wc(patch, out_c);
    for (std::size_t oc = 0; oc < out_c; ++oc)
        for (std::size_t p = 0; p < patch; ++p) Wc(p, oc) = W_.data()[oc * patch + p];

    Tensor out_mat = matmul(col_, Wc);  // (N*oh*ow, out_c)

    Tensor out({N, out_c, oh_, ow_});
    auto& od = out.data();
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t oc = 0; oc < out_c; ++oc)
            for (std::size_t i = 0; i < oh_; ++i)
                for (std::size_t j = 0; j < ow_; ++j) {
                    const std::size_t row = (n * oh_ + i) * ow_ + j;
                    od[((n * out_c + oc) * oh_ + i) * ow_ + j] =
                        out_mat(row, oc) + b_.data()[oc];
                }
    return out;
}

Tensor Conv2D::backward(const Tensor& grad_out) {
    const std::size_t N = x_shape_[0], C = x_shape_[1], H = x_shape_[2], W = x_shape_[3];
    const std::size_t out_c = W_.dim(0);
    const std::size_t patch = C * k_ * k_;

    // g_mat: (N*oh*ow, out_c)
    Tensor g_mat(N * oh_ * ow_, out_c);
    const auto& gd = grad_out.data();
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t oc = 0; oc < out_c; ++oc)
            for (std::size_t i = 0; i < oh_; ++i)
                for (std::size_t j = 0; j < ow_; ++j) {
                    const std::size_t row = (n * oh_ + i) * ow_ + j;
                    g_mat(row, oc) = gd[((n * out_c + oc) * oh_ + i) * ow_ + j];
                }

    // db = sum over rows
    db_ = g_mat.sum_rows();  // (1, out_c)

    // dW = (g_mat^T @ col) reshaped to (out_c, in_c, k, k)
    Tensor dW_mat = matmul(g_mat.transpose(), col_);  // (out_c, patch)
    for (std::size_t oc = 0; oc < out_c; ++oc)
        for (std::size_t p = 0; p < patch; ++p) dW_.data()[oc * patch + p] = dW_mat(oc, p);

    // Wc^T: (out_c, patch); dcol = g_mat @ Wc^T -> (N*oh*ow, patch)
    Tensor WcT(out_c, patch);
    for (std::size_t oc = 0; oc < out_c; ++oc)
        for (std::size_t p = 0; p < patch; ++p) WcT(oc, p) = W_.data()[oc * patch + p];
    Tensor dcol = matmul(g_mat, WcT);  // (N*oh*ow, patch)

    // col2im: scatter-add back into dx
    Tensor dx({N, C, H, W});
    auto& dxd = dx.data();
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t i = 0; i < oh_; ++i)
            for (std::size_t j = 0; j < ow_; ++j) {
                const std::size_t row = (n * oh_ + i) * ow_ + j;
                for (std::size_t c = 0; c < C; ++c)
                    for (std::size_t ky = 0; ky < k_; ++ky)
                        for (std::size_t kx = 0; kx < k_; ++kx) {
                            const long h = static_cast<long>(i * stride_ + ky) - static_cast<long>(pad_);
                            const long w = static_cast<long>(j * stride_ + kx) - static_cast<long>(pad_);
                            if (h >= 0 && h < static_cast<long>(H) && w >= 0 &&
                                w < static_cast<long>(W)) {
                                const std::size_t hh = static_cast<std::size_t>(h);
                                const std::size_t ww = static_cast<std::size_t>(w);
                                dxd[((n * C + c) * H + hh) * W + ww] +=
                                    dcol(row, (c * k_ + ky) * k_ + kx);
                            }
                        }
            }
    return dx;
}

std::vector<ParamGrad> Conv2D::params_and_grads() {
    return {{&W_, &dW_}, {&b_, &db_}};
}

}  // namespace nn
