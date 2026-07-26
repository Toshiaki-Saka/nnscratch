// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#include "nnscratch/activations.hpp"
#include "nnscratch/tensor.hpp"
#include "check.hpp"

using nn::Tensor;

int main() {
    // matmul: [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = Tensor::from({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor c = nn::matmul(a, b);
    CHECK(c.rows() == 2 && c.cols() == 2);
    CHECK_CLOSE(c(0, 0), 58.0, 1e-12);
    CHECK_CLOSE(c(0, 1), 64.0, 1e-12);
    CHECK_CLOSE(c(1, 0), 139.0, 1e-12);
    CHECK_CLOSE(c(1, 1), 154.0, 1e-12);

    // transpose
    Tensor at = a.transpose();
    CHECK(at.rows() == 3 && at.cols() == 2);
    CHECK_CLOSE(at(2, 1), 6.0, 1e-12);

    // sum_rows
    Tensor s = a.sum_rows();
    CHECK(s.rows() == 1 && s.cols() == 3);
    CHECK_CLOSE(s(0, 0), 5.0, 1e-12);
    CHECK_CLOSE(s(0, 2), 9.0, 1e-12);

    // add_row_vector broadcasts bias across rows
    Tensor d = Tensor::from({2, 3}, {1, 1, 1, 2, 2, 2});
    d.add_row_vector(Tensor::from({1, 3}, {10, 20, 30}));
    CHECK_CLOSE(d(0, 1), 21.0, 1e-12);
    CHECK_CLOSE(d(1, 2), 32.0, 1e-12);

    // softmax rows sum to 1 and are monotone in the logits
    Tensor logits = Tensor::from({1, 3}, {1.0, 2.0, 3.0});
    Tensor p = nn::softmax(logits);
    CHECK_CLOSE(p(0, 0) + p(0, 1) + p(0, 2), 1.0, 1e-12);
    CHECK(p(0, 2) > p(0, 1) && p(0, 1) > p(0, 0));

    // reshape preserves data, changes shape
    Tensor r = a.reshape({3, 2});
    CHECK(r.rows() == 3 && r.cols() == 2);
    CHECK_CLOSE(r(0, 0), 1.0, 1e-12);
    CHECK_CLOSE(r(2, 1), 6.0, 1e-12);

    return nntest::summary("tensor");
}
