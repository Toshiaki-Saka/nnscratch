# Design notes

This document explains the few non-obvious engineering choices in nnscratch.

## A single `Tensor` type

The library only needs rank-2 tensors `(rows, cols)` for the matrix math and
rank-4 tensors `(N, C, H, W)` for convolutions. Rather than introduce a general
n-dimensional array abstraction, `Tensor` keeps one flat, row-major
`std::vector<double>` and a small `shape` vector. This mirrors the "numpy only"
spirit of the original reference while staying dependency-free, and it makes the
`reshape`/`flatten` operations that connect `Conv2D` → `Flatten` → `Dense`
zero-copy at the conceptual level.

## The layer interface

Every layer implements:

```cpp
Tensor forward(const Tensor& x);          // cache what backward needs
Tensor backward(const Tensor& grad_out);  // return grad w.r.t. input
std::vector<ParamGrad> params_and_grads(); // expose weights + their gradients
```

A `ParamGrad` is a pair of pointers `{Tensor* param, Tensor* grad}`. The
optimizer mutates `*param` in place and uses the *address* of `param` as a
stable key for per-parameter state (the velocity in Momentum, the moment
estimates in Adam). This is the C++ analogue of the reference code keying its
optimizer state on Python's `id(p)`.

## Fused softmax + cross-entropy

`SoftmaxCrossEntropy` deliberately fuses the softmax activation with the
cross-entropy loss. Done together, the gradient of the loss with respect to the
logits collapses to the famously simple `(prediction - target) / N`, avoiding a
separate (and numerically delicate) softmax-Jacobian multiply. The softmax used
internally subtracts each row's maximum before exponentiating for numerical
stability — mathematically a no-op.

## Convolution via im2col

`Conv2D` lowers convolution to a single matrix multiply using the classic
im2col transformation: each output position's receptive field is unrolled into a
row, so the whole layer becomes `col @ W`. The backward pass reverses this with
col2im, scatter-adding gradients back into the input. This keeps the
convolution's forward and backward expressible in terms of the same `matmul` the
rest of the library uses, and it is exactly the lowering PyTorch/TF kernels
perform under the hood.

## Reproducibility

`Rng` wraps a `std::mt19937_64`. The comparison demo reseeds it to a fixed value
immediately before constructing each model so competing configurations start
from identical random weights — otherwise the experiment would be confounded by
initialization luck rather than the variable under study. The mini-batch order
is driven by a *separate* fixed seed so the only thing that differs between runs
is the axis being compared.

## Correctness: gradient checking

The decisive test is `tests/test_gradcheck.cpp`. For every learnable parameter
it compares the analytic gradient from `backward()` against a central-difference
estimate `(L(w+eps) - L(w-eps)) / (2*eps)`. If any layer's hand-derived backward
pass is wrong — including the convolution, which is the easiest to get wrong —
this test fails. Passing it is the evidence that the "from scratch" math is
actually correct.
