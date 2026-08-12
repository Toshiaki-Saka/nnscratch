# Extending nnscratch

There is no autograd here, so a new layer means a hand-derived `backward()` —
and a gradient check that proves it. This document walks through that, then
covers optimizers, losses and datasets.

Prerequisites: the chain-rule conventions in [MATH.md](MATH.md) and the layer
contract in [ARCHITECTURE.md](ARCHITECTURE.md#layer-abstraction).

- [Adding a layer](#adding-a-layer)
- [Worked example: LeakyReLU](#worked-example-leakyrelu)
- [Worked example: MaxPool2D](#worked-example-maxpool2d)
- [Adding an optimizer](#adding-an-optimizer)
- [Adding a loss](#adding-a-loss)
- [Adding a dataset](#adding-a-dataset)
- [House rules](#house-rules)

---

## Adding a layer

Five steps. Step 4 is the one that matters.

### Step 1: derive the backward pass on paper

Write the forward map in indices, then apply

$$\frac{\partial L}{\partial \theta} = \sum_k \frac{\partial L}{\partial Y_k}\frac{\partial Y_k}{\partial \theta}$$

summing over **every** output element that $\theta$ influenced. Collect the sum
into a matrix product, a reduction, or a scatter-add. Then sanity-check by
shape: $\partial L/\partial W$ must have $W$'s shape, and usually only one
contraction produces it.

Remember that the $1/N$ batch average lives in the loss, so a layer never
divides by $N$ itself.

### Step 2: declare it

Add the class to `include/nnscratch/layers.hpp` (parametric layers) or
`activations.hpp` (element-wise). Members hold parameters, their gradients, and
whatever the backward pass needs cached:

```cpp
class MyLayer final : public Layer {
public:
    MyLayer(std::size_t n_in, std::size_t n_out, Rng& rng, Init init = Init::He);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamGrad> params_and_grads() override;   // omit if no parameters

private:
    Tensor W_, b_;     // parameters
    Tensor dW_, db_;   // gradients, same shapes
    Tensor cache_;     // whatever backward() needs
};
```

Cache the *smallest* sufficient thing: a 0/1 mask beats the input, and the
output beats the input when the derivative can be written in terms of it (as for
Tanh and Sigmoid).

### Step 3: implement it

In `src/layers.cpp`, `src/activations.cpp`, or its own file if it is long —
`Conv2D` has `src/conv2d.cpp` to itself. A new `.cpp` must be added to the
`add_library(nnscratch ...)` list in `CMakeLists.txt`.

```cpp
Tensor MyLayer::forward(const Tensor& x) {
    if (x.rank() != 2) throw std::logic_error("MyLayer expects a rank-2 input");
    cache_ = /* ... */;   // cache BEFORE any in-place modification
    return /* ... */;
}

Tensor MyLayer::backward(const Tensor& grad_out) {
    dW_ = /* ... */;      // assign, never accumulate
    db_ = /* ... */;
    return /* grad w.r.t. input, shaped exactly like forward()'s x */;
}

std::vector<ParamGrad> MyLayer::params_and_grads() {
    return {{&W_, &dW_}, {&b_, &db_}};
}
```

Non-negotiables:

- **Assign gradients, don't accumulate.** The library has no `zero_grad()`;
  `+=` would silently accumulate across steps
  ([ARCHITECTURE.md](ARCHITECTURE.md#invariants)).
- **Return a tensor shaped exactly like the input** to `forward()`.
- **Return stable pointers** from `params_and_grads()` — always `&member_`,
  never the address of a local or of an element in a vector that may reallocate.
- **Validate the rank** you require, and throw. A silent wrong-stride read is
  much harder to debug than an exception.

### Step 4: prove it with a gradient check

**This step is mandatory** ([CONTRIBUTING.md](../CONTRIBUTING.md)). Add a block
to `tests/test_gradcheck.cpp`:

```cpp
{
    Rng rng(17);
    Model m;
    m.add<Dense>(6, 5, rng, Init::Xavier);
    m.add<MyLayer>(5, 4, rng, Init::He);   // the layer under test
    m.add<Dense>(4, 3, rng, Init::Xavier);

    Tensor X = rng.normal({4, 6}, 1.0);
    std::vector<int> y = {0, 2, 1, 1};
    grad_check(m, X, y, 3);
}
```

Guidelines:

- **Sandwich it between `Dense` layers.** Then a non-zero gradient must have
  flowed *through* your layer to reach the first `Dense`, so both the parameter
  gradients and the input gradient are tested.
- **Keep it tiny** — 4–6 units, 3–5 samples. Every probe is two full forward
  passes.
- **Cover the awkward configurations.** If your layer has a stride, a padding
  or a rate, test a non-default value; the default is usually the case where
  bugs hide (see the `Conv2D` note in
  [TESTING.md](TESTING.md#coverage-gaps)).
- **If your layer has kinks** (anything piecewise, like ReLU), pick a seed where
  no pre-activation lands within $10^{-5}$ of a kink, or the finite difference
  will straddle two linear pieces and disagree legitimately
  ([MATH.md](MATH.md#the-kink-caveat)).

Then run it and read the failure table in
[TESTING.md](TESTING.md#debugging-a-failure) — the ratio between the analytic
and numeric values normally names the bug.

### Step 5: document the correspondence

Add a row to the `numpy → C++ → framework` table in the top-level
[README.md](../README.md#numpy--c--framework-correspondence). Mapping each piece
back to its framework equivalent is the project's stated purpose, so a layer
without that row is only half-delivered.

---

## Worked example: LeakyReLU

The smallest complete example: element-wise, no parameters.

**Math.** $f(x) = x$ for $x > 0$, $\alpha x$ otherwise; $f'(x) = 1$ or $\alpha$.
Element-wise, so the Jacobian is diagonal and backward is a Hadamard product —
exactly ReLU's structure with $0$ replaced by $\alpha$.

`include/nnscratch/activations.hpp`:

```cpp
/// LeakyReLU: x for x > 0, alpha * x otherwise. Keeps a small gradient alive
/// on the negative side, so units cannot become permanently dead.
class LeakyReLU final : public Layer {
public:
    explicit LeakyReLU(double alpha = 0.01) : alpha_(alpha) {}
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

private:
    double alpha_;
    Tensor slope_;   // 1 where x > 0, else alpha
};
```

`src/activations.cpp`:

```cpp
Tensor LeakyReLU::forward(const Tensor& x) {
    const double a = alpha_;
    slope_ = x.map([a](double v) { return v > 0.0 ? 1.0 : a; });
    return x * slope_;          // Hadamard: exact for both branches
}

Tensor LeakyReLU::backward(const Tensor& grad_out) {
    return grad_out * slope_;
}
```

No `params_and_grads()` override — the base class returns an empty list, which
is correct for a layer with no learnable state.

Test block in `tests/test_gradcheck.cpp`:

```cpp
{
    Rng rng(23);
    Model m;
    m.add<Dense>(6, 5, rng, Init::He);
    m.add<LeakyReLU>(0.1);      // exaggerated alpha: a wrong branch is obvious
    m.add<Dense>(5, 3, rng, Init::Xavier);

    Tensor X = rng.normal({4, 6}, 1.0);
    std::vector<int> y = {1, 0, 2, 1};
    grad_check(m, X, y, 3);
}
```

Using $\alpha = 0.1$ rather than the 0.01 default is deliberate: if backward
took the wrong branch, the discrepancy is 10× rather than 100× below the
signal — but more importantly, an $\alpha$ that is *too* small makes the
negative-side gradient so tiny that the finite-difference noise floor could
mask an error.

---

## Worked example: MaxPool2D

Sketch of a harder case — rank-4, no parameters, but a *routing* backward pass.

**Math.** Forward takes the maximum over each $K \times K$ window. Only the
argmax contributes to the output, so

```math
\frac{\partial L}{\partial X_{n,c,h,w}} =
\sum_{\text{windows } (i,j) \text{ whose argmax is } (h,w)} G_{n,c,i,j}
```

Every non-maximal element gets exactly zero. So:

- `forward` caches the **flat index of the argmax** per output position (one
  `std::size_t` per output element — much cheaper than caching the input).
- `backward` allocates a zero tensor of the input's shape and scatter-adds each
  upstream gradient into its stored argmax position. The `+=` matters as soon as
  windows overlap (stride < K), for the same reason it does in col2im
  ([MATH.md](MATH.md#col2im-why-scatter-add)).

Two traps specific to pooling:

- **Ties.** If two elements in a window are exactly equal, the true function is
  non-differentiable there. Pick one deterministically (the first, as
  frameworks do) and use fixed-seed random inputs in the gradient check so ties
  do not arise.
- **The kink.** Max is piecewise linear, so the gradient check straddles a
  boundary if two elements in a window are within $\epsilon$ of each other.
  Same mitigation as ReLU.

Add it to `layers.hpp` alongside `Conv2D`, implement it in its own `.cpp` if it
grows past a screen, register that file in `CMakeLists.txt`, and gradient-check
it with both stride $=K$ (no overlap) and stride $< K$ (overlap) — those are
genuinely different code paths.

---

## Adding an optimizer

Much easier: no derivation, and the gradient check does not apply.

```cpp
// include/nnscratch/optimizer.hpp
class RMSProp final : public Optimizer {
public:
    explicit RMSProp(double lr, double rho = 0.9, double eps = 1e-8)
        : lr_(lr), rho_(rho), eps_(eps) {}
    void step(const std::vector<ParamGrad>& pgs) override;

private:
    double lr_, rho_, eps_;
    std::unordered_map<const Tensor*, Tensor> s_;   // keyed by parameter ADDRESS
};
```

```cpp
// src/optimizer.cpp
void RMSProp::step(const std::vector<ParamGrad>& pgs) {
    for (const auto& [p, g] : pgs) {
        auto it = s_.find(p);
        if (it == s_.end()) it = s_.emplace(p, Tensor::zeros_like(*p)).first;
        Tensor& s = it->second;
        for (std::size_t i = 0; i < s.size(); ++i) {
            const double gi = g->data()[i];
            s.data()[i] = rho_ * s.data()[i] + (1.0 - rho_) * gi * gi;
            p->data()[i] -= lr_ * gi / (std::sqrt(s.data()[i]) + eps_);
        }
    }
}
```

Points to respect:

- **Lazy state creation, keyed on the parameter's address.** Copy the pattern
  above verbatim; it is what lets an optimizer work with any model without being
  told the parameter list up front.
- **A step counter, if you need bias correction, belongs to the optimizer, not
  the parameter** — increment it once per `step()` call, as `Adam` does.
- **Update in place.** `axpy()` is available for the simple `p += alpha * g`
  case; anything element-wise-nonlinear needs the explicit loop.

Then add a block to `tests/test_optimizer.cpp` minimising $(w-3)^2$, and a row
to the optimizer table in [experiments.md](experiments.md).

---

## Adding a loss

`SoftmaxCrossEntropy` is deliberately **not** a `Layer` — it takes targets and
returns a scalar, so it has its own shape of interface:

```cpp
double forward(const Tensor& predictions, const Tensor& targets);
Tensor backward() const;   // d(loss)/d(predictions), cached from forward
```

Follow the same shape for a new loss, and consider whether fusing the final
activation into it simplifies the gradient — that is exactly what makes
softmax + cross-entropy collapse to $P - Y$
([MATH.md](MATH.md#the-collapse)).

Note that `train()` hard-codes `SoftmaxCrossEntropy`. Using a different loss
means writing your own loop; `src/training.cpp` is 86 lines and is meant to be
copied and adapted.

---

## Adding a dataset

Match `load_digits`'s shape: return the same samples in **both** a rank-2 view
(for MLPs) and a rank-4 view (for CNNs), plus `std::vector<int>` labels, split
by a seeded permutation so the split is reproducible.

```cpp
struct MyData {
    struct Split { Tensor flat, img; std::vector<int> labels; };
    Split train, test;
};
MyData load_mine(const std::string& path, double train_frac = 0.8,
                 std::uint64_t split_seed = 0);
```

Normalise inputs to roughly $[0,1]$ or zero-mean/unit-variance — the
initialisation formulas assume inputs of order 1
([MATH.md](MATH.md#weight-initialisation)) — and throw `std::runtime_error`
with the path in the message on any I/O or format problem, as `load_digits`
does.

Keep it dependency-free: the bundled loader is a `std::ifstream` and
`std::stod`, nothing more.

---

## House rules

From [CONTRIBUTING.md](../CONTRIBUTING.md), restated because they shape what an
acceptable extension looks like:

1. **No runtime dependencies.** C++20 standard library only — for the library,
   the tests and the demos alike.
2. **Clarity beats cleverness.** This is a teaching library. A readable loop
   that mirrors the math is worth more than a fast one that obscures it; if you
   want to know how much performance that costs, [PERFORMANCE.md](PERFORMANCE.md)
   quantifies it.
3. **Every new layer is exercised by `test_gradcheck`.** If `backward()` cannot
   pass a finite-difference check, it is not correct.
4. **Format before committing:** `clang-format -i $(git ls-files '*.cpp' '*.hpp')`.
   CI fails on unformatted code.
5. **Warnings are errors** (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion
   -Wsign-conversion -Werror`, `/W4 /WX` on MSVC). Expect to write explicit
   `static_cast<std::size_t>` and `static_cast<double>` conversions; the
   existing code does, and that is why it is warning-clean on four toolchains.
