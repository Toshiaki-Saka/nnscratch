# API reference

Everything in `namespace nn`. Include `<nnscratch/nnscratch.hpp>` for the whole
public API; `<nnscratch/pgm.hpp>` is separate (a demo utility, not part of the
network API).

Complexity is quoted in element operations, with $N$ = batch size. All types are
`double`-valued.

| Section | Contents |
|---|---|
| [Tensor](#tensor) | storage, shape queries, element access, linear algebra |
| [Layer / ParamGrad](#layer--paramgrad) | the interface every layer implements |
| [Layers](#layers) | `Dense`, `Flatten`, `Conv2D`, `Init` |
| [Activations](#activations) | `ReLU`, `Tanh`, `Sigmoid`, `softmax()` |
| [Loss](#loss) | `SoftmaxCrossEntropy`, `one_hot()` |
| [Model](#model) | sequential container, prediction, accuracy |
| [Optimizers](#optimizers) | `SGD`, `Momentum`, `Adam` |
| [Training](#training) | `train()`, `TrainConfig`, `History` |
| [Dataset](#dataset) | `load_digits()`, `DigitsData` |
| [Rng](#rng) | reproducible random numbers |
| [PGM output](#pgm-output) | `write_pgm()`, `write_pgm_grid()` |
| [Version macros](#version-macros) | `NNSCRATCH_VERSION_*` |

---

## Tensor

`<nnscratch/tensor.hpp>` — a dense, row-major, `double` array with a shape
vector. The library needs only rank 2 (`(rows, cols)`) and rank 4
(`(N, C, H, W)`), so this is deliberately not a general n-d array type.

### Construction

```cpp
Tensor();                                                  // empty, rank 0
Tensor(std::size_t rows, std::size_t cols);                // zero-filled rank-2
explicit Tensor(std::vector<std::size_t> shape);           // zero-filled, any rank
static Tensor from(std::vector<std::size_t> shape, std::vector<double> data);
static Tensor zeros(std::vector<std::size_t> shape);
static Tensor zeros_like(const Tensor& t);
```

All constructors zero-initialise. `from()` takes ownership of an existing flat
buffer in row-major order and **throws `std::invalid_argument`** if
`prod(shape) != data.size()`.

```cpp
Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});  // [[1,2,3],[4,5,6]]
Tensor z = Tensor::zeros({4, 1, 8, 8});
```

### Shape queries

| Signature | Returns | Throws |
|---|---|---|
| `const std::vector<std::size_t>& shape() const` | full shape | — |
| `std::size_t rank() const` | number of dimensions | — |
| `std::size_t size() const` | total element count | — |
| `std::size_t dim(std::size_t i) const` | extent of axis `i` | `std::out_of_range` (uses `.at()`) |
| `std::size_t rows() const` | `shape()[0]` | `std::logic_error` if rank ≠ 2 |
| `std::size_t cols() const` | `shape()[1]` | `std::logic_error` if rank ≠ 2 |

### Element access

```cpp
double& operator()(std::size_t i, std::size_t j);
double  operator()(std::size_t i, std::size_t j) const;
std::vector<double>& data();
const std::vector<double>& data() const;
```

> **No bounds checking.** `operator()` computes `data_[i * shape_[1] + j]`
> directly, and does not verify the rank either — calling it on a rank-4 tensor
> silently indexes with the wrong stride. Use `data()` with an explicit offset
> for rank-4 access:
> ```cpp
> // element (n, c, h, w) of an (N, C, H, W) tensor
> t.data()[((n * C + c) * H + h) * W + w];
> ```

### Reshaping

```cpp
Tensor reshape(std::vector<std::size_t> new_shape) const;  // throws if size differs
Tensor flatten_batch() const;                               // (N, ...) -> (N, prod(rest))
```

Both **return a copy** — there are no views in this library. `flatten_batch`
throws `std::out_of_range` on a rank-0 tensor (it calls `shape_.at(0)`).

### Linear algebra

| Signature | Result | Complexity | Throws |
|---|---|---|---|
| `Tensor transpose() const` | $A^\top$ | $O(mn)$ | `std::logic_error` if rank ≠ 2 |
| `Tensor sum_rows() const` | $(1, \text{cols})$, summed over axis 0 | $O(mn)$ | `std::logic_error` if rank ≠ 2 |
| `Tensor map(const std::function<double(double)>&) const` | element-wise $f$ | $O(\text{size})$ | — |
| `void add_row_vector(const Tensor& bias)` | in-place broadcast add | $O(mn)$ | `logic_error` rank ≠ 2; `invalid_argument` width mismatch |
| `void axpy(double alpha, const Tensor& other)` | in-place `*this += alpha * other` | $O(\text{size})$ | `std::invalid_argument` on size mismatch |

`add_row_vector` accepts a bias of shape `(1, cols)` or `(cols)` — it only
checks `bias.size() == cols`.

`axpy` compares total *size*, not shape, so it will happily add a
`(1, 6)` to a `(2, 3)`.

### Free functions

```cpp
Tensor matmul(const Tensor& a, const Tensor& b);   // (m,k) x (k,n) -> (m,n)

Tensor operator+(const Tensor& a, const Tensor& b);  // element-wise, shapes must match exactly
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);  // Hadamard
Tensor operator*(double s, const Tensor& a);         // scalar scale (scalar on the LEFT)

std::ostream& operator<<(std::ostream& os, const Tensor& t);  // prints shape only
```

`matmul` is a naive $O(mkn)$ triple loop in `ikj` order (see
[PERFORMANCE.md](PERFORMANCE.md#matmul)). It throws `std::logic_error` if either
operand is not rank 2 and `std::invalid_argument` if the inner dimensions
disagree.

The element-wise operators require **identical shapes** — no broadcasting. Use
`add_row_vector` for bias-style broadcasting.

Only `double * Tensor` exists, not `Tensor * double`: `Tensor * Tensor` is the
Hadamard product, and allowing both scalar orders next to it invites confusion.

`operator<<` prints `Tensor(shape=[2,3])` — the shape, never the values.

---

## Layer / ParamGrad

`<nnscratch/layer.hpp>`

```cpp
struct ParamGrad {
    Tensor* param;   // owned by the layer; the optimizer mutates it in place
    Tensor* grad;    // the matching gradient, overwritten by each backward()
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad_out) = 0;
    virtual std::vector<ParamGrad> params_and_grads() { return {}; }
};
```

`param`'s **address** is the optimizer's key for per-parameter state — see
[ARCHITECTURE.md](ARCHITECTURE.md#ownership-and-lifetimes) for the lifetime
rules that come with that.

To implement your own layer, see [EXTENDING.md](EXTENDING.md).

---

## Layers

`<nnscratch/layers.hpp>`

### Init

```cpp
enum class Init { He, Xavier };
```

| Value | `Dense` $\sigma$ | `Conv2D` $\sigma$ | Pair with |
|---|---|---|---|
| `He` | $\sqrt{2/n_{\mathrm{in}}}$ | $\sqrt{2/n_{\mathrm{in}}}$ | ReLU |
| `Xavier` | $\sqrt{2/(n_{\mathrm{in}} + n_{\mathrm{out}})}$ | $\sqrt{1/n_{\mathrm{in}}}$ | Tanh, Sigmoid |

For `Conv2D`, $n_{\mathrm{in}} = C_{\mathrm{in}} K^2$. Note the two `Xavier`
formulas differ — see the implementation note in
[MATH.md](MATH.md#weight-initialisation). Biases always start at zero.

### Dense

```cpp
Dense(std::size_t n_in, std::size_t n_out, Rng& rng, Init init = Init::Xavier);

Tensor forward(const Tensor& x) override;         // (N, n_in)  -> (N, n_out)
Tensor backward(const Tensor& grad_out) override; // (N, n_out) -> (N, n_in)
std::vector<ParamGrad> params_and_grads() override;  // {{&W, &dW}, {&b, &db}}

Tensor& weight() noexcept;  // (n_in, n_out)
Tensor& bias()   noexcept;  // (1, n_out)
```

$Y = XW + b$. Parameters: $n_{\mathrm{in}} n_{\mathrm{out}} + n_{\mathrm{out}}$.
Complexity $O(N \cdot n_{\mathrm{in}} \cdot n_{\mathrm{out}})$ forward, the same
up to a constant backward. Caches the input by value.

`weight()` returns a mutable reference, which is how `from_scratch.cpp` renders
the first layer's learned features. Writing through it is allowed and is the
only way to set weights manually (there is no serialisation).

### Flatten

```cpp
Tensor forward(const Tensor& x) override;         // (N, d1, ..., dk) -> (N, prod di)
Tensor backward(const Tensor& grad_out) override; // restores the cached input shape
```

No parameters. $O(\text{size})$ (a copy). Required between `Conv2D` and `Dense`.

### Conv2D

```cpp
Conv2D(std::size_t in_c, std::size_t out_c, std::size_t k,
       std::size_t stride, std::size_t pad, Rng& rng, Init init = Init::He);

Tensor forward(const Tensor& x) override;   // (N,C,H,W) -> (N,out_c,OH,OW)
Tensor backward(const Tensor& grad_out) override;
std::vector<ParamGrad> params_and_grads() override;  // {{&W, &dW}, {&b, &db}}

Tensor& weight() noexcept;  // (out_c, in_c, k, k)
```

Output geometry:

```math
OH = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1, \qquad
OW = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1
```

- Parameters: $C_{\mathrm{out}}(C_{\mathrm{in}}K^2 + 1)$.
- Forward: $O(N \cdot OH \cdot OW \cdot C_{\mathrm{in}} K^2 \cdot
  C_{\mathrm{out}})$ time, plus an $N \cdot OH \cdot OW \cdot C_{\mathrm{in}}
  K^2$-element im2col buffer that is cached for the backward pass.
- Zero padding only; square kernels only; the same stride on both axes.

**Preconditions (not all enforced):**

| Condition | Behaviour if violated |
|---|---|
| input rank = 4 | throws `std::logic_error` |
| `stride >= 1` | division by zero |
| `k <= H + 2*pad` and `k <= W + 2*pad` | **unsigned underflow** — `H + 2P - K` wraps to a huge value, giving an absurd `OH` and a failed allocation rather than a clear error |
| `x.dim(1) == in_c` | no check; produces a wrong-shaped `col_` and a `matmul` inner-dimension error |

`forward()` must have been called before `backward()`: the backward pass reads
the cached `col_`, `x_shape_`, `oh_`, `ow_`.

---

## Activations

`<nnscratch/activations.hpp>`

```cpp
Tensor softmax(const Tensor& logits);   // (N, K) -> (N, K), rows sum to 1

class ReLU    final : public Layer { /* forward, backward */ };
class Tanh    final : public Layer { /* forward, backward */ };
class Sigmoid final : public Layer { /* forward, backward */ };
```

All three are element-wise, shape-preserving, parameter-free, and work at any
rank. $O(\text{size})$ each way.

| Class | $f(x)$ | Cache | Backward |
|---|---|---|---|
| `ReLU` | $\max(0, x)$ | 0/1 mask | $G \odot \text{mask}$ |
| `Tanh` | $\tanh x$ | output $y$ | $G \odot (1 - y^2)$ |
| `Sigmoid` | $1/(1+e^{-x})$ | output $y$ | $G \odot y(1-y)$ |

At $x = 0$ `ReLU` uses the sub-gradient $0$, matching PyTorch and TensorFlow.

The free `softmax()` requires rank 2 (it calls `rows()`/`cols()`) and subtracts
each row's max for stability. You rarely need it directly: for training, use
`SoftmaxCrossEntropy`, which fuses it with the loss. Call it when you want
calibrated probabilities out of a trained model:

```cpp
Tensor probs = softmax(model.forward(x_test));   // (N, 10), each row sums to 1
```

---

## Loss

`<nnscratch/loss.hpp>`

```cpp
class SoftmaxCrossEntropy {
public:
    double forward(const Tensor& logits, const Tensor& targets_onehot);
    Tensor backward() const;   // (P - Y) / N
};

Tensor one_hot(const std::vector<int>& labels, std::size_t num_classes);
```

`forward` takes **logits, not probabilities** — the softmax happens inside — and
returns the mean cross-entropy over the batch. It caches the probabilities and
targets, so `backward()` takes no arguments and may only be called afterwards.
$O(NK)$ each way. See [MATH.md](MATH.md#softmax--cross-entropy-fused) for why
the fused gradient is just $P - Y$ over $N$.

The reported loss uses $\log(p + 10^{-9})$, so a confidently-wrong prediction
reports about 20.7 rather than infinity. The gradient is unaffected.

`one_hot` builds an $(\text{labels.size}, K)$ tensor. **Labels must lie in
$[0, K)$** — there is no bounds check.

```cpp
SoftmaxCrossEntropy loss_fn;
Tensor Y = one_hot(y_train, 10);
double L = loss_fn.forward(model.forward(x_train), Y);
model.backward(loss_fn.backward());
```

---

## Model

`<nnscratch/model.hpp>` — a sequential stack, the analogue of `nn.Sequential`.

```cpp
Model();
explicit Model(std::vector<std::unique_ptr<Layer>> layers);

template <class L, class... Args> L& add(Args&&... args);  // construct in place
Layer& push(std::unique_ptr<Layer> layer);                 // append an existing layer

Tensor forward(const Tensor& x);
void   backward(const Tensor& grad);
std::vector<ParamGrad> params_and_grads();

std::vector<int> predict(const Tensor& x);                       // argmax per row
double accuracy(const Tensor& x, const std::vector<int>& labels);

Layer& layer(std::size_t i);          // throws std::out_of_range
std::size_t size() const noexcept;
```

`add<L>(...)` forwards its arguments to `L`'s constructor and returns a
reference to the constructed layer — valid for the model's lifetime:

```cpp
Rng rng(42);
Model net;
Dense& first = net.add<Dense>(64, 64, rng, Init::He);   // keep the reference
net.add<ReLU>();
net.add<Dense>(64, 10, rng, Init::He);
// ... train ...
const Tensor& W = first.weight();                        // still valid
```

`push()` is for cases where the layer type is decided at runtime —
`compare.cpp` uses it with an activation factory so the same builder can produce
ReLU, Tanh and Sigmoid variants.

`backward(grad)` walks the layers in reverse and discards the final gradient
(nobody needs $\partial L/\partial\text{input}$). It does not return a value.

`predict` and `accuracy` each run a full forward pass; calling both costs two.
`accuracy` assumes `labels.size() == x.dim(0)`.

`Model` is **movable but not copyable** — `Layer` has no `clone()`.

---

## Optimizers

`<nnscratch/optimizer.hpp>`

```cpp
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<ParamGrad>& pgs) = 0;
};

class SGD      final : public Optimizer { explicit SGD(double lr); };
class Momentum final : public Optimizer { explicit Momentum(double lr, double mu = 0.9); };
class Adam     final : public Optimizer { explicit Adam(double lr, double b1 = 0.9,
                                                        double b2 = 0.999, double eps = 1e-8); };
```

| | Update | State per parameter | Typical `lr` here |
|---|---|---|---|
| `SGD` | $p \mathrel{-}= \eta g$ | none | 0.2 – 0.5 |
| `Momentum` | $v \leftarrow \mu v - \eta g$; $p \mathrel{+}= v$ | one velocity tensor | 0.05 |
| `Adam` | see [MATH.md](MATH.md#adam) | two moment tensors + a shared step counter | 0.01 |

Momentum's effective step is roughly $1/(1-\mu)$ times SGD's, which is why the
learning rates that make a fair comparison differ by about that factor in
`compare.cpp`.

`step()` applies one update to every pair and is the only mutating call. It
creates per-parameter state lazily, keyed on `param`'s address.

> **Do not reuse a `Momentum` or `Adam` instance across different models.**
> Address-keyed state can collide with recycled heap addresses after a model is
> destroyed, silently carrying stale moments into a fresh run. Construct a new
> optimizer per model.

Adam's step counter `t_` increments once per `step()` call (not per parameter),
which is what makes bias correction correct.

---

## Training

`<nnscratch/training.hpp>`

```cpp
struct TrainConfig {
    int          epochs     = 40;
    std::size_t  batch_size = 32;
    std::uint64_t batch_seed = 123;   // fixes the mini-batch order
    bool         verbose    = false;  // print per-epoch metrics to stdout
};

struct History {
    std::vector<int>    epoch;
    std::vector<double> loss;        // mean CE on the FULL training set
    std::vector<double> train_acc;
    std::vector<double> test_acc;
};

History train(Model& model, Optimizer& opt,
              const Tensor& x_train, const std::vector<int>& y_train,
              const Tensor& x_test,  const std::vector<int>& y_test,
              const TrainConfig& cfg);
```

Mini-batch training with `SoftmaxCrossEntropy` as the fixed loss. Behaviour
worth knowing:

- **`History` has `epochs + 1` rows.** Epoch 0 records the untrained network
  before any update, giving the full "random → trained" trajectory.
- **Metrics are computed on the full training set after each epoch**, not
  averaged over mini-batches, so `loss` is the true epoch-end loss. This costs
  two extra full forward passes per epoch and overwrites every layer's forward
  cache — harmless, because no `backward()` follows.
- **`num_classes` is inferred** as `max(y_train) + 1`. A class absent from the
  training labels shrinks the output width, so the shape must still match your
  final `Dense`.
- **The last mini-batch of an epoch may be smaller** than `batch_size`.
- **Shuffling uses a `Rng` local to the call**, seeded from `cfg.batch_seed`, so
  the batch order is independent of the model's initialisation RNG.
- `x_train` may be rank 2 or rank 4 — batching preserves the trailing
  dimensions, so the same loop trains MLPs and CNNs.

```cpp
TrainConfig cfg;
cfg.epochs = 60;
cfg.verbose = true;
History h = train(net, opt, data.train.flat, data.train.labels,
                  data.test.flat, data.test.labels, cfg);
std::printf("best test acc: %.1f%%\n",
            *std::max_element(h.test_acc.begin(), h.test_acc.end()) * 100.0);
```

---

## Dataset

`<nnscratch/dataset.hpp>`

```cpp
struct DigitsData {
    struct Split {
        Tensor flat;              // (N, 64)      for MLPs
        Tensor img;               // (N, 1, 8, 8) for CNNs
        std::vector<int> labels;  // N entries, 0..9
    };
    Split train;
    Split test;
};

DigitsData load_digits(const std::string& csv_path,
                       double train_frac = 0.8,
                       std::uint64_t split_seed = 0);
```

Both views hold the *same* samples, so you can switch between an MLP and a CNN
without reloading. Pixels are normalised from the source range 0–16 to $[0, 1]$
by dividing by 16.

The split is a Fisher–Yates permutation seeded by `split_seed`, then a cut at
`floor(N * train_frac)`. With the bundled 1797-record file and the defaults that
is **1437 train / 360 test**.

Throws `std::runtime_error` if the file cannot be opened, contains no records,
or has a row that is not exactly 65 fields. Comment lines (`#`) and the `p0,...`
header row are skipped. Full format spec: [DATA_FORMATS.md](DATA_FORMATS.md).

---

## Rng

`<nnscratch/rng.hpp>` — header-only wrapper around `std::mt19937_64`.

```cpp
explicit Rng(std::uint64_t seed = 42);
void reseed(std::uint64_t seed);

double normal();                                              // one N(0,1) sample
Tensor normal(std::vector<std::size_t> shape, double std_dev); // N(0, std^2) tensor
std::vector<std::size_t> permutation(std::size_t n);           // Fisher-Yates, O(n)
```

`reseed()` is what makes the comparison experiments fair — rewinding to the same
seed before building each model gives identical starting weights:

```cpp
Rng rng(42);
rng.reseed(42);  Model a = build_mlp(rng, relu);
rng.reseed(42);  Model b = build_mlp(rng, tanh_);   // same initial weights as a
```

Results are bit-reproducible for a fixed toolchain but **not across toolchains**:
the standard specifies the engine exactly, but not
`std::normal_distribution` / `std::uniform_int_distribution`. See
[ARCHITECTURE.md](ARCHITECTURE.md#determinism).

---

## PGM output

`<nnscratch/pgm.hpp>` — header-only, used by the demos, not included by
`nnscratch.hpp`.

```cpp
void write_pgm(const std::string& path, const std::vector<double>& pixels,
               std::size_t width, std::size_t height);

void write_pgm_grid(const std::string& path,
                    const std::vector<std::vector<double>>& cells,
                    std::size_t cell, std::size_t cols, std::size_t pad = 1);
```

`write_pgm` writes binary PGM (`P5`), clamping values to $[0,1]$ and quantising
to 0–255. `write_pgm_grid` tiles equally-sized square images into one canvas,
**min–max normalising each cell independently** so that structure is visible
regardless of the weights' absolute scale — meaning brightness is comparable
*within* a cell, never *between* cells.

Neither function reports I/O errors; a failed open silently writes nothing.
Layout details and viewing tips: [DATA_FORMATS.md](DATA_FORMATS.md#pgm-images).

---

## Version macros

```cpp
#define NNSCRATCH_VERSION_MAJOR 0
#define NNSCRATCH_VERSION_MINOR 1
#define NNSCRATCH_VERSION_PATCH 0
```

Defined by `<nnscratch/nnscratch.hpp>` and kept in sync with `project(VERSION)`
in `CMakeLists.txt`. The installed CMake package version file uses
`SameMajorVersion` compatibility, so at 0.x every minor bump is treated as
incompatible.
