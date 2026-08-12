# Architecture

How the pieces fit together: what depends on what, what happens during one
training step, who owns which memory, and which invariants you must not break.

For *why* the design is shaped this way, see [DESIGN.md](DESIGN.md). For the
math inside each layer, see [MATH.md](MATH.md). For signatures, see
[API.md](API.md).

- [Module map](#module-map)
- [Layer abstraction](#layer-abstraction)
- [Life of one training step](#life-of-one-training-step)
- [Shape contracts](#shape-contracts)
- [Ownership and lifetimes](#ownership-and-lifetimes)
- [Invariants](#invariants)
- [Error handling](#error-handling)
- [Determinism](#determinism)
- [What is deliberately absent](#what-is-deliberately-absent)

---

## Module map

Ten headers, nine translation units, no cycles. Everything bottoms out at
`Tensor`.

```
                        tensor.hpp
             (the only truly foundational type)
                            |
        +-------------------+--------------------+---------+
        |                   |                    |         |
     rng.hpp            layer.hpp             loss.hpp   pgm.hpp
   (seeded PRNG)   (Layer, ParamGrad)      (softmax+CE)  (header-only,
        |                   |                    |        apps only)
        |          +--------+--------+           |
        |          |                 |           |
        +----> layers.hpp     activations.hpp    |
        |    (Dense, Conv2D,   (ReLU, Tanh,      |
        |     Flatten)          Sigmoid,         |
        |          |            softmax())       |
        |          |                 |           |
        |          +--------+--------+           |
        |                   |                    |
        |               model.hpp                |
        |          (owns a layer stack)          |
        |                   |                    |
        |            optimizer.hpp               |
        |         (SGD/Momentum/Adam)            |
        |                   |                    |
        +-------------> training.hpp <-----------+
        |             (the epoch loop)
        |
    dataset.hpp
  (digits.csv loader)

  nnscratch.hpp  =  umbrella header, includes all of the above except pgm.hpp
```

| Header | Translation unit | Role |
|---|---|---|
| `tensor.hpp` | `src/tensor.cpp` | Flat row-major storage, `matmul`, element-wise ops, reshape/transpose |
| `rng.hpp` | header-only | `std::mt19937_64` wrapper: normals, permutations, reseeding |
| `layer.hpp` | header-only | The `Layer` interface and `ParamGrad` |
| `layers.hpp` | `src/layers.cpp`, `src/conv2d.cpp` | `Dense`, `Flatten`, `Conv2D`, `Init` |
| `activations.hpp` | `src/activations.cpp` | `ReLU`, `Tanh`, `Sigmoid`, free `softmax()` |
| `loss.hpp` | `src/loss.cpp` | `SoftmaxCrossEntropy`, `one_hot()` |
| `model.hpp` | `src/model.cpp` | Sequential container, `predict`, `accuracy` |
| `optimizer.hpp` | `src/optimizer.cpp` | `Optimizer` base + three implementations |
| `training.hpp` | `src/training.cpp` | `train()`, `TrainConfig`, `History` |
| `dataset.hpp` | `src/dataset.cpp` | `load_digits()`, `DigitsData` |
| `pgm.hpp` | header-only | Grayscale PGM writers, used by the demos only |
| `report.hpp` | header-only | Self-contained HTML report writer, used by the demos only |

`Conv2D` lives in its own `.cpp` despite being declared in `layers.hpp`: it is
by far the longest implementation, and separating it keeps `layers.cpp` short
enough to read in one sitting.

`pgm.hpp` and `report.hpp` are intentionally *not* in the umbrella header. They
are demo utilities for turning results into something you can look at — a
grayscale image and a browser page — not part of the neural-network API; the
apps include them explicitly.

---

## Layer abstraction

Everything that participates in the forward/backward flow implements three
methods (`include/nnscratch/layer.hpp`):

```cpp
virtual Tensor forward(const Tensor& x) = 0;
virtual Tensor backward(const Tensor& grad_out) = 0;
virtual std::vector<ParamGrad> params_and_grads() { return {}; }
```

The contract:

1. **`forward` caches whatever `backward` will need.** There is no tape and no
   graph; the cache *is* the graph. `Dense` caches its input, `ReLU` a 0/1 mask,
   `Tanh`/`Sigmoid` their output, `Conv2D` the im2col matrix plus the input
   shape, `Flatten` the input shape.
2. **`backward` consumes $\partial L/\partial \text{output}$ and returns
   $\partial L/\partial \text{input}$**, storing parameter gradients in its own
   members along the way.
3. **`params_and_grads` hands the optimizer stable pointers.** Layers with no
   learnable state inherit the default empty implementation.

`SoftmaxCrossEntropy` deliberately does *not* implement `Layer`. It has a
different signature — `forward(logits, targets) -> double` and a nullary
`backward()` — because it consumes labels and produces a scalar. Making it a
`Layer` would mean smuggling the targets in through a side channel; keeping it
separate makes the training loop's structure explicit.

---

## Life of one training step

What `train()` does for a single mini-batch (`src/training.cpp`), with the
network `Dense → ReLU → Dense`:

```
 1. gather_rows(x_train, batch_indices)          -> xb  (B, 64)
    gather_rows(Y_train, batch_indices)          -> yb  (B, 10)
         (Y_train was one-hot encoded once, before the epoch loop)

 2. model.forward(xb)
       Dense[0].forward   xb -> h1     caches xb
       ReLU[1].forward    h1 -> a1     caches mask(h1 > 0)
       Dense[2].forward   a1 -> logits caches a1
                                                 -> logits (B, 10)

 3. loss_fn.forward(logits, yb)
       softmax(logits) -> probs, cached with yb
       returns the scalar mean cross-entropy    (value discarded here;
                                                 the epoch metric is
                                                 recomputed on the full set)

 4. loss_fn.backward()  ->  (probs - yb) / B     -> g (B, 10)

 5. model.backward(g)      [iterates layers in reverse]
       Dense[2].backward  g  -> g2    writes dW_, db_
       ReLU[1].backward   g2 -> g3    (g2 * mask)
       Dense[0].backward  g3 -> g4    writes dW_, db_
                                                 (g4 is discarded: nobody
                                                  needs d(loss)/d(input))

 6. opt.step(model.params_and_grads())
       flattens to [{&W0,&dW0}, {&b0,&db0}, {&W2,&dW2}, {&b2,&db2}]
       and mutates every param in place
```

Then, once per epoch, after all batches:

```
 7. model.forward(x_train)  on the FULL training set -> metrics
    loss_fn.forward(...)  -> train_loss
    model.accuracy(x_test, y_test) -> test_acc
    push into History
```

Three things about step 7 that are easy to miss:

- It **overwrites every layer's forward cache** with full-dataset activations.
  Safe only because no `backward()` follows it. If you add a callback that
  calls `backward()` after the metric pass, you will be differentiating the
  wrong thing.
- It is not free — two extra full-dataset forward passes per epoch. On this
  dataset that is a small constant; on a large one you would sample instead.
- **Epoch 0 runs it before any update**, which is what gives `History` the
  "untrained → trained" starting point. `cfg.epochs = 60` therefore yields 61
  rows.

There are no gradient-zeroing calls anywhere, because gradients are *assigned*,
not accumulated: `dW_ = matmul(...)` overwrites. This is why nnscratch has no
equivalent of PyTorch's `optimizer.zero_grad()`, and also why the library cannot
express gradient accumulation across micro-batches without a change to
`backward()`.

---

## Shape contracts

| Layer | Input | Output | Notes |
|---|---|---|---|
| `Dense(n_in, n_out)` | $(N, n_{\mathrm{in}})$ | $(N, n_{\mathrm{out}})$ | rank-2 only |
| `ReLU` / `Tanh` / `Sigmoid` | any | same | element-wise, rank-agnostic |
| `Flatten` | $(N, d_1, \ldots, d_k)$ | $(N, \prod d_i)$ | restores rank on backward |
| `Conv2D(C_in, C_out, K, S, P)` | $(N, C_{\mathrm{in}}, H, W)$ | $(N, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})$ | rank-4 required; see [MATH.md](MATH.md#output-geometry) |
| `SoftmaxCrossEntropy` | logits $(N, K)$, targets $(N, K)$ | scalar; backward gives $(N, K)$ | targets one-hot |

A CNN therefore needs `Flatten` between `Conv2D` and `Dense`, and must be fed
`data.train.img` rather than `data.train.flat`. `load_digits` returns both views
of the same samples precisely so this choice costs nothing (`dataset.hpp`).

Shape errors surface as exceptions from `Tensor`, not as silent misbehaviour —
see [Error handling](#error-handling).

---

## Ownership and lifetimes

This is the part with actual sharp edges, all of them stemming from
`ParamGrad` holding raw pointers.

```
Model                            owns  std::vector<std::unique_ptr<Layer>>
  └── Dense                      owns  W_, b_, dW_, db_, x_   (Tensor by value)
        └── Tensor               owns  std::vector<double>    (heap buffer)

ParamGrad {Tensor* param, Tensor* grad}
  └── non-owning views into the layer's members

Optimizer (Momentum/Adam)
  └── std::unordered_map<const Tensor*, Tensor>
        └── keyed by the ADDRESS of the parameter Tensor
```

**Why the addresses are stable.** Layers are held through `unique_ptr`, so the
`Layer` objects live at fixed heap addresses even when the owning
`std::vector` reallocates as you `add()` more layers. Their member `Tensor`s
move with them, so `&W_` is stable for the lifetime of the layer. That is what
makes address-keyed optimizer state sound. (A `std::vector<Dense>` would not be:
reallocation would invalidate every pointer the optimizer holds.)

**Rules that follow:**

| Rule | Why |
|---|---|
| A `Model` must outlive any `ParamGrad` taken from it, and any optimizer that has stepped on it. | The pointers dangle otherwise. |
| Don't reuse a `Momentum`/`Adam` instance across a rebuilt model. | State is keyed by address. A new model can land on freed addresses and inherit stale moments — no crash, just quietly wrong training. `compare.cpp` constructs a fresh optimizer for every run. |
| `params_and_grads()` is cheap but not free — it builds a fresh vector each call. | `train()` calls it once per mini-batch; fine at this scale. Hoist it if you profile and care. |
| References from `Model::add<L>()` / `Model::layer(i)` stay valid for the model's lifetime. | Same `unique_ptr` indirection. `from_scratch.cpp` keeps the `Dense&` from `add()` for hundreds of epochs, then reads its weights. |

`Model` is movable (it is just a vector of `unique_ptr`) but **not copyable** —
`Layer` has no `clone()`. `compare.cpp` relies on move assignment
(`m = build_mlp(...)`) to rebuild a model in place.

---

## Invariants

Breaking any of these compiles fine and produces wrong numbers rather than an
error, so they are worth stating explicitly.

1. **`forward()` before `backward()`, on the same data.** Every `backward()`
   reads a cache written by the matching `forward()`. Calling `backward()` first
   reads a default-constructed (empty) cache.
2. **One `backward()` per `forward()`.** Gradients are assigned, not
   accumulated; two `backward()` calls do not sum, the second overwrites.
3. **`SoftmaxCrossEntropy::forward` before its `backward()`.** Same reason:
   `backward()` reads the cached probabilities and targets.
4. **Batch size may vary between calls** — nothing caches $N$ across calls, and
   the final ragged mini-batch of an epoch exercises this every run.
5. **Targets are a normalised distribution per row.** The fused gradient
   derivation uses $\sum_k y_{ik} = 1$; feeding un-normalised targets silently
   changes the loss being minimised ([MATH.md](MATH.md#the-collapse)).
6. **Label values must be in $[0, K)$.** `one_hot()` indexes with the label and
   `Tensor::operator()` does not bounds-check.

---

## Error handling

The library uses exceptions, and only for programmer errors and I/O:

| Exception | Thrown by | Cause |
|---|---|---|
| `std::invalid_argument` | `matmul`, `operator+/-/*`, `reshape`, `Tensor::from`, `add_row_vector`, `axpy` | shape or size mismatch |
| `std::logic_error` | `rows`, `cols`, `transpose`, `sum_rows`, `add_row_vector`, `matmul`, `Conv2D::forward` | wrong rank for the operation |
| `std::runtime_error` | `load_digits` | file missing, malformed row, or empty file |

Deliberately *not* checked, for speed and simplicity in the inner loops:

- `Tensor::operator()(i, j)` — no bounds check (`data_[i * cols + j]`, raw).
  `dim(i)` and `shape().at(i)` *are* checked; the element accessor is not.
- `Conv2D` geometry — a kernel larger than the padded input makes
  `H + 2P - K` wrap around as unsigned, producing an absurd `oh_` and, in
  practice, a bad allocation rather than a clear message.
- Label range in `one_hot()`.

Everything else is `noexcept`-in-spirit: no allocation failure handling beyond
what the standard library does, and no error codes anywhere.

---

## Determinism

Three independent seeds, deliberately separated:

| Seed | Set by | Controls |
|---|---|---|
| model seed | `Rng rng(42)` in the app | initial weights |
| batch seed | `TrainConfig::batch_seed` (default 123) | mini-batch shuffling order, via a `Rng` local to `train()` |
| split seed | `load_digits(..., split_seed)` (default 0) | which samples land in train vs test |

Keeping them separate is what makes the comparison experiments honest: reseeding
the model RNG before each run gives competing configurations identical starting
weights, while the batch order stays fixed independently. Two runs then differ
*only* in the axis under study.

**Cross-platform caveat.** `Rng::permutation` implements Fisher–Yates directly
in terms of `std::uniform_int_distribution`, and `Rng::normal` uses
`std::normal_distribution`. The *engine* (`mt19937_64`) is specified bit-exactly
by the standard, but the *distributions* are not — libstdc++, libc++ and MSVC
may map the same engine output to different values. So results are reproducible
for a given toolchain, and vary by a few tenths of a percent across toolchains.
That is why the README quotes a range rather than a single number.

---

## What is deliberately absent

Knowing what is *not* here is as useful as knowing what is:

| Absent | Consequence | Rationale |
|---|---|---|
| Autograd tape | Every new layer needs a hand-derived `backward()` | That derivation is the entire point of the project |
| BLAS / SIMD / threads | `matmul` is a naive `ikj` triple loop | Readability; see [PERFORMANCE.md](PERFORMANCE.md) |
| `float` / mixed precision | 8 bytes per element throughout | `double` is what makes the gradient check sharp ([MATH.md](MATH.md#choosing-epsilon)) |
| Views / strides / lazy ops | Every operation allocates a new `Tensor` | One obvious cost model, no aliasing questions |
| Serialisation | Weights cannot be saved or loaded | Runs are seconds long and reproducible from a seed |
| Regularisation, dropout, batch-norm, LR schedules | Not expressible without new layers | Out of scope for the reference being ported |
| Non-sequential topologies | `Model` is a straight stack; no branches or skips | A DAG needs a graph representation, i.e. a tape |

Adding a layer or an optimizer fits the existing structure cleanly —
see [EXTENDING.md](EXTENDING.md). Adding branching does not.
