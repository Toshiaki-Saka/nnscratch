# Performance characteristics

nnscratch optimises for readability, and this document is the honest accounting
of what that costs. It covers the complexity of every operation, measured
numbers for the bundled demos, where the time and memory actually go, and what
you would change first if you wanted speed.

Nothing here is a to-do list. Most of these costs are deliberate: see
[DESIGN.md](DESIGN.md) and the [house rules](EXTENDING.md#house-rules).

- [Complexity summary](#complexity-summary)
- [matmul](#matmul)
- [Measured cost of the demos](#measured-cost-of-the-demos)
- [Where the time goes](#where-the-time-goes)
- [Memory](#memory)
- [Known inefficiencies](#known-inefficiencies)
- [If you did want it faster](#if-you-did-want-it-faster)
- [Scaling limits](#scaling-limits)

---

## Complexity summary

$N$ = batch size, in element operations (a multiply-add counts as one).

| Operation | Forward | Backward | Extra memory |
|---|---|---|---|
| `matmul((m,k), (k,n))` | $O(mkn)$ | — | $mn$ |
| `transpose` | $O(mn)$ | — | $mn$ |
| `sum_rows` | $O(mn)$ | — | $n$ |
| `map`, `axpy`, `operator+/-/*` | $O(\text{size})$ | — | $\text{size}$ (except `axpy`, in place) |
| `Dense(d_{in}, d_{out})` | $O(N d_{in} d_{out})$ | $O(2 N d_{in} d_{out})$ | cached input: $N d_{in}$ |
| `ReLU` / `Tanh` / `Sigmoid` | $O(N d)$ | $O(N d)$ | cache: $N d$ |
| `Flatten` | $O(\text{size})$ | $O(\text{size})$ | a copy |
| `Conv2D` | $O(N OH OW C K^2 C_{out})$ | $O(2 \times \text{same})$ | im2col: $N OH OW C K^2$ |
| `softmax` | $O(NK)$ | — | $NK$ |
| `SoftmaxCrossEntropy` | $O(NK)$ | $O(NK)$ | $2NK$ |
| `SGD::step` | $O(P)$ | — | none |
| `Momentum::step` | $O(P)$ | — | $P$ (velocity) |
| `Adam::step` | $O(P)$ | — | $2P$ (two moments) |
| `Rng::permutation(n)` | $O(n)$ | — | $n$ |

$P$ = total parameter count. Backward being about $2\times$ forward for `Dense`
and `Conv2D` is structural, not incidental: both compute a parameter gradient
*and* an input gradient, each roughly the cost of the forward contraction
([MATH.md](MATH.md#dense)).

---

## matmul

Every heavy operation in the library funnels into this triple loop
(`src/tensor.cpp`):

```cpp
for (std::size_t i = 0; i < m; ++i)
    for (std::size_t p = 0; p < k; ++p) {
        const double aip = a(i, p);
        for (std::size_t j = 0; j < n; ++j) out(i, j) += aip * b(p, j);
    }
```

Naive $O(mkn)$, no blocking, no packing, no threads, no intrinsics. One thing it
*does* get right is the **`ikj` loop order**: the innermost loop walks a
contiguous row of `b` and a contiguous row of `out`, with the scalar `aip`
hoisted. That makes the inner loop a textbook `axpy` — unit stride in both
arrays, no reduction dependency — which is both cache-friendly and something
compilers will auto-vectorise.

The obvious `ijk` ordering (accumulating a dot product into `out(i,j)`) would
stride through `b` by `n` elements, missing cache on every access and carrying a
loop-carried dependency into the accumulator. Same asymptotics, several times
slower in practice. This is the one performance decision in the library that
was made on purpose.

What is still left on the table versus a real GEMM: no cache blocking (large
matrices thrash L2), no explicit FMA, no multithreading, and `operator()`
recomputing `i * cols + j` per access — cheap, and normally hoisted by the
optimiser, but not free in a debug build.

---

## Measured cost of the demos

Release build, MSVC, one desktop core. Treat the wall-clock times as
order-of-magnitude — the operation counts are exact.

### `from_scratch` — 64→64→32→10 MLP, 60 epochs

| Quantity | Value |
|---|---|
| Parameters | $64\cdot64+64 + 64\cdot32+32 + 32\cdot10+10 = 6570$ |
| Forward MACs per sample | $4096 + 2048 + 320 = 6464$ |
| Training MACs per epoch | $1437 \times 3 \times 6464 \approx 27.9\text{M}$ |
| Metric-pass MACs per epoch | $(1437 + 360) \times 6464 \approx 11.6\text{M}$ |
| Total for the run | $\approx 2.4$ G MAC |
| **Wall clock** | **≈ 1.1 s** |

That works out to roughly 2 G multiply-adds per second — perhaps 15% of what a
single core can do with well-tuned double-precision SIMD. For a naive scalar
triple loop, that is about what the `ikj` ordering plus auto-vectorisation
buys you.

### `compare` — nine training runs

| Quantity | Value |
|---|---|
| Runs | 3 optimizers (40 ep) + 3 activations (40 ep) + 3 architectures (25 ep) |
| CNN parameters | conv $8(1\cdot9+1) = 80$; dense $288\cdot10+10 = 2890$; total 2970 |
| CNN forward MACs per sample | conv $36 \cdot 9 \cdot 8 = 2592$; dense $2880$; total $5472$ |
| **Wall clock** | **≈ 6.4 s** |

Note that the CNN has **less than half the parameters** of the deep MLP (2970 vs
6570) but comparable arithmetic per sample — the conv layer's 80 weights are
reused at all 36 spatial positions. That reuse is the entire point of
convolution, and it is visible here as an arithmetic-intensity difference rather
than a parameter-count one.

---

## Where the time goes

**The metric pass is ~30% of an epoch.** After every epoch, `train()` runs a
forward pass over the *full* training set for the loss, plus another over the
test set for accuracy: 11.6M MACs against the 27.9M spent actually training.
That is a deliberate trade — reporting the true epoch-end loss rather than a
running average of mini-batch losses computed against stale weights makes the
learning curves interpretable — but it is the single biggest lever if you ever
need this loop to be faster. Evaluating every $k$-th epoch would cut roughly a
quarter of the runtime.

**The first layer's input gradient is computed and discarded.** `Model::backward`
calls `backward()` on every layer including the first, whose returned
$\partial L/\partial X$ nobody uses. For this MLP that is $1437 \times 64 \times
64 \approx 5.9$M wasted MACs per epoch, about 15% of the backward pass. Skipping
it would require the first layer to know it is first — structure the library
does not have, and the cost is not worth introducing it for.

**Activations pay for `std::function`.** `Tensor::map` takes a
`const std::function<double(double)>&`, so every element goes through an
indirect call that cannot be inlined. On an $(N, 64)$ tensor that is thousands
of indirect calls where the arithmetic is a single comparison. A template
parameter would inline it; `std::function` keeps the signature readable in the
header, and activations are not the bottleneck here.

**Allocation is everywhere.** No operation is in place except `axpy` and
`add_row_vector`, so `grad_out * mask_` allocates a full tensor, as does every
`transpose()`, every `map()`, and every `matmul()`. One mini-batch step through
the demo MLP allocates on the order of 20 tensors. Sizes are small enough that
the allocator keeps up, and the alternative — expression templates or an
in-place op set — would obscure exactly the code the project exists to show.

---

## Memory

**Peak memory is driven by the metric pass, not by training.** Layer caches are
sized by the batch, and the metric pass runs with the whole training set as one
batch:

| | Batch of 32 | Full training set (1437) |
|---|---|---|
| `Dense[0]` cached input | 16 KB | 736 KB |
| `ReLU[1]` mask | 16 KB | 736 KB |
| `Dense[2]` cached input | 16 KB | 736 KB |
| `ReLU[3]` mask | 8 KB | 368 KB |
| `Dense[4]` cached input | 8 KB | 368 KB |
| **Total caches** | **≈ 64 KB** | **≈ 2.9 MB** |

Parameters and gradients together are 6570 × 2 × 8 B ≈ 105 KB; Adam adds two
more copies. The dataset itself is 1797 × 64 × 8 B ≈ 920 KB, held twice
(`flat` and `img` are separate buffers, not views), plus a copy of each split.

**im2col is the memory story for convolutions.** The lowered matrix holds
$N \cdot OH \cdot OW \cdot C K^2$ doubles — for the demo CNN, $9\times$ the
input volume, since each 3×3 window overlaps its neighbours. Over the full
training set that is 1437 × 36 × 9 × 8 B ≈ 3.7 MB, cached until the next
forward pass because `backward()` needs it. The expansion factor is
$OH \cdot OW \cdot K^2 / (H \cdot W)$, so it grows with kernel size and shrinks
with stride: a 5×5 kernel on this input would be 25×.

Nothing here is close to a problem at this scale. It becomes one immediately at
[larger scales](#scaling-limits).

---

## Known inefficiencies

Catalogued so you know they are known, roughly in descending order of cost:

| # | Inefficiency | Cost | Why it stays |
|---|---|---|---|
| 1 | `matmul` has no blocking, threading or intrinsics | 5–50× vs a tuned GEMM | A blocked, packed kernel is unreadable; the naive loop *is* the definition of matrix multiplication |
| 2 | Full-dataset metric pass every epoch | ~30% of runtime | Makes learning curves exact and interpretable |
| 3 | Every operation allocates | many small allocations per step | No views, no aliasing questions, one obvious cost model |
| 4 | `map` dispatches through `std::function` | indirect call per element | Keeps the header signature simple |
| 5 | `Dense` caches its input by value | $N d_{in}$ doubles per layer | The cache *is* the autograd tape; a reference would dangle |
| 6 | First layer's input gradient computed and discarded | ~15% of the backward pass | Would need position-awareness in the layer interface |
| 7 | `Conv2D` rebuilds `Wc` / `WcT` on every call | $O(C K^2 C_{out})$ per call | Small next to the matmul; caching adds invalidation logic |
| 8 | `elementwise` copies `a`, then overwrites it | one extra pass over the data | Three lines instead of a hand-written loop per operator |
| 9 | `params_and_grads()` builds a fresh vector per mini-batch | one small allocation per step | Keeps `Model` stateless |
| 10 | Optimizer does a hash lookup per parameter per step | negligible (a handful of parameters) | Address-keying is what makes optimizers model-agnostic |
| 11 | `gather_rows` copies each mini-batch | $N \times$ row size per batch | No view type exists |

---

## If you did want it faster

In rough order of payoff per line changed, without restructuring the library:

1. **Evaluate metrics every $k$ epochs.** A few lines in `src/training.cpp`,
   removes most of a 30% overhead.
2. **Parallelise `matmul`'s outer loop.** `#pragma omp parallel for` over `i`,
   or a `std::thread` fan-out — the iterations are independent, and `out` rows do
   not alias. Near-linear speedup, but OpenMP is a build-level dependency the
   project does not take.
3. **Block `matmul` for cache.** Tiling to L1-sized sub-blocks typically buys
   2–4× at these dimensions and costs a good deal of readability.
4. **Template `map` on the functor** instead of `std::function`, so activations
   inline and vectorise.
5. **Reuse buffers.** Give each layer a persistent output tensor and write into
   it, rather than returning a fresh one per call. Removes most allocations at
   the cost of introducing aliasing rules to reason about.
6. **Switch to `float`.** Halves memory traffic and doubles SIMD width — but it
   would break the gradient check's precision budget
   ([MATH.md](MATH.md#choosing-epsilon)), which is a poor trade for a library
   whose correctness argument rests on that check.

For anything beyond this, you want BLAS — which is exactly the dependency this
project exists to do without.

---

## Scaling limits

What breaks first if you point this at a bigger problem:

| Scale | What happens |
|---|---|
| MNIST (60 000 × 28×28) | The full-dataset metric pass allocates ~1.5 GB of layer caches in one go. Sample the metric set, or evaluate in batches |
| Any conv with a large input | im2col expansion of $OH \cdot OW \cdot K^2/(HW)$ makes the lowered matrix the dominant allocation |
| Deep networks (dozens of layers) | Fine numerically — but the gradient check's cost grows with parameter count, and sampling 16 elements per tensor gets thin |
| Large batches | `Conv2D`'s im2col scales linearly with $N$ and is materialised in full; there is no chunking |
| Long training runs | No checkpointing exists. A crash loses everything; re-running from the seed is the only recovery |

At the 8×8-digits scale the library is built for, none of these bind — the whole
demo suite runs in about seven seconds.
