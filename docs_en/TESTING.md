# Testing

Three test executables, no test framework, one decisive test. This document
explains what each one proves, how the gradient check works internally, and
where the coverage gaps are.

- [Running the tests](#running-the-tests)
- [The harness](#the-harness)
- [test_tensor](#test_tensor)
- [test_gradcheck](#test_gradcheck)
- [test_optimizer](#test_optimizer)
- [Coverage gaps](#coverage-gaps)
- [Adding a test](#adding-a-test)
- [Debugging a failure](#debugging-a-failure)

---

## Running the tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

On a multi-config generator (Visual Studio), add the configuration:

```powershell
cmake --build build --config Debug
ctest --test-dir build --build-config Debug --output-on-failure
```

Run one test directly to see its output without CTest in the way:

```bash
./build/tests/test_gradcheck        # single-config
.\build\tests\Debug\test_gradcheck.exe   # MSVC
```

Tests are built when `NNSCRATCH_BUILD_TESTS` is `ON` — the default when
nnscratch is the top-level project, and `OFF` when it is consumed via
`add_subdirectory` ([BUILD.md](BUILD.md#options)).

Prefer **Debug** for testing: it enables the standard library's assertions on
some toolchains, and the tests are fast enough that the optimizer buys nothing.
CI runs Release, so both configurations are covered between the two.

---

## The harness

`tests/check.hpp`, about 40 lines. Two macros and a counter:

```cpp
CHECK(expr);                      // boolean assertion
CHECK_CLOSE(a, b, tol);           // |a - b| <= tol
return nntest::summary("suite");  // prints PASS/FAIL, returns the exit code
```

Failures print `FAIL file:line expr` — with the actual values and the margin
for `CHECK_CLOSE` — increment a counter, and **do not abort**. One run therefore
reports every failure, not just the first, which matters when a broken
`backward()` fails at dozens of sampled elements and you want to see the
pattern.

`summary()` returns 0 or 1, and CTest reads the process exit code. That is the
whole integration: no framework, no registration macros, no dependency to
install. Adding GoogleTest would contradict the project's zero-dependency rule
for the sake of features these tests do not use.

---

## test_tensor

Forward correctness of the `Tensor` primitives, against values computed by
hand:

| Checked | How |
|---|---|
| `matmul` | a $(2,3) \times (3,2)$ product whose four entries (58, 64, 139, 154) are checked individually |
| `transpose` | shape swap and one element |
| `sum_rows` | shape and two column sums |
| `add_row_vector` | broadcast across two rows |
| `softmax` | rows sum to 1; output order matches logit order |
| `reshape` | data preserved, shape changed |

Tolerance is $10^{-12}$ — these are exact small-integer computations, so
anything looser would hide a real bug.

This test exists because the gradient check *cannot* catch a wrong `forward()`:
a self-consistent but incorrect operation passes it happily. Hand-computed
expected values are the only defence.

---

## test_gradcheck

The decisive test. For every learnable parameter of three different networks,
the analytic gradient from `backward()` must match a central-difference estimate
of $\partial L/\partial w$. The theory — why central differences, why
$\epsilon = 10^{-5}$, why that tolerance — is in
[MATH.md](MATH.md#gradient-checking). Here is what the test actually does.

### Networks covered

| Network | Exercises |
|---|---|
| `Dense → ReLU → Dense → Tanh → Dense` (8→6→5→4) | Dense, ReLU, Tanh |
| `Dense → Sigmoid → Dense` (6→5→3) | Sigmoid |
| `Conv2D(1→2, k=3) → ReLU → Flatten → Dense` on (3,1,5,5) | Conv2D, Flatten, the rank-4 path |

Every layer type in the library appears in at least one, and each network ends
in `SoftmaxCrossEntropy`, so the fused loss gradient is on every path.

The networks are tiny on purpose: each probe costs two full forward passes, and
the check must stay fast enough to run on every commit.

### The procedure

```cpp
// 1. analytic gradients, once
Tensor logits = m.forward(X);
L.forward(logits, Y);
m.backward(L.backward());
auto pgs = m.params_and_grads();

// 2. per parameter tensor, sample ~16 elements
const std::size_t stride = std::max<std::size_t>(1, n / 16);
for (std::size_t i = 0; i < n; i += stride) {
    // 3. central difference at element i
    P->data()[i] = orig + eps;  const double lp = loss_of(m, L, X, Y);
    P->data()[i] = orig - eps;  const double lm = loss_of(m, L, X, Y);
    P->data()[i] = orig;                              // restore exactly
    const double numeric = (lp - lm) / (2.0 * eps);
    CHECK_CLOSE(numeric, analytic[i], 1e-4 * (1.0 + std::fabs(analytic[i])));
}
```

Three details that matter:

- **The analytic gradients are snapshotted** (`pg.grad->data()` copies) before
  probing. Each `loss_of()` call runs a fresh `forward()`, which overwrites the
  layer caches; without the snapshot the comparison would be against gradients
  from a perturbed forward pass.
- **The original value is restored by assignment**, not by subtracting
  $\epsilon$ back — `orig + eps - eps` is not exactly `orig` in floating point,
  and the drift would accumulate over hundreds of probes.
- **Sampling, not exhaustion.** `stride = max(1, n/16)` visits about 16 elements
  per tensor. A wrong `backward()` is essentially never wrong in only $1/16$ of
  its entries, so this costs no detection power and turns a multi-second test
  into a sub-second one.

### What a failure looks like

```
FAIL  tests/test_gradcheck.cpp:47  numeric ~= analytic[i]   (|0.0231 - 0.0154| = 7.7e-03 > 1e-04)
2 failure(s) in gradcheck
```

The two numbers are the diagnosis. See
[Debugging a failure](#debugging-a-failure).

---

## test_optimizer

Two levels: each optimizer in isolation, then the whole stack end to end.

**Isolation.** Each of SGD, Momentum and Adam minimises $f(w) = (w-3)^2$ — the
gradient is supplied directly as $2(w-3)$, so no layer code is involved — and
must arrive within $10^{-2}$ (SGD: $10^{-3}$) of $w = 3$ after 500 steps. This
catches sign errors, misapplied learning rates, and Adam's bias correction being
wrong, none of which the gradient check can see: gradient checking validates
gradients, not what an optimizer does with them.

**End to end.** A 2→8→2 MLP trains for 30 epochs on a trivially separable
synthetic 2-class problem and must exceed 95% accuracy. This is the smoke test
for `train()` itself — batching, shuffling, one-hot encoding, the metric pass —
and it fails loudly if the pieces are individually correct but wired together
wrong.

---

## Coverage gaps

Honest inventory of what is *not* tested:

| Not covered | Risk |
|---|---|
| `load_digits` — parsing, split, normalisation | Medium. Exercised by the demos but never asserted. A malformed-CSV regression would surface as a demo crash. |
| `write_pgm` / `write_pgm_grid` | Low. Output is inspected visually; the functions do not report I/O errors. |
| Error paths (shape-mismatch exceptions) | Low. No test asserts that `matmul` throws on mismatched dimensions. |
| `Conv2D` with stride > 1 or padding > 0 | **Medium-high.** The gradient check uses stride 1, pad 0 only — precisely the configuration where a missing `+=` in col2im is invisible ([MATH.md](MATH.md#col2im-why-scatter-add)). |
| Multi-channel `Conv2D` input ($C_{\mathrm{in}} > 1$) | Medium. The tested conv is 1→2 channels. |
| Cross-toolchain numeric reproducibility | By design — distributions are not portable ([ARCHITECTURE.md](ARCHITECTURE.md#determinism)). |

The `Conv2D` gap is the one worth closing first: adding a stride-2, pad-1,
2-channel case to `grad_check` is a few lines and covers the library's most
error-prone code path.

---

## Adding a test

1. Write `tests/test_<name>.cpp` with a `main()` that ends in
   `return nntest::summary("<name>");`.
2. Add `<name>` to the `foreach` list in `tests/CMakeLists.txt`:

   ```cmake
   foreach(test test_tensor test_gradcheck test_optimizer test_<name>)
   ```

That is all — the loop creates the executable, links `nnscratch` and the warning
interface, and registers it with CTest.

Conventions to keep:

- Fixed seeds everywhere. A flaky test in a library whose selling point is
  reproducibility is worse than no test.
- Tolerances tight enough to fail on a real bug: $10^{-12}$ for exact
  arithmetic, $10^{-4}$ relative for finite differences.
- Keep runtimes under a second. The whole suite should stay fast enough that
  nobody is tempted to skip it.
- New layers **must** be added to `test_gradcheck.cpp` — this is a stated
  requirement in [CONTRIBUTING.md](../CONTRIBUTING.md), and
  [EXTENDING.md](EXTENDING.md#step-4-prove-it-with-a-gradient-check) shows how.

---

## Debugging a failure

When `test_gradcheck` fails, the ratio of the two printed numbers usually
identifies the bug outright:

| Symptom | Likely cause |
|---|---|
| Analytic is exactly $N\times$ the numeric | Double-counting the batch average — e.g. averaging the bias gradient when the loss already divides by $N$ ([MATH.md](MATH.md#notation-and-conventions)) |
| Analytic is the numeric with the opposite sign | A sign flip in the update or the derivative |
| Analytic is zero, numeric is not | The gradient never reached the parameter: `params_and_grads()` returning the wrong pointers, or a `backward()` that forgets to write its `d*_` member |
| Numeric is zero, analytic is not | The parameter does not actually affect the loss — a shape or indexing bug in `forward()` |
| Roughly right, wrong in the last 2–3 digits only | Not a bug. Tighten nothing; this is finite-difference noise ([MATH.md](MATH.md#choosing-epsilon)) |
| Only one element fails, and it moves when you change the seed | A ReLU kink straddled by the probe ([MATH.md](MATH.md#the-kink-caveat)) |
| Everything fails in one layer only | That layer's `backward()`; the rest of the chain is fine, since a broken upstream layer would fail its own probes too |

Because failures do not abort, one run gives you the full pattern — which layer,
which parameter, and whether the error is uniform (a scale factor) or scattered
(an indexing bug).
