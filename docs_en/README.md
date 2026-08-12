# nnscratch documentation (English)

日本語版は [`docs_ja/`](../docs_ja/README.md) にあります。

Ten documents. Start wherever your question is.

| Document | Answers |
|---|---|
| [DESIGN.md](DESIGN.md) | *Why* is it built this way? The non-obvious engineering choices |
| [ARCHITECTURE.md](ARCHITECTURE.md) | *How* does it fit together? Module map, one training step end to end, ownership, invariants |
| [MATH.md](MATH.md) | Where does each `backward()` come from? Every gradient derived from the chain rule |
| [API.md](API.md) | What can I call? Every public type and function, with shapes, preconditions and complexity |
| [experiments.md](experiments.md) | What do the demos show? The three comparisons, and how they map to PyTorch/TensorFlow |
| [EXTENDING.md](EXTENDING.md) | How do I add a layer, optimizer, loss or dataset? |
| [TESTING.md](TESTING.md) | What is tested, what is not, and how do I read a gradient-check failure? |
| [BUILD.md](BUILD.md) | How do I build, install, consume and troubleshoot it? |
| [DATA_FORMATS.md](DATA_FORMATS.md) | What exactly is in `digits.csv`, the output CSVs and the PGM images? |
| [PERFORMANCE.md](PERFORMANCE.md) | What does readability cost? Complexity, measured numbers, known inefficiencies |

---

## Reading paths

**Learning how backpropagation actually works** — the reason the project exists:

1. [MATH.md](MATH.md) — derive `Dense`, then the [fused softmax + cross-entropy](MATH.md#softmax--cross-entropy-fused), then [Conv2D](MATH.md#conv2d-via-im2col)
2. Read the matching source next to each derivation: `src/layers.cpp`,
   `src/loss.cpp`, `src/conv2d.cpp`
3. [TESTING.md § test_gradcheck](TESTING.md#test_gradcheck) — how the derivations are proven correct
4. [experiments.md](experiments.md) — see the consequences in the training curves

**Using it as a library:**

1. [BUILD.md § Consuming the library](BUILD.md#consuming-the-library)
2. [API.md](API.md)
3. [ARCHITECTURE.md § Ownership and lifetimes](ARCHITECTURE.md#ownership-and-lifetimes) — the pointer rules that come with `ParamGrad`
4. [DATA_FORMATS.md](DATA_FORMATS.md) if you are bringing your own data

**Contributing:**

1. [CONTRIBUTING.md](../CONTRIBUTING.md) — ground rules
2. [ARCHITECTURE.md](ARCHITECTURE.md) — the layer contract and the invariants
3. [EXTENDING.md](EXTENDING.md) — step-by-step, with worked examples
4. [TESTING.md](TESTING.md) — your change is not done until the gradient check covers it

**Judging whether it fits your problem:**

1. [PERFORMANCE.md § Scaling limits](PERFORMANCE.md#scaling-limits)
2. [ARCHITECTURE.md § What is deliberately absent](ARCHITECTURE.md#what-is-deliberately-absent)

---

## Conventions used throughout

- $N$ is the batch size and always the leading axis. Rank-2 tensors are
  $(\text{rows}, \text{cols})$; rank-4 are $(N, C, H, W)$, row-major.
- $G = \partial L/\partial Y$ is the upstream gradient — the `grad_out`
  argument of every `backward()`.
- The $1/N$ batch average lives in the loss and nowhere else, so no layer
  divides by $N$.
- Code references are file-relative to the repository root: `src/conv2d.cpp`,
  `include/nnscratch/tensor.hpp`.
