# The math, derived

Every `backward()` in nnscratch is hand-written. This document derives each one
from the chain rule and points at the exact lines of code that implement it, so
you can check the implementation against the math rather than trusting it.

- [Notation and conventions](#notation-and-conventions)
- [Dense](#dense)
- [Activations](#activations)
- [Softmax + cross-entropy, fused](#softmax--cross-entropy-fused)
- [Conv2D via im2col](#conv2d-via-im2col)
- [Flatten](#flatten)
- [Weight initialisation](#weight-initialisation)
- [Optimizers](#optimizers)
- [Gradient checking](#gradient-checking)

---

## Notation and conventions

| Symbol | Meaning |
|---|---|
| $N$ | mini-batch size (always the leading axis) |
| $X$ | layer input, shape $(N, d_{\mathrm{in}})$ for rank-2 layers |
| $Y$ | layer output |
| $L$ | the scalar loss (already averaged over the batch) |
| $G = \partial L / \partial Y$ | the *upstream* gradient, the `grad_out` argument of `backward()` |
| $\odot$ | element-wise (Hadamard) product |
| $\mathbf{1}_N$ | column vector of $N$ ones |

Three conventions hold throughout the library and are worth stating once:

**1. Row-major, batch-first.** A rank-2 tensor is $(\text{rows}, \text{cols})$
with rows indexing samples. A rank-4 tensor is $(N, C, H, W)$ with the flat
offset $((n C + c) H + h) W + w$. Every layer's `backward()` returns a tensor
shaped exactly like its `forward()` input.

**2. The batch average lives in the loss, nowhere else.** `SoftmaxCrossEntropy`
divides by $N$ once (`src/loss.cpp`); every layer downstream of it simply
propagates whatever it receives. This is why `Dense::backward` *sums* the bias
gradient over the batch instead of averaging — the $1/N$ is already baked into
$G$. Getting this wrong is the classic way to end up with an effective learning
rate that silently scales with batch size.

**3. Gradients have the same shape as the thing they differentiate.**
$\partial L/\partial W$ has the shape of $W$, $\partial L/\partial X$ the shape
of $X$. `ParamGrad` pairs them (`include/nnscratch/layer.hpp`).

The general pattern every derivation below follows: write $L$ as a function of
the layer's output, apply the multivariable chain rule

$$\frac{\partial L}{\partial \theta} = \sum_{k} \frac{\partial L}{\partial Y_k} \cdot \frac{\partial Y_k}{\partial \theta}$$

and collect the sum into a matrix product. The sum over $k$ — over *every*
output element that $\theta$ influenced — is where all the interesting structure
comes from, and it is exactly what becomes a `matmul`, a `sum_rows`, or a
scatter-add.

---

## Dense

**Implementation:** `src/layers.cpp`, `Dense::forward` / `Dense::backward`.

### Forward

```math
Y = X W + \mathbf{1}_N b, \qquad
X \in \mathbb{R}^{N \times d_{\mathrm{in}}}, \quad
W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}, \quad
b \in \mathbb{R}^{1 \times d_{\mathrm{out}}}
```

In indices, $Y_{ij} = \sum_{p} X_{ip} W_{pj} + b_j$. The bias is broadcast down
the rows; `Tensor::add_row_vector` does that in place rather than materialising
$\mathbf{1}_N b$.

### Backward

Start from $\partial L / \partial W_{pj}$. The weight $W_{pj}$ feeds every
sample $i$, but only output column $j$:

```math
\frac{\partial L}{\partial W_{pj}}
  = \sum_{i}\sum_{k} \frac{\partial L}{\partial Y_{ik}} \frac{\partial Y_{ik}}{\partial W_{pj}}
  = \sum_{i} G_{ij} X_{ip}
  = \bigl(X^{\top} G\bigr)_{pj}
```

because $\partial Y_{ik} / \partial W_{pj} = X_{ip} \delta_{kj}$. Similarly
$\partial Y_{ik}/\partial b_j = \delta_{kj}$, so the bias gradient sums the
upstream gradient down the batch axis, and $\partial Y_{ik}/\partial X_{ip} =
W_{pk}$ gives the input gradient:

```math
\frac{\partial L}{\partial W} = X^{\top} G, \qquad
\frac{\partial L}{\partial b} = \mathbf{1}_N^{\top} G, \qquad
\frac{\partial L}{\partial X} = G\, W^{\top}
```

which is line-for-line the implementation:

```cpp
dW_ = matmul(x_.transpose(), grad_out);   //  X^T G      (d_in, d_out)
db_ = grad_out.sum_rows();                //  1^T G      (1, d_out)
return matmul(grad_out, W_.transpose());  //  G W^T      (N, d_in)
```

Note the shapes are forced: $X^\top G$ is the only way to contract
$(N, d_{\mathrm{in}})$ against $(N, d_{\mathrm{out}})$ into $W$'s shape. If you
ever forget the formula, the shape algebra reconstructs it.

`Dense::forward` caches `x_` by value — a full copy of the input batch. That is
the memory price of not having an autograd tape; see
[PERFORMANCE.md](PERFORMANCE.md#memory).

---

## Activations

**Implementation:** `src/activations.cpp`.

All three are element-wise, so the Jacobian $\partial Y_{ij}/\partial X_{kl}$ is
diagonal and the chain rule degenerates from a matrix product to a Hadamard
product: $\partial L/\partial X = G \odot f'(X)$.

| Layer | $f(x)$ | $f'$ expressed for backward | Cached |
|---|---|---|---|
| `ReLU` | $\max(0, x)$ | $\mathbb{1}[x > 0]$ | the 0/1 mask |
| `Tanh` | $\tanh x$ | $1 - y^2$ | the output $y$ |
| `Sigmoid` | $\sigma(x) = (1+e^{-x})^{-1}$ | $y(1 - y)$ | the output $y$ |

Tanh and Sigmoid cache the *output*, not the input, because both derivatives can
be written in terms of $y$ alone — one fewer `exp` in the backward pass:

```math
\frac{d}{dx}\tanh x = 1 - \tanh^2 x = 1 - y^2, \qquad
\frac{d}{dx}\sigma(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr) = y(1-y)
```

### The kink at zero

ReLU is not differentiable at $x = 0$. The code picks the sub-gradient $0$
there (`v > 0.0 ? 1.0 : 0.0`), matching PyTorch and TensorFlow. This is a
convention, not a derivation, and it is the one place where a finite-difference
check can legitimately disagree with the analytic gradient — see
[Gradient checking](#gradient-checking).

### Why Sigmoid loses the race

$y(1-y)$ peaks at $y = 1/2$, giving $f'_{\max} = 1/4$. Each Sigmoid layer
therefore shrinks the backward signal by at least a factor of 4, and the product
across $\ell$ layers decays like $4^{-\ell}$. ReLU's derivative is exactly $1$
on the active side, so it contributes no such factor. This is the mechanism
behind Experiment 2 in [experiments.md](experiments.md), where Sigmoid is
visibly the slowest to reach 90%.

---

## Softmax + cross-entropy, fused

**Implementation:** `src/loss.cpp` and `softmax()` in `src/activations.cpp`.

This is the derivation that most justifies the library existing: done naively it
needs a $K \times K$ Jacobian per sample, and done properly it collapses to a
subtraction.

### Forward

For logits $Z \in \mathbb{R}^{N \times K}$ and one-hot targets $Y$:

```math
p_{ij} = \frac{e^{z_{ij}}}{\sum_{k} e^{z_{ik}}}, \qquad
L = -\frac{1}{N} \sum_{i}\sum_{j} y_{ij} \log p_{ij}
```

### The softmax Jacobian

Differentiating $p_{ik}$ with respect to $z_{ij}$ (same row $i$; different rows
are independent) via the quotient rule:

```math
\frac{\partial p_{ik}}{\partial z_{ij}} = p_{ik}\bigl(\delta_{kj} - p_{ij}\bigr)
```

### The collapse

Substituting into $\partial L/\partial z_{ij}$ and using $\partial \log p_{ik} /
\partial z_{ij} = \delta_{kj} - p_{ij}$:

```math
\frac{\partial L}{\partial z_{ij}}
  = -\frac{1}{N} \sum_{k} y_{ik} \bigl(\delta_{kj} - p_{ij}\bigr)
  = -\frac{1}{N} \Bigl( y_{ij} - p_{ij} \sum_{k} y_{ik} \Bigr)
  = \frac{p_{ij} - y_{ij}}{N}
```

The last step uses $\sum_k y_{ik} = 1$ — the targets are one-hot (or, more
generally, a normalised distribution). Everything quadratic in $p$ cancels:

$$\frac{\partial L}{\partial Z} = \frac{P - Y}{N}$$

which is the whole of `SoftmaxCrossEntropy::backward`:

```cpp
const double inv_n = 1.0 / static_cast<double>(probs_.rows());
return inv_n * (probs_ - targets_);
```

Keep the two operations separate — a `Softmax` layer followed by a
`CrossEntropy` loss — and you must build and multiply by that $K \times K$
Jacobian, at $O(NK^2)$ cost and with a real risk of catastrophic cancellation.
Fused, it is $O(NK)$ and exact. Every mainstream framework does the same thing;
this is why `nn.CrossEntropyLoss` takes *logits*, not probabilities.

### Numerical stability

`softmax()` subtracts each row's maximum before exponentiating:

$$p_{ij} = \frac{e^{z_{ij} - m_i}}{\sum_k e^{z_{ik} - m_i}}, \qquad m_i = \max_j z_{ij}$$

Numerator and denominator are both scaled by $e^{-m_i}$, so this is
mathematically the identity. Numerically it is essential: the largest exponent
becomes exactly $e^0 = 1$, so nothing can overflow, and the denominator is
$\geq 1$, so nothing can divide by zero. Without it, a logit of $800$ — quite
reachable if a learning rate is too high — produces `inf/inf = NaN`.

The forward loss also computes $\log(p + 10^{-9})$ rather than $\log p$, which
floors the reported loss for a confidently-wrong prediction at about $20.7$
instead of $-\infty$. Two things to note:

- The $10^{-9}$ affects only the *reported* loss value. The gradient path uses
  $P - Y$ exactly, so training is unaffected by the floor.
- It does perturb the finite-difference check very slightly (relative error
  $\sim 10^{-9}/p$), which is far below the $10^{-4}$ tolerance used there.

---

## Conv2D via im2col

**Implementation:** `src/conv2d.cpp`. This is the longest derivation, and the
one where a hand-written `backward()` is most likely to be wrong — which is
precisely why the gradient check covers it.

### Output geometry

For input $(N, C, H, W)$, kernel size $K$, stride $S$, padding $P$:

```math
H_{\mathrm{out}} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1, \qquad
W_{\mathrm{out}} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1
```

Integer division in C++ truncates toward zero and all quantities are unsigned,
so `(H + 2*pad_ - k_) / stride_ + 1` *is* the floor. A kernel larger than the
padded input underflows `std::size_t` rather than reporting an error; see
[API.md](API.md#conv2d) for the preconditions.

### The lowering

Convolution is a sum of products over a sliding receptive field:

```math
Y_{n,o,i,j} = \sum_{c}\sum_{k_y}\sum_{k_x} X_{n,\,c,\,iS + k_y - P,\;jS + k_x - P} \cdot W_{o,c,k_y,k_x} + b_o
```

The inner triple sum is a dot product between a length-$CK^2$ patch of the input
and a length-$CK^2$ slice of the kernel. im2col makes that literal: build a
matrix `col` whose row $(n, i, j)$ *is* that patch,

$$\texttt{col} \in \mathbb{R}^{(N H_{\mathrm{out}} W_{\mathrm{out}}) \times (CK^2)}, \qquad \texttt{col}[(nH_{\mathrm{out}} + i)W_{\mathrm{out}} + j, (cK + k_y)K + k_x] = X_{n,c,iS+k_y-P, jS+k_x-P}$$

with out-of-range coordinates contributing $0$ (that *is* the zero padding — no
padded copy of the input is ever materialised), and reshape the kernel to
$W_c \in \mathbb{R}^{(CK^2) \times C_{\mathrm{out}}}$. Then the entire layer is

```math
\texttt{out\_mat} = \texttt{col} \cdot W_c + b
```

a single matrix multiply, scattered back into $(N, C_{\mathrm{out}},
H_{\mathrm{out}}, W_{\mathrm{out}})$. This is the classic CS231n trick and it is
what cuDNN's `IMPLICIT_GEMM` algorithm does, minus the materialisation.

Note the ordering: `col`'s row index is $(n, i, j)$ **batch-major**, and its
column index is $(c, k_y, k_x)$ **channel-major**. Both orderings must match
between `forward` (im2col) and `backward` (col2im) or the gradients scatter into
the wrong places — a bug that produces plausible-looking but wrong training.

### Backward

Reshape the upstream gradient the same way `col` was built, into
$G_{\mathrm{mat}} \in \mathbb{R}^{(N H_{\mathrm{out}} W_{\mathrm{out}}) \times C_{\mathrm{out}}}$.
Now the forward pass is *just* an affine map, so the Dense derivation applies
verbatim:

```math
\frac{\partial L}{\partial b} = \mathbf{1}^{\top} G_{\mathrm{mat}}, \qquad
\frac{\partial L}{\partial W_c} = \texttt{col}^{\top} G_{\mathrm{mat}}, \qquad
\frac{\partial L}{\partial \texttt{col}} = G_{\mathrm{mat}} W_c^{\top}
```

The code computes $\partial L/\partial W$ transposed —
`matmul(g_mat.transpose(), col_)`, shape $(C_{\mathrm{out}}, CK^2)$ — because
that is already the memory layout of $W$ as $(C_{\mathrm{out}}, C, K, K)$, so
the copy into `dW_` is a straight linear scan.

That $b$ gradient sums over $N \cdot H_{\mathrm{out}} \cdot W_{\mathrm{out}}$
rows, not just $N$: one bias is shared by every spatial position, so every
position contributes.

### col2im: why scatter-add

The one genuinely new step. $\partial L / \partial \texttt{col}$ is a gradient
w.r.t. the *patches*, and with stride $< K$ the patches overlap — a single input
pixel appears in several rows of `col`. The multivariable chain rule says the
gradient w.r.t. that pixel is the **sum** over every copy:

```math
\frac{\partial L}{\partial X_{n,c,h,w}} = \sum_{(i,j,k_y,k_x)\;:\;iS+k_y-P = h,\; jS+k_x-P = w} \frac{\partial L}{\partial \texttt{col}[(nH_{\mathrm{out}}+i)W_{\mathrm{out}}+j,\;(cK+k_y)K+k_x]}
```

So col2im walks the *same* loop nest as im2col and replaces the read with a
`+=`. Writing `=` instead of `+=` there is the single most common im2col bug: it
is invisible at stride $\geq K$ (no overlap) and quietly wrong everywhere else.
Coordinates that fell outside the input are simply skipped — the gradient with
respect to a padding zero goes nowhere, since padding is a constant.

For the record, this scatter-add is mathematically a *full* correlation of the
upstream gradient with the $180°$-rotated kernel. Frameworks that describe
"the backward of convolution is a transposed convolution" mean exactly this;
im2col just lets you get there without writing a second convolution kernel.

---

## Flatten

**Implementation:** `src/layers.cpp`.

Reshaping is the identity map on the flat buffer, so its gradient is the
identity too — the backward pass only has to restore the original shape from
the one cached in `forward`. It exists because `Conv2D` speaks $(N, C, H, W)$
and `Dense` speaks $(N, d)$, and the row-major layout makes the conversion free:
channel-major flattening $(c, h, w) \mapsto (cHW + hW + w)$ is already what the
buffer contains.

---

## Weight initialisation

**Implementation:** `init_std()` in `src/layers.cpp`, `conv_init_std()` in
`src/conv2d.cpp`.

The concern is variance propagation. For $y = \sum_{p=1}^{n_{\mathrm{in}}} x_p
w_p$ with independent zero-mean terms, $\mathrm{Var}(y) = n_{\mathrm{in}}
\cdot \mathrm{Var}(w) \cdot \mathrm{Var}(x)$. Signal magnitude is preserved
layer-to-layer only if $n_{\mathrm{in}} \mathrm{Var}(w) \approx 1$.

| Strategy | $\sigma$ | Rationale |
|---|---|---|
| **He** | $\sqrt{2 / n_{\mathrm{in}}}$ | ReLU zeroes half the inputs, halving the variance; the factor 2 compensates. Use with ReLU. |
| **Xavier** (`Dense`) | $\sqrt{2 / (n_{\mathrm{in}} + n_{\mathrm{out}})}$ | Compromises between preserving forward variance ($1/n_{\mathrm{in}}$) and backward variance ($1/n_{\mathrm{out}}$). Activation-neutral; use with Tanh/Sigmoid. |

Biases start at zero: there is no symmetry to break once the weights are random.

For `Conv2D`, $n_{\mathrm{in}} = C_{\mathrm{in}} K^2$ — the fan-in is the
receptive-field volume, not the channel count.

> **Implementation note.** `Conv2D`'s `Init::Xavier` branch uses
> $\sqrt{1 / n_{\mathrm{in}}}$, not the $\sqrt{2/(n_{\mathrm{in}} +
> n_{\mathrm{out}})}$ that `Dense` uses. The source comment is explicit that
> convolution only ever uses He in the reference and that the Xavier branch
> exists for completeness. If you rely on Xavier-initialised convolutions,
> read `conv_init_std()` first rather than assuming the `Dense` formula.

---

## Optimizers

**Implementation:** `src/optimizer.cpp`. Full framework-by-framework comparison
in [experiments.md](experiments.md#experiment-1-optimizers-sgd-vs-momentum-vs-adam).

Write $g = \partial L/\partial p$ for one parameter tensor and $\eta$ for the
learning rate. All three update in place, element-wise.

### SGD

$$p \leftarrow p - \eta g$$

One line: `p->axpy(-lr_, *g)`.

### Momentum

```math
v \leftarrow \mu v - \eta g, \qquad p \leftarrow p + v
```

$v$ is an exponentially-weighted sum of past gradients with decay $\mu$. Along a
direction where the gradient is consistent, $v$ approaches $-\eta g /(1-\mu)$ —
an effective step $1/(1-\mu) = 10\times$ larger at $\mu = 0.9$. That factor is
why `compare.cpp` pairs Momentum with a learning rate of 0.05 against SGD's 0.2:
matching the *effective* step is the fair comparison.

Along a direction where the gradient oscillates, consecutive contributions
cancel — which is the "damps the zig-zag" story told geometrically.

### Adam

```math
m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2
```

```math
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}, \qquad
p \leftarrow p - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}
```

Two moving averages: $m$ estimates the mean gradient, $v$ its uncentred
variance. Dividing by $\sqrt{\hat v}$ makes the step scale-invariant — multiply
every gradient by 1000 and the update is unchanged — which is why Adam needs so
little tuning.

**Bias correction.** Both moments start at $0$, so the early averages are biased
toward zero. Taking expectations of the recursion with $g$ roughly stationary,
$\mathbb{E}[m_t] \approx (1 - \beta_1^{t}) \mathbb{E}[g]$; dividing by
$1 - \beta_1^{t}$ removes exactly that factor. Without it the first step at
$\beta_2 = 0.999$ would be about $\sqrt{1000} \approx 31$ times too small. The
correction decays to nothing as $t$ grows — which is why `Adam` keeps a step
counter `t_`, incremented once per `step()` call, not per parameter.

**Where $\varepsilon$ sits.** This library places it *outside* the square root,
$\eta \hat m / (\sqrt{\hat v} + \varepsilon)$, following the original paper and
TensorFlow. PyTorch's default is $\eta \hat m / \sqrt{\hat v + \varepsilon}$.
The two differ only when $\hat v \lll \varepsilon^2$; see
[experiments.md](experiments.md).

### Per-parameter state and pointer identity

Momentum and Adam need one state tensor per parameter, kept in
`std::unordered_map<const Tensor*, Tensor>` keyed by the parameter's **address**
— the C++ analogue of the reference implementation's `id(p)`. Two consequences
worth knowing before you reuse an optimizer:

- State is created lazily on first sight of a pointer, so an optimizer works
  with any model without being told the parameter list up front.
- The key is only meaningful while the model is alive. Destroying a model and
  building a new one can reuse the same heap addresses, so an optimizer carried
  across would silently inherit stale momentum. `compare.cpp` constructs a fresh
  optimizer per run; do the same. See
  [ARCHITECTURE.md](ARCHITECTURE.md#ownership-and-lifetimes).

---

## Gradient checking

**Implementation:** `tests/test_gradcheck.cpp`. This is the test that makes
"from scratch" a verifiable claim rather than an aspiration.

### The estimator

The check compares each analytic gradient against a central difference:

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$$

Central, not forward, because the Taylor expansions of $L(w \pm \epsilon)$ have
their second-order terms cancel: the truncation error is $O(\epsilon^2)$ rather
than the forward difference's $O(\epsilon)$.

### Choosing $\epsilon$

Two errors pull in opposite directions:

- **Truncation**, $\approx \tfrac{1}{6} |L'''| \epsilon^2$ — shrinks with $\epsilon$.
- **Round-off**, $\approx u |L| / \epsilon$ where $u \approx 2.2 \times 10^{-16}$
  is the double-precision unit round-off — *grows* as $\epsilon$ shrinks, since
  $L(w+\epsilon)$ and $L(w-\epsilon)$ agree in ever more leading digits and the
  subtraction cancels them away.

Minimising the sum gives $\epsilon^{\ast} \sim u^{1/3} \approx 6 \times
10^{-6}$. The test uses $10^{-5}$, right at that optimum, and expects roughly
6–8 correct digits. This is also why the library is `double` throughout: in
`float` ($u \approx 10^{-7}$) the best achievable agreement is around $10^{-3}$,
too loose to catch a subtly wrong `backward()`.

### Tolerance

```cpp
CHECK_CLOSE(numeric, analytic[i], 1e-4 * (1.0 + std::fabs(analytic[i])));
```

A mixed absolute/relative bound: the constant term keeps near-zero gradients
from demanding impossible relative precision, the proportional term keeps large
gradients from passing on a technicality.

### Cost, and why it samples

Each probe costs two full forward passes, so checking every one of a model's
$P$ parameters costs $2P$ forwards — thousands, for even this small MLP. The test walks each parameter tensor with
`stride = max(1, n/16)`, sampling about 16 elements per tensor. A wrong
`backward()` is essentially never wrong in only $1/16$ of its entries, so the
sampling costs no real detection power.

### The kink caveat

If a probe straddles a ReLU kink — $|x| < \epsilon$ at some unit — then
$L(w+\epsilon)$ and $L(w-\epsilon)$ lie on different linear pieces and the
finite difference legitimately disagrees with either sub-gradient. With random
inputs the chance is tiny (it needs a pre-activation within $10^{-5}$ of zero),
and the fixed seeds in the test avoid it. If you add a layer with kinks and see
one isolated failure that moves when you change the seed, suspect this before
suspecting your derivation. See
[EXTENDING.md](EXTENDING.md#step-4-prove-it-with-a-gradient-check).

### What it does and does not prove

It proves that each `backward()` is consistent with its own `forward()` —
which is the property autograd would give you for free, and the one most easily
broken by hand-derivation. It does **not** check that `forward()` computes what
you *meant*; a convolution with a transposed index ordering is self-consistent
and passes. `tests/test_tensor.cpp` covers forward correctness with
hand-computed values for that reason.
