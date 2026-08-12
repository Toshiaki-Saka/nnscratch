# Experiment Guide: Optimizers / Activations / Network Architecture

This document explains the concepts and code behind the three comparison
experiments that nnscratch implements, as well as the relationship between the
original implementation (nnscratch C++), PyTorch, and TensorFlow.

---

## Experiment design principle

Each experiment **holds every condition constant except the one under
comparison**, so the effect of a single factor is measured fairly.

- Same data (8×8 handwritten digits, `data/digits.csv`)
- Same initial weights (`Rng` is rewound to a fixed seed before each model is built)
- Same mini-batch order (a separate batch seed is fixed as well)

---

## Experiment 1: Optimizers (SGD vs Momentum vs Adam)

### Concept

Only the rule for "which direction and how far to move the weights" changes.
The network architecture, activation function, and initial weights are all
identical.

| Method | Mechanism | Characteristics |
|---|---|---|
| **SGD** | Step a fixed amount opposite the gradient | Simplest. Can be slow to converge |
| **Momentum** | Carry over the previous direction (velocity) | Accelerates like rolling down a valley; tends to converge faster than SGD |
| **Adam** | Auto-tune each parameter's learning rate via 1st/2nd moments | The modern default. Needs little tuning |

```
SGD      : each step looks only at "the current slope" → prone to zig-zag
Momentum : combines "the current slope" + "accumulated momentum" → moves straighter
Adam     : additionally auto-tunes step size — smaller on steep slopes, larger on gentle ones
```

### How the algorithms relate

The three are not independent; they form an extension chain.

```
SGD
 └─ + velocity v ─────────────────────────→ Momentum
                    └─ replace v with m (1st moment),
                         and add v (2nd moment) → Adam
```

Adam can be understood as "Momentum with two velocity terms, also normalized by
the gradient magnitude."

### Formulas

**SGD**

$$p \leftarrow p - \eta \cdot g$$

$p$: parameter (weight), $g$: gradient, $\eta$: learning rate

---

**Momentum**

$$v \leftarrow \mu v - \eta \cdot g$$

$$p \leftarrow p + v$$

$v$: velocity (carries over the previous step's direction), $\mu$: inertia coefficient (typically $0.9$)

---

**Adam**

$$m \leftarrow \beta_1 m + (1 - \beta_1) g$$

$$v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$

$$\hat{m} = \frac{m}{1 - \beta_1^t}, \qquad \hat{v} = \frac{v}{1 - \beta_2^t}$$

$$p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$$

$t$: step count, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$

$\hat{m}$ and $\hat{v}$ are bias-correction terms (they compensate for the moments
being pulled toward 0 in the early steps).

### Code (nnscratch C++)

```cpp
// src/optimizer.cpp

// SGD: p += -lr * g
void SGD::step(const std::vector<ParamGrad>& pgs) {
    for (const auto& [p, g] : pgs)
        p->axpy(-lr_, *g);
}

// Momentum: v ← μv − lr·g,  p ← p + v
void Momentum::step(const std::vector<ParamGrad>& pgs) {
    for (const auto& [p, g] : pgs) {
        Tensor& v = v_[p];
        for (std::size_t i = 0; i < v.size(); ++i)
            v.data()[i] = mu_ * v.data()[i] - lr_ * g->data()[i];
        p->axpy(1.0, v);
    }
}

// Adam: maintain the two moments m, v to update the parameters
void Adam::step(const std::vector<ParamGrad>& pgs) {
    ++t_;
    const double bc1 = 1.0 - std::pow(b1_, t_);
    const double bc2 = 1.0 - std::pow(b2_, t_);
    for (const auto& [p, g] : pgs) {
        Tensor& m = m_[p];  Tensor& v = v_[p];
        for (std::size_t i = 0; i < m.size(); ++i) {
            const double gi = g->data()[i];
            m.data()[i] = b1_ * m.data()[i] + (1.0 - b1_) * gi;
            v.data()[i] = b2_ * v.data()[i] + (1.0 - b2_) * gi * gi;
            const double mh = m.data()[i] / bc1;
            const double vh = v.data()[i] / bc2;
            p->data()[i] -= lr_ * mh / (std::sqrt(vh) + eps_);
        }
    }
}
```

---

## Experiment 2: Activation functions (ReLU vs Tanh vs Sigmoid)

### Concept

An activation function gives each layer's output its "non-linearity". Without
it, stacking any number of layers is equivalent to a single linear
transformation, and image recognition — which needs curved decision boundaries —
becomes impossible. The experiment fixes the network architecture, SGD, and
initial weights, and swaps only the activation function.

| Function | Formula | Output range | Characteristics |
|---|---|---|---|
| **ReLU** | $\max(0, x)$ | $[0, +\infty)$ | Resistant to vanishing gradients. The modern mainstream |
| **Tanh** | $\dfrac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, +1)$ | Zero-centered output. Larger gradient than Sigmoid |
| **Sigmoid** | $\dfrac{1}{1 + e^{-x}}$ | $(0, 1)$ | Prone to vanishing gradients in deep networks |

**Vanishing gradients**: Sigmoid's maximum gradient is only $\frac{1}{4}$, so the
error signal attenuates with each layer it propagates back through. ReLU keeps a
gradient of $1$ in the positive region, so it resists vanishing.

### Backward-pass derivatives

In each function's $\text{backward}$, the upstream gradient $g$ is multiplied by
the following.

```math
\text{ReLU}': \quad \frac{\partial}{\partial x}\max(0,x) = \begin{cases} 1 & (x > 0) \\ 0 & (x \le 0) \end{cases}
```

$$\text{Tanh}': \quad \frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

$$\text{Sigmoid}': \quad \frac{d}{dx}\sigma(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)$$

### Code (nnscratch C++)

```cpp
// src/activations.cpp

Tensor ReLU::forward(const Tensor& x) {
    mask_ = x.map([](double v){ return v > 0.0 ? 1.0 : 0.0; });
    return x.map([](double v){ return v > 0.0 ? v : 0.0; });
}
Tensor ReLU::backward(const Tensor& g) { return g * mask_; }

Tensor Tanh::forward(const Tensor& x) {
    out_ = x.map([](double v){ return std::tanh(v); });
    return out_;
}
Tensor Tanh::backward(const Tensor& g) {
    return g * out_.map([](double v){ return 1.0 - v * v; });  // 1 - tanh²(x)
}

Tensor Sigmoid::forward(const Tensor& x) {
    out_ = x.map([](double v){ return 1.0 / (1.0 + std::exp(-v)); });
    return out_;
}
Tensor Sigmoid::backward(const Tensor& g) {
    return g * out_.map([](double v){ return v * (1.0 - v); });  // σ(x)(1 - σ(x))
}
```

---

## Experiment 3: Network architecture (shallow linear model vs deep MLP vs CNN)

### Concept

Fix the optimizer to Adam and vary only the "shape" of the network.

| Architecture | Layer composition | What it can do |
|---|---|---|
| **Shallow (no hidden layer)** | $64 \to 10$ | Linear boundaries only. Equivalent to logistic regression |
| **Deep MLP** | $64 \to 64 \to 32 \to 10$ | Can learn complex patterns via non-linear transforms |
| **CNN** | $\text{Conv}(1 \to 8, 3\times3) \to \text{Flatten} \to \text{Dense}(10)$ | Captures local image patterns (edges, curves) |

**MLP vs CNN difference**:

```
MLP : looks at all 64 pixels at once → treats a shifted pattern as a different feature
CNN : slides a small 3×3 window to find local patterns → detects the same feature even when shifted
```

### Code (nnscratch C++)

```cpp
// apps/compare.cpp

nn::Model build_shallow(nn::Rng& rng) {
    nn::Model m;
    m.add<nn::Dense>(64, 10, rng, nn::Init::Xavier);
    return m;
}

nn::Model build_mlp(nn::Rng& rng, const ActFactory& act) {
    nn::Model m;
    m.add<nn::Dense>(64, 64, rng, nn::Init::Xavier);  m.push(act());
    m.add<nn::Dense>(64, 32, rng, nn::Init::Xavier);  m.push(act());
    m.add<nn::Dense>(32, 10, rng, nn::Init::Xavier);
    return m;
}

nn::Model build_cnn(nn::Rng& rng) {
    nn::Model m;
    m.add<nn::Conv2D>(1, 8, 3, 1, 0, rng, nn::Init::He);  // 1ch→8ch, 8×8→6×6
    m.add<nn::ReLU>();
    m.add<nn::Flatten>();
    m.add<nn::Dense>(8 * 6 * 6, 10, rng, nn::Init::Xavier);
    return m;
}
```

---

## Relationship between nnscratch / PyTorch / TensorFlow

### Where the three implementations stand

PyTorch and TensorFlow **run exactly the same math as nnscratch**. The
difference is "who writes that computation (automatic vs by hand)" and "how much
extra functionality comes bundled."

```
nnscratch          PyTorch           TensorFlow (high-level)
──────────────     ──────────────    ──────────────────────
backward() by hand loss.backward()   one-line model.fit()
optimizer by hand  torch.optim.*     model.compile() + fit
training loop hand training loop hnd automatic training loop
CPU only           GPU: .to(device)  GPU: auto-detected
double (64-bit)    float32           float32
```

### The four layers of difference

#### 1. Presence of automatic differentiation (the biggest difference)

```
nnscratch
  A human hand-derives each layer's backward() from the chain rule.

PyTorch
  loss.backward() traverses the computation graph automatically (autograd).

TensorFlow
  tf.GradientTape records the graph and differentiates automatically.
  model.fit() hides even the loop.
```

Even though the optimizer math itself is identical, **who prepares its input (the
gradient $g$) differs fundamentally**.

#### 2. Numeric type

| Implementation | Type | Note |
|---|---|---|
| nnscratch | `double` (64-bit) | Precision-first, suited for teaching |
| PyTorch | `float32` (default) | Fast on GPU |
| TensorFlow | `float32` (default) | Same as above |

Even at the same learning rate, the precision difference can slightly shift the
convergence curve.

#### 3. Subtle difference in Momentum's internal formula

Mathematically equivalent, but the internal velocity tensor $v$ is stored
differently.

$$\text{nnscratch / TF Keras}: \quad v \leftarrow \mu v - \eta g, \quad p \leftarrow p + v$$

$$\text{PyTorch SGD}: \quad v \leftarrow \mu v + g, \quad p \leftarrow p - \eta v$$

Only the scale of $v$ differs by a factor of $\eta$; the result is the same.
PyTorch has a `dampening` option that changes the first-step behavior (nnscratch
does not).

#### 4. Presence of bundled features

| Feature | nnscratch | PyTorch | TensorFlow |
|---|---|---|---|
| Weight decay (L2 regularization) | none | `weight_decay=` | `decay=` |
| Learning-rate scheduler | none | `lr_scheduler.*` | `LearningRateSchedule` |
| Per-parameter-group lr | none | supports `param_groups` | none |
| Adam AMSGrad variant | none | `amsgrad=True` | none |
| Gradient clipping | none | `clip_grad_norm_()` | `clipnorm=` |

### Which difference layer each of SGD / Momentum / Adam touches

| Difference layer | SGD | Momentum | Adam |
|---|---|---|---|
| 1. Automatic differentiation | ○ | ○ | ○ |
| 2. float64 vs float32 | ○ | ○ | ○ |
| 3. Subtle internal-formula difference | **none** | ○ (formula shape differs) | △ ($\varepsilon$ placement differs) |
| 4. Bundled features | weight_decay / clipping | same as left | + AMSGrad |

**Why SGD has no difference (3)**: it holds no state (velocity or moment), so
there is no room for implementation variation.

**On Adam's (3)**: the skeleton and default values ($\beta_1=0.9, \beta_2=0.999, \varepsilon=10^{-8}$)
match, but the placement of $\varepsilon$ differs.

$$\text{nnscratch / paper / TF}: \quad p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$$

$$\text{PyTorch (default)}: \quad p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v} + \varepsilon}}$$

The behavior changes when $\hat{v}$ is extremely small, but in ordinary training
there is almost no difference.

---

## How to run

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run the demo apps

```bash
# Experiments 1–3 (optimizer / activation / architecture comparison)
./build/compare

# The full untrained → trained process
./build/from_scratch
```

Output is written to the `output/` directory in CSV and PGM formats.

### Tests

```bash
ctest --test-dir build --output-on-failure
```

| Test file | Contents |
|---|---|
| `tests/test_tensor.cpp` | Tensor arithmetic / reshape / matmul |
| `tests/test_optimizer.cpp` | Verifies SGD / Momentum / Adam update values |
| `tests/test_gradcheck.cpp` | Confirms analytic gradients match numerical differentiation for all layers |
