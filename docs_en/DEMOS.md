# Running the demos, and reading them

Every way to run this project, and — the part that is usually missing — what
you are supposed to notice in the output.

All commands are given from the repository root. Build first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

| Command | What you get | Time |
|---|---|---|
| [`./build/from_scratch`](#1-from_scratch--one-network-learning) | one network going from random to trained, live in the terminal | ~1 s |
| [`./build/compare`](#2-compare--three-controlled-experiments) | three controlled experiments as result tables | ~6 s |
| [open `output/*.html`](#3-the-html-reports) | the same runs as interactive charts in a browser | instant |
| [`python reference/gui_demo.py`](#4-the-gui-demo--watching-it-learn) | a window that trains in front of you | ~20 s per run |
| [`.\run_demo.ps1`](#5-run_demops1--everything-in-one-command) | build + tests + both demos with progress bars and ASCII charts | ~15 s |
| [`./build/export_reference`](#6-the-reference-comparison) | data for the numpy/PyTorch/TensorFlow comparison | ~10 s |

On Windows the executables live in `build/Release/` — use
`.\build\Release\from_scratch.exe`.

---

## 1. `from_scratch` — one network learning

```bash
./build/from_scratch                        # bundled dataset, output to .
./build/from_scratch data/digits.csv output # explicit paths
```

A 64→64→32→10 MLP with ReLU, trained by plain SGD for 60 epochs.

### Reading the output

```
Untrained test accuracy: 5.3%  (≈10% = random for 10 classes)

epoch   0 | loss 2.5534 | train_acc   7.5% | test_acc   5.3%
epoch   1 | loss 0.3813 | train_acc  91.1% | test_acc  91.4%
epoch   2 | loss 0.6214 | train_acc  79.6% | test_acc  82.5%
epoch   3 | loss 0.1171 | train_acc  97.0% | test_acc  95.3%
...
epoch  60 | loss 0.0010 | train_acc 100.0% | test_acc  97.8%
```

**Epoch 0 is the untrained network**, recorded before any update — that is why
the history has 61 rows for 60 epochs. A loss of 2.55 is what you expect from a
network guessing uniformly over 10 classes: $-\log(1/10) = 2.30$, plus a little
for guessing badly rather than uniformly.

**The jump at epoch 1 is real, not a bug.** 1437 training images at batch size
32 is 45 updates, and this problem is easy — most of the learning genuinely
happens in the first pass.

**The dip at epoch 2 is worth understanding.** Loss goes *up* (0.38 → 0.62) and
accuracy drops 11 points before recovering. Plain SGD at a learning rate of 0.3
overshoots: the step is big enough to jump past the minimum along some
directions. It is not a defect — it is what "the optimizer has no memory and a
fixed step" looks like, and it is exactly what Momentum and Adam smooth out in
[experiment 1](#2-compare--three-controlled-experiments). Expect a few of these
bumps over 60 epochs.

**Train hits 100% while test settles near 97.8%.** The 2% gap is memorisation of
the training set. It stops growing, which is the sign that matters: a gap that
keeps widening while test accuracy falls would be overfitting worth acting on.

**Your numbers will differ slightly.** The untrained figure ranges about 5–9%
and the final about 97–98% depending on compiler, because `std::normal_distribution`
is not specified to give the same values from the same seed across standard
libraries ([ARCHITECTURE.md](ARCHITECTURE.md#determinism)). A *large* deviation
— final accuracy of 60%, or a loss that grows without recovering — is a real
problem worth reporting.

### What it writes

| File | Contents |
|---|---|
| `learning_curve.csv` | the table above, machine-readable |
| `learned_features.pgm` | the 64 first-layer units as 8×8 images |
| `from_scratch.html` | all of it, interactive — see [below](#3-the-html-reports) |

---

## 2. `compare` — three controlled experiments

```bash
./build/compare data/digits.csv output
```

Nine training runs. Each experiment varies **one** axis and holds the data, the
initial weights and the batch order fixed, so any difference is caused by the
thing under study rather than by luck. The concepts are explained in
[experiments.md](experiments.md).

### Reading the tables

```
=== Experiment 1: optimizers (SGD vs Momentum vs Adam) ===
name                  final test acc  best test acc     epochs to 90%
---------------------------------------------------------------------
Adam                          97.8%         97.8%                 1
Momentum                      97.5%         98.1%                 2
SGD                           97.2%         97.5%                 3
```

**`epochs to 90%` is the column that carries the story.** Final accuracy barely
separates the three — they all end within a point of each other, because this
dataset is easy enough that everything gets there eventually. *How fast* they
get there is the actual difference, and that is what the optimizers are about.

**`best` above `final` means the run peaked and drifted back.** Momentum
reaching 98.1% at its best and finishing at 97.5% is normal late-training
wobble on 360 test images — 0.6 points is two images.

Experiment 2 is the clearest result in the project:

```
ReLU                          97.2%         97.5%                 3
Sigmoid                       96.7%         96.7%                 9
Tanh                          97.2%         97.5%                 1
```

**Sigmoid takes three times as many epochs to reach 90%.** Its derivative peaks
at 1/4, so every layer it passes through shrinks the backward signal by at least
4× — the vanishing-gradient problem, visible in a nine-second demo. ReLU's
derivative is exactly 1 on the active side and contributes no such factor
([MATH.md](MATH.md#why-sigmoid-loses-the-race)).

Experiment 3 needs a caveat rather than a conclusion:

```
1_shallow                     96.4%         96.7%                 2
2_deep_mlp                    97.8%         97.8%                 1
3_cnn                         97.2%         97.5%                 1
```

**Depth beating the linear model is solid; any CNN-vs-MLP ordering is not.**
Over 10 seeds the deep MLP and the CNN differ by 0.31 points on average, which
is less than their own seed-to-seed spread and about one test image. Do not read
a winner into a single run — see
[experiments.md](experiments.md#what-actually-happens), and check it yourself
with `python reference/architecture_trials.py --seeds 10`.

### What it writes

`cmp_optimizers.csv`, `cmp_activations.csv`, `cmp_architecture.csv` (long
format, one row per run per epoch), `cnn_filters.pgm`, and `compare.html`.

---

## 3. The HTML reports

Both demos write a self-contained page. Open it directly — no server, no
install; CSS, JavaScript and every data point are inlined, so it also works from
a USB stick or an email attachment.

```powershell
start output\from_scratch.html      # Windows
open  output/from_scratch.html      # macOS
xdg-open output/from_scratch.html   # Linux
```

### What to do with it

- **Hover a chart** for a crosshair and the values at that epoch. This is the
  fastest way to answer "when exactly did it cross 90%?".
- **Click a legend entry** to hide a series and see what is underneath. Colours
  follow the run, so hiding one never repaints the others.
- **Show data table** prints the numbers behind the chart.
- **Linear scale / Log scale** on loss charts. Start on log: on a linear axis
  the whole curve after epoch 5 is flat against zero and you can see nothing.
  On log, steady progress is a straight line, and a *plateau* — a flat stretch
  on log — is the thing to look for.
- **Theme** toggles light/dark; it also follows your OS setting.

The weight images are min–max normalised **per tile**, so brightness is
comparable inside one tile and *not* between tiles. Read structure from them,
never magnitude.

---

## 4. The GUI demo — watching it learn

```bash
python -m pip install -r reference/requirements.txt   # once
python reference/gui_demo.py                          # then press Train
python reference/gui_demo.py --autostart              # start training immediately
```

A window with four live panels. This is the one to use if you want to *see* the
thing the rest of the docs describe.

| Panel | What to watch for |
|---|---|
| Training loss (log) | the steep first epochs, then the long straight decline; bumps are SGD overshooting |
| Accuracy | whether train and test stay together — they separate when the network starts memorising |
| First-layer weights | the interesting one: noise at epoch 0, resolving into stroke- and edge-shaped structure within a few epochs |
| Held-out digits | ten test images with the current prediction; a wrong one shows `predicted→true` in red and bold |

The dropdowns rebuild from the same seed, so switching is a controlled
comparison, not a reshuffle. Three worth trying:

1. **Sigmoid vs ReLU** on the deep MLP: the loss curve visibly crawls where ReLU
   drops. That is the vanishing gradient, live.
2. **CNN**: the weights panel switches to the eight 3×3 kernels, and you can
   watch light/dark opposition — edge detectors — appear out of noise.
3. **Shallow (64→10)**: it converges almost instantly and then stops improving.
   The ceiling of a linear model, in about five seconds.

The network is the numpy reference from `reference/`, which agrees with the C++
library to 2e-15, so you are watching nnscratch's arithmetic. Training runs
inside the UI event loop, one epoch per tick — about 20 seconds for a 60-epoch
MLP run.

---

## 5. `run_demo.ps1` — everything in one command

PowerShell 7+, Windows:

```powershell
.\run_demo.ps1                # configure, build, ctest, then both demos
.\run_demo.ps1 -SkipBuild     # just the demos
.\run_demo.ps1 > demo.log     # non-interactive: no prompts, no animation
```

Interactively it draws animated progress bars, ASCII learning curves and
comparison bar charts, and pauses on "Press Enter" between parts. When stdout is
not a console it drops the animation automatically and prints one plain line per
epoch, so redirecting and piping work — `-NonInteractive` forces that on a
terminal. Everything it does is reachable with plain CMake commands; nothing in
the library depends on it.

---

## 6. The reference comparison

Trains the same network in numpy, PyTorch and TensorFlow from the *same* initial
weights and batch order, to show that the hand-written backward pass and
`loss.backward()` are the same computation. Full write-up, results and caveats:
[reference/README.md](../reference/README.md).

```bash
./build/export_reference                       # split, weights, batch order
python reference/numpy_reference.py            # hand-written backward
python reference/pytorch_reference.py          # autograd
python reference/tensorflow_reference.py       # GradientTape
python reference/compare_curves.py             # overlay + divergence table
```

Reading the divergence table: **2e-15 is round-off** and means "identical".
**2e-08 is also identical** — it is this library reporting `log(p + 1e-9)` where
the frameworks report `log p`, an offset already present at epoch 0 that the
gradients never see. Anything above that is a real difference, and there is
exactly one: TensorFlow's CNN, because Keras applies Adam's epsilon before bias
correction.

Two standalone checks:

```bash
python reference/check_optimizer_equivalence.py   # which update each framework implements
python reference/architecture_trials.py --seeds 10 # does experiment 3's ranking hold up?
```

---

## When something looks wrong

| Symptom | Meaning |
|---|---|
| Final accuracy differs by a few tenths from these numbers | Expected across compilers — distributions are not portable ([ARCHITECTURE.md](ARCHITECTURE.md#determinism)) |
| Loss rises for an epoch or two, then recovers | Normal for plain SGD at lr 0.3. Persistent growth is not |
| Untrained accuracy is 5% rather than 10% | Also fine. Untrained argmax is arbitrary, not uniform — it can land below chance |
| `load_digits: cannot open ...` | Run from the repository root, or pass the CSV path explicitly |
| Accuracy stuck near 10% for many epochs | A real problem. Start with `ctest` — the gradient check catches a broken `backward()` ([TESTING.md](TESTING.md)) |
| The HTML page is blank | Check the browser console. The page needs JavaScript, though it needs no network |
