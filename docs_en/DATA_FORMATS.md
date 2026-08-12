# Data and output formats

Exact specifications for everything nnscratch reads and writes: the bundled
dataset, the learning-curve CSVs, and the PGM images. All of it is plain text
or trivially-parsed binary, so no library is needed on either side.

- [digits.csv (input)](#digitscsv-input)
- [The train/test split](#the-traintest-split)
- [learning_curve.csv](#learning_curvecsv)
- [cmp_*.csv](#cmp_csv)
- [PGM images](#pgm-images)
- [Plotting the outputs](#plotting-the-outputs)

---

## digits.csv (input)

`data/digits.csv` — the 8×8 handwritten-digits dataset, a subset of the UCI
optical-recognition dataset as shipped with scikit-learn
(`sklearn.datasets.load_digits`).

| Property | Value |
|---|---|
| Lines | 1800 total: 2 comment lines, 1 header, **1797 records** |
| Fields per record | **65** — 64 pixels then the label |
| Pixel range | integers 0–16 (a 4× downsample of 32×32 bitmaps, hence 0–16 rather than 0–255) |
| Label range | integers 0–9 |
| Pixel order | row-major over the 8×8 image: `p0..p7` is the top row |
| Separator | `,` — no quoting, no embedded commas |

```
# UCI optical handwritten digits (8x8). 64 pixel columns (0..16) + label (0..9).
# Source: sklearn.datasets.load_digits (subset of UCI ML hand-written digits).
p0,p1,p2,...,p63,label
0,0,5,13,9,1,0,0,...,0,0
```

### What the parser accepts

`read_csv` in `src/dataset.cpp` is deliberately small. Precisely:

- **Skipped:** empty lines; lines beginning with `#`; any line beginning with
  `p0` (the header).
- **Required:** exactly 65 comma-separated fields. Any other count throws
  `std::runtime_error("load_digits: malformed row in <path>")`.
- **Parsed with `std::stod`**, so `5`, `5.0` and `5e0` are all accepted, and
  leading whitespace is tolerated. A non-numeric field throws
  `std::invalid_argument` from `stod` itself.
- **No trailing-newline requirement**, no BOM handling, no `\r` stripping — a
  CRLF file works because `\r` lands in the final field and `stod` stops at it,
  but do not rely on that if you generate your own data.
- An empty result throws `load_digits: no data in <path>`.

### Normalisation

Pixels are divided by 16 on load, mapping 0–16 to $[0, 1]$. Labels are stored
as-is. This matters if you supply your own CSV: **the loader assumes the 0–16
range**, so 0–255 data comes out 16× too large and will train badly. Rescale it
before saving, or write your own loader
([EXTENDING.md](EXTENDING.md#adding-a-dataset)).

### Generating an equivalent file

```python
from sklearn.datasets import load_digits
import numpy as np

d = load_digits()
rows = np.column_stack([d.data.astype(int), d.target.astype(int)])
header = ",".join(f"p{i}" for i in range(64)) + ",label"
np.savetxt("digits.csv", rows, fmt="%d", delimiter=",", header=header, comments="")
```

The bundled file is committed so the demos run fully offline; this is only for
reproducing or extending it.

---

## The train/test split

Performed by `load_digits`, not stored in the file:

1. Seed an `Rng` with `split_seed` (default `0`).
2. Fisher–Yates permutation of the 1797 record indices.
3. Cut at `floor(1797 * train_frac)`; with `train_frac = 0.8` that is index
   1437.

**Result with the defaults: 1437 train / 360 test.**

Each split is materialised in two views of the *same* samples:

| Member | Shape | For |
|---|---|---|
| `flat` | (N, 64) | `Dense` models |
| `img` | (N, 1, 8, 8) | `Conv2D` models |
| `labels` | N ints | both |

The two views hold identical numbers — the row-major layout means `flat` is just
`img` with the shape metadata changed — so switching between an MLP and a CNN
costs nothing.

The split is reproducible for a fixed toolchain but **not portable across
toolchains**: `permutation()` uses `std::uniform_int_distribution`, whose
mapping from engine output to values is implementation-defined
([ARCHITECTURE.md](ARCHITECTURE.md#determinism)). Different platforms get
different — but equally valid — 1437/360 splits, which is one reason accuracy
figures vary slightly.

---

## learning_curve.csv

Written by `from_scratch` (`apps/from_scratch.cpp`).

```
epoch,train_loss,train_acc,test_acc
0,2.55344,0.0751566,0.0527778
1,0.381343,0.910926,0.913889
...
60,0.000971,1,0.977778
```

| Column | Meaning |
|---|---|
| `epoch` | 0 … `cfg.epochs`. **Row 0 is the untrained network**, before any update |
| `train_loss` | mean softmax cross-entropy on the **full** training set at epoch end |
| `train_acc` | accuracy on the full training set, a **fraction in [0,1]**, not a percentage |
| `test_acc` | accuracy on the test set, likewise a fraction |

61 data rows for the default 60 epochs. Values are written with the default
`std::ostream` precision (6 significant digits), so `1` means exactly 1.0.

---

## cmp_*.csv

Written by `compare` (`apps/compare.cpp`) — three files,
`cmp_optimizers.csv`, `cmp_activations.csv`, `cmp_architecture.csv`, all in the
same **long (tidy) format**, one row per run per epoch:

```
name,epoch,loss,test_acc
Adam,0,2.38628,0.0527778
Adam,1,0.239651,0.925
...
Momentum,0,...
SGD,0,...
```

| Column | Meaning |
|---|---|
| `name` | the run's label within its experiment |
| `epoch` | 0 … `cfg.epochs` (40 for experiments 1–2, 25 for experiment 3) |
| `loss` | full-training-set loss at epoch end |
| `test_acc` | test accuracy, fraction |

`train_acc` is not included here — the comparison charts plot loss and test
accuracy only.

Rows are grouped by `name` and sorted **alphabetically**, because the runs live
in a `std::map<std::string, History>`. That is why experiment 3 uses the labels
`1_shallow`, `2_deep_mlp`, `3_cnn`: the numeric prefixes force the ordering to
be the pedagogically meaningful one instead of `cnn, deep_mlp, shallow`.

Long format means a group-by is needed before plotting; the snippet
[below](#plotting-the-outputs) does it in three lines.

---

## PGM images

Both demos emit **binary PGM** (Netpbm "P5"), written by
`include/nnscratch/pgm.hpp`. PGM was chosen because a correct encoder is about
eight lines and needs no dependency — consistent with the rest of the project.

### File structure

```
P5\n<width> <height>\n255\n<width*height bytes, row-major, one byte per pixel>
```

The header is ASCII with exactly one `\n` after each of the three parts; pixel
data follows immediately with no padding or alignment. Values are
`clamp(v, 0, 1) * 255 + 0.5` truncated to an integer, so the input range is
$[0,1]$ and anything outside is clipped, not scaled.

### The two outputs

| File | Written by | Contents | Geometry |
|---|---|---|---|
| `learned_features.pgm` | `from_scratch` | the 64 first-layer `Dense` neurons, each as an 8×8 image of its 64 input weights | 8×8 grid of 8-px cells, 1-px gaps → **73×73**, 5342 bytes |
| `cnn_filters.pgm` | `compare` | the CNN's 8 learned 3×3 convolution filters | 1×8 grid of 3-px cells, 1-px gaps → **33×5**, 177 bytes |

Grid geometry from `write_pgm_grid(path, cells, cell, cols, pad)`:

```math
\text{rows} = \left\lceil \frac{|\text{cells}|}{\text{cols}} \right\rceil, \qquad
W = \text{cols}\cdot\text{cell} + (\text{cols}+1)\,\text{pad}, \qquad
H = \text{rows}\cdot\text{cell} + (\text{rows}+1)\,\text{pad}
```

The background is 0.5 → byte 128, mid-grey, so both positive and negative
weights are visible against it.

### Per-cell normalisation — read this before interpreting the images

`write_pgm_grid` min–max normalises **each cell independently**:

```cpp
const double range = (hi - lo) > 1e-12 ? (hi - lo) : 1.0;
canvas[...] = (src[y * cell + x] - lo) / range;
```

Consequences:

- Within one cell, brighter means a larger weight. **Across cells, brightness is
  not comparable** — every cell is stretched to fill the full 0–255 range, so a
  filter with a tiny weight range looks exactly as contrasty as a dominant one.
- Zero is not a fixed grey level. It lands wherever it happens to fall between
  that cell's min and max, so you cannot read the *sign* of a weight off the
  image.
- A constant cell (`hi - lo <= 1e-12`) uses `range = 1.0`, rendering as uniform
  black rather than dividing by zero.

This is the right trade for *seeing structure* — the digit-stroke detectors in
`learned_features.pgm` are visible only because of it — and the wrong one for
comparing magnitudes. Read the weights out of `Dense::weight()` /
`Conv2D::weight()` if you need those.

### Viewing and converting

PGM opens directly in GIMP, IrfanView, XnView and most image tools. To convert:

```bash
magick learned_features.pgm learned_features.png      # ImageMagick 7
convert learned_features.pgm learned_features.png     # ImageMagick 6
```

```python
# Python, no PGM-specific library needed
import matplotlib.pyplot as plt
img = plt.imread("learned_features.pgm")             # matplotlib reads P5
plt.imshow(img, cmap="gray", interpolation="nearest")
plt.axis("off"); plt.show()
```

At 73×73 the images are small by design; view them with nearest-neighbour
scaling, since smooth interpolation blurs away exactly the stroke structure you
are looking for.

---

## Plotting the outputs

Visualisation is deliberately left to the consumer so the library stays
dependency-free. Two short recipes.

**Learning curve:**

```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("learning_curve.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(df.epoch, df.train_loss);  ax1.set(xlabel="epoch", ylabel="loss")
ax2.plot(df.epoch, df.train_acc * 100, label="train")
ax2.plot(df.epoch, df.test_acc * 100, label="test")
ax2.set(xlabel="epoch", ylabel="accuracy (%)"); ax2.legend()
plt.tight_layout(); plt.show()
```

**Comparison curves** (long format, so group by `name`):

```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("cmp_optimizers.csv")
for name, g in df.groupby("name"):
    plt.plot(g.epoch, g.test_acc * 100, label=name)
plt.xlabel("epoch"); plt.ylabel("test accuracy (%)"); plt.legend(); plt.show()
```

Both files are small enough (a few kilobytes) to open in any spreadsheet if you
would rather not write code.
