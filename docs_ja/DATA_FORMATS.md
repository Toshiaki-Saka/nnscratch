# データと出力の形式

nnscratch が読み書きするすべてのものの正確な仕様。同梱データセット、学習曲線の CSV、
そして PGM 画像である。いずれもプレーンテキストか単純なバイナリなので、読む側にも
書く側にもライブラリは要らない。

- [digits.csv（入力）](#digitscsv入力)
- [train/test 分割](#traintest-分割)
- [learning_curve.csv](#learning_curvecsv)
- [cmp_*.csv](#cmp_csv)
- [PGM 画像](#pgm-画像)
- [出力をプロットする](#出力をプロットする)

---

## digits.csv（入力）

`data/digits.csv` — 8×8 の手書き数字データセット。UCI の光学式手書き数字認識
データセットの部分集合で、scikit-learn 同梱のもの
（`sklearn.datasets.load_digits`）である。

| 項目 | 値 |
|---|---|
| 行数 | 全 1800 行: コメント 2 行、ヘッダ 1 行、**レコード 1797 行** |
| レコードあたりのフィールド数 | **65** — ピクセル 64 個の後にラベル |
| ピクセルの範囲 | 整数 0–16（32×32 のビットマップを 4 倍にダウンサンプルしたもの。0–255 でないのはそのため） |
| ラベルの範囲 | 整数 0–9 |
| ピクセルの順序 | 8×8 画像の行優先。`p0..p7` が最上段 |
| 区切り文字 | `,` — 引用符もエスケープされたカンマもない |

```
# UCI optical handwritten digits (8x8). 64 pixel columns (0..16) + label (0..9).
# Source: sklearn.datasets.load_digits (subset of UCI ML hand-written digits).
p0,p1,p2,...,p63,label
0,0,5,13,9,1,0,0,...,0,0
```

### パーサが受け付けるもの

`src/dataset.cpp` の `read_csv` は意図的に小さい。正確には次のとおりである。

- **読み飛ばす:** 空行、`#` で始まる行、`p0` で始まる行（ヘッダ）。
- **必須:** カンマ区切りでちょうど 65 フィールド。それ以外は
  `std::runtime_error("load_digits: malformed row in <path>")` を送出する。
- **`std::stod` でパースする。** したがって `5`、`5.0`、`5e0` はいずれも受理され、
  先頭の空白も許容される。数値でないフィールドは `stod` 自身が
  `std::invalid_argument` を送出する。
- **末尾改行は必須ではなく**、BOM の処理も `\r` の除去も行わない。CRLF のファイルが
  動くのは `\r` が最終フィールドに入り `stod` がそこで止まるからだが、自分でデータを
  生成する場合はこれに依存しないほうがよい。
- 結果が空なら `load_digits: no data in <path>` を送出する。

### 正規化

ピクセルは読み込み時に 16 で割られ、0–16 が $[0, 1]$ に写される。ラベルはそのまま
保持される。独自の CSV を用意する場合に効いてくる。**ローダは 0–16 の範囲を前提に
している** ので、0–255 のデータは 16 倍大きくなってうまく学習できない。保存前に
スケールし直すか、独自のローダを書くこと
（[EXTENDING.md](EXTENDING.md#データセットを追加する)）。

### 同等のファイルを生成する

```python
from sklearn.datasets import load_digits
import numpy as np

d = load_digits()
rows = np.column_stack([d.data.astype(int), d.target.astype(int)])
header = ",".join(f"p{i}" for i in range(64)) + ",label"
np.savetxt("digits.csv", rows, fmt="%d", delimiter=",", header=header, comments="")
```

デモが完全にオフラインで動くよう、ファイル自体はリポジトリにコミットされている。
上記は再現や拡張のためのものである。

---

## train/test 分割

分割はファイルに保存されておらず、`load_digits` が行う。

1. `split_seed`（既定 `0`）で `Rng` をシードする。
2. 1797 個のレコード添字を Fisher–Yates で置換する。
3. `floor(1797 * train_frac)` で切る。`train_frac = 0.8` なら添字 1437 である。

**既定値での結果: train 1437 / test 360。**

各スプリットは、*同じ* サンプルの 2 つのビューとして実体化される。

| メンバ | 形状 | 用途 |
|---|---|---|
| `flat` | (N, 64) | `Dense` のモデル |
| `img` | (N, 1, 8, 8) | `Conv2D` のモデル |
| `labels` | N 個の int | 両方 |

2 つのビューは同一の数値を保持する。行優先の配置により `flat` は `img` の形状
メタデータを変えただけのものなので、MLP と CNN の切り替えにコストはかからない。

分割は特定のツールチェーンでは再現するが、**ツールチェーンをまたぐと再現しない**。
`permutation()` が使う `std::uniform_int_distribution` は、エンジン出力から値への
写像が実装依存だからである（[ARCHITECTURE.md](ARCHITECTURE.md#決定性)）。
プラットフォームが違えば異なる — しかし同じく妥当な — 1437/360 の分割になる。
正解率の数値がわずかにぶれる一因である。

---

## learning_curve.csv

`from_scratch`（`apps/from_scratch.cpp`）が書き出す。

```
epoch,train_loss,train_acc,test_acc
0,2.55344,0.0751566,0.0527778
1,0.381343,0.910926,0.913889
...
60,0.000971,1,0.977778
```

| 列 | 意味 |
|---|---|
| `epoch` | 0 … `cfg.epochs`。**0 行目は未学習のネットワーク**（更新前）である |
| `train_loss` | エポック終了時、学習データ **全体** に対する平均 softmax cross-entropy |
| `train_acc` | 学習データ全体に対する正解率。パーセントではなく **[0,1] の割合** |
| `test_acc` | テストデータに対する正解率。同じく割合 |

既定の 60 エポックならデータ行は 61 行になる。値は `std::ostream` の既定精度
（有効数字 6 桁）で書かれるので、`1` はちょうど 1.0 を意味する。

---

## cmp_*.csv

`compare`（`apps/compare.cpp`）が書き出す。`cmp_optimizers.csv`、
`cmp_activations.csv`、`cmp_architecture.csv` の 3 ファイルで、いずれも同じ
**long（tidy）形式** である。1 行が 1 実行 1 エポックに対応する。

```
name,epoch,loss,test_acc
Adam,0,2.38628,0.0527778
Adam,1,0.239651,0.925
...
Momentum,0,...
SGD,0,...
```

| 列 | 意味 |
|---|---|
| `name` | その実験内での実行のラベル |
| `epoch` | 0 … `cfg.epochs`（実験 1・2 は 40、実験 3 は 25） |
| `loss` | エポック終了時、学習データ全体の損失 |
| `test_acc` | テスト正解率、割合 |

`train_acc` は含まれない。比較用のグラフは損失とテスト正解率だけを描くためである。

行は `name` でまとめられ、**アルファベット順** に並ぶ。実行が
`std::map<std::string, History>` に入っているからである。実験 3 のラベルが
`1_shallow`、`2_deep_mlp`、`3_cnn` になっているのはそのためで、数字の接頭辞によって
`cnn, deep_mlp, shallow` ではなく教育的に意味のある順序を強制している。

long 形式なので、プロットの前にグループ化が必要になる。
[後述のスニペット](#出力をプロットする)は 3 行でそれを行う。

---

## PGM 画像

どちらのデモも **バイナリ PGM**（Netpbm の「P5」）を出力する。書き出しは
`include/nnscratch/pgm.hpp` である。PGM を選んだのは、正しいエンコーダが 8 行ほどで
書けて依存関係を必要としないからで、プロジェクト全体の方針と一貫している。

### ファイル構造

```
P5\n<width> <height>\n255\n<幅*高さ バイト。行優先、1 ピクセル 1 バイト>
```

ヘッダは ASCII で、3 つの部分それぞれの後にちょうど 1 つの `\n` が入る。ピクセル
データはその直後から始まり、パディングもアラインメントもない。値は
`clamp(v, 0, 1) * 255 + 0.5` を整数に切り捨てたものなので、入力の範囲は $[0,1]$ で、
それを外れた値はスケールされるのではなくクリップされる。

### 2 つの出力

| ファイル | 出力元 | 内容 | 寸法 |
|---|---|---|---|
| `learned_features.pgm` | `from_scratch` | 第 1 層 `Dense` の 64 ユニットを、それぞれ 64 個の入力重みからなる 8×8 画像として | 8×8 のグリッド、セル 8 px、隙間 1 px → **73×73**、5342 バイト |
| `cnn_filters.pgm` | `compare` | CNN が学習した 3×3 の畳み込みフィルタ 8 枚 | 1×8 のグリッド、セル 3 px、隙間 1 px → **33×5**、177 バイト |

`write_pgm_grid(path, cells, cell, cols, pad)` のグリッド寸法は次のとおり。

```math
\text{rows} = \left\lceil \frac{|\text{cells}|}{\text{cols}} \right\rceil, \qquad
W = \text{cols}\cdot\text{cell} + (\text{cols}+1)\,\text{pad}, \qquad
H = \text{rows}\cdot\text{cell} + (\text{rows}+1)\,\text{pad}
```

背景は 0.5、すなわちバイト値 128 の中間グレーである。正の重みも負の重みも背景に
対して見えるようにするためである。

### セルごとの正規化 — 画像を解釈する前に読むこと

`write_pgm_grid` は **セルごとに独立して** min–max 正規化する。

```cpp
const double range = (hi - lo) > 1e-12 ? (hi - lo) : 1.0;
canvas[...] = (src[y * cell + x] - lo) / range;
```

その帰結:

- 1 つのセルの内部では、明るいほど重みが大きい。しかし **セルどうしでは明るさを
  比較できない。** すべてのセルが 0–255 の全域に引き伸ばされるので、重みの範囲が
  ごく小さいフィルタも支配的なフィルタとまったく同じコントラストに見える。
- ゼロは固定のグレーレベルにならない。そのセルの最小値と最大値の間のどこに落ちるかで
  変わるので、画像から重みの *符号* を読み取ることはできない。
- 定数のセル（`hi - lo <= 1e-12`）は `range = 1.0` を使うので、ゼロ除算にはならず
  一様な黒として描かれる。

*構造を見る* という目的にはこれが正しい選択で、`learned_features.pgm` に数字の
ストローク検出器が見えるのはまさにこの正規化のおかげである。一方、大きさを比較する
目的には誤った選択である。それが必要なら `Dense::weight()` /
`Conv2D::weight()` から重みを直接読み出すこと。

### 閲覧と変換

PGM は GIMP、IrfanView、XnView をはじめ多くの画像ツールでそのまま開ける。変換するなら:

```bash
magick learned_features.pgm learned_features.png      # ImageMagick 7
convert learned_features.pgm learned_features.png     # ImageMagick 6
```

```python
# Python。PGM 専用のライブラリは不要
import matplotlib.pyplot as plt
img = plt.imread("learned_features.pgm")             # matplotlib は P5 を読める
plt.imshow(img, cmap="gray", interpolation="nearest")
plt.axis("off"); plt.show()
```

73×73 という小ささは意図的である。表示は最近傍補間で行うこと。滑らかな補間は、
まさに見たいストロークの構造をぼかしてしまう。

---

## 出力をプロットする

ライブラリを依存関係ゼロに保つため、可視化は意図的に利用側に委ねている。短いレシピを
2 つ挙げる。

**学習曲線:**

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

**比較曲線**（long 形式なので `name` でグループ化する）:

```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("cmp_optimizers.csv")
for name, g in df.groupby("name"):
    plt.plot(g.epoch, g.test_acc * 100, label=name)
plt.xlabel("epoch"); plt.ylabel("test accuracy (%)"); plt.legend(); plt.show()
```

どちらのファイルも数キロバイトなので、コードを書きたくなければ表計算ソフトで
開いてもよい。
