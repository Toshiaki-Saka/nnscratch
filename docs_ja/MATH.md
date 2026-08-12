# 数式の導出

nnscratch のすべての `backward()` は手で書かれている。このドキュメントは、その
一つひとつを連鎖律から導出し、実装しているコードの該当箇所を示す。実装を信用する
のではなく、数式と突き合わせて確かめられるようにするためのものである。

- [記法と規約](#記法と規約)
- [Dense](#dense)
- [活性化関数](#活性化関数)
- [softmax + cross-entropy の融合](#softmax--cross-entropy-の融合)
- [im2col による Conv2D](#im2col-による-conv2d)
- [Flatten](#flatten)
- [重みの初期化](#重みの初期化)
- [オプティマイザ](#オプティマイザ)
- [勾配チェック](#勾配チェック)

---

## 記法と規約

| 記号 | 意味 |
|---|---|
| $N$ | ミニバッチサイズ（常に先頭軸） |
| $X$ | レイヤーの入力。ランク 2 のレイヤーでは $(N, d_{\mathrm{in}})$ |
| $Y$ | レイヤーの出力 |
| $L$ | スカラーの損失（バッチ平均を取った後の値） |
| $G = \partial L / \partial Y$ | *上流の* 勾配。`backward()` の `grad_out` 引数 |
| $\odot$ | 要素ごとの積（アダマール積） |
| $\mathbf{1}_N$ | 要素がすべて 1 の $N$ 次元列ベクトル |

ライブラリ全体を貫く規約が 3 つある。一度だけ明示しておく。

**1. 行優先・バッチ先頭。** ランク 2 テンソルは $(\text{rows}, \text{cols})$ で、
行がサンプルに対応する。ランク 4 テンソルは $(N, C, H, W)$ で、フラットな
オフセットは $((n C + c) H + h) W + w$ である。すべてのレイヤーの `backward()`
は、`forward()` の入力とまったく同じ形状のテンソルを返す。

**2. バッチ平均は損失にのみ存在する。** `SoftmaxCrossEntropy` が一度だけ $N$ で
割り（`src/loss.cpp`）、それより下流のレイヤーは受け取ったものをそのまま伝播する。
`Dense::backward` がバイアスの勾配をバッチ方向に *平均ではなく合計* するのは
このためである（$1/N$ はすでに $G$ に含まれている）。ここを取り違えると、実効
学習率がバッチサイズに応じて静かに変わってしまう、という典型的なバグになる。

**3. 勾配は、微分される対象と同じ形状を持つ。** $\partial L/\partial W$ は $W$
と同じ形状、$\partial L/\partial X$ は $X$ と同じ形状である。`ParamGrad` は
その両者を対にする（`include/nnscratch/layer.hpp`）。

以下の導出はすべて同じ手順を踏む。$L$ をレイヤーの出力の関数として書き、
多変数の連鎖律

$$\frac{\partial L}{\partial \theta} = \sum_{k} \frac{\partial L}{\partial Y_k} \cdot \frac{\partial Y_k}{\partial \theta}$$

を適用し、和を行列積へまとめる。$k$ についての和 — $\theta$ が影響を与えた
*すべての* 出力要素についての和 — が構造のすべてを生み、それがそのまま
`matmul` になり、`sum_rows` になり、scatter-add になる。

---

## Dense

**実装:** `src/layers.cpp` の `Dense::forward` / `Dense::backward`。

### forward

```math
Y = X W + \mathbf{1}_N b, \qquad
X \in \mathbb{R}^{N \times d_{\mathrm{in}}}, \quad
W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}, \quad
b \in \mathbb{R}^{1 \times d_{\mathrm{out}}}
```

添字で書けば $Y_{ij} = \sum_{p} X_{ip} W_{pj} + b_j$ である。バイアスは各行に
ブロードキャストされる。`Tensor::add_row_vector` は $\mathbf{1}_N b$ を実体化
せず、その場で加算する。

### backward

$\partial L / \partial W_{pj}$ から始める。重み $W_{pj}$ はすべてのサンプル $i$
に効くが、出力の列は $j$ だけである。

```math
\frac{\partial L}{\partial W_{pj}}
  = \sum_{i}\sum_{k} \frac{\partial L}{\partial Y_{ik}} \frac{\partial Y_{ik}}{\partial W_{pj}}
  = \sum_{i} G_{ij} X_{ip}
  = \bigl(X^{\top} G\bigr)_{pj}
```

$\partial Y_{ik} / \partial W_{pj} = X_{ip} \delta_{kj}$ だからである。同様に
$\partial Y_{ik}/\partial b_j = \delta_{kj}$ なので、バイアスの勾配は上流の勾配を
バッチ軸方向に合計したものになり、$\partial Y_{ik}/\partial X_{ip} = W_{pk}$
から入力に対する勾配が得られる。

```math
\frac{\partial L}{\partial W} = X^{\top} G, \qquad
\frac{\partial L}{\partial b} = \mathbf{1}_N^{\top} G, \qquad
\frac{\partial L}{\partial X} = G\, W^{\top}
```

これは実装と一行ずつ対応する。

```cpp
dW_ = matmul(x_.transpose(), grad_out);   //  X^T G      (d_in, d_out)
db_ = grad_out.sum_rows();                //  1^T G      (1, d_out)
return matmul(grad_out, W_.transpose());  //  G W^T      (N, d_in)
```

形状によって答えが一意に決まる点に注意したい。$(N, d_{\mathrm{in}})$ と
$(N, d_{\mathrm{out}})$ を縮約して $W$ の形にする方法は $X^\top G$ しかない。
式を忘れても、形状の代数から復元できる。

`Dense::forward` は `x_` を値でキャッシュする（入力バッチの完全なコピー）。これは
自動微分のテープを持たないことの代償である。
[PERFORMANCE.md](PERFORMANCE.md#メモリ) を参照。

---

## 活性化関数

**実装:** `src/activations.cpp`。

3 つとも要素ごとの演算なので、ヤコビアン $\partial Y_{ij}/\partial X_{kl}$ は
対角行列になり、連鎖律は行列積からアダマール積へと退化する。すなわち
$\partial L/\partial X = G \odot f'(X)$ である。

| レイヤー | $f(x)$ | backward で使う $f'$ | キャッシュ |
|---|---|---|---|
| `ReLU` | $\max(0, x)$ | $\mathbb{1}[x > 0]$ | 0/1 マスク |
| `Tanh` | $\tanh x$ | $1 - y^2$ | 出力 $y$ |
| `Sigmoid` | $\sigma(x) = (1+e^{-x})^{-1}$ | $y(1 - y)$ | 出力 $y$ |

Tanh と Sigmoid が入力ではなく *出力* をキャッシュするのは、どちらの導関数も
$y$ だけで書けるからである。backward での `exp` が 1 回減る。

```math
\frac{d}{dx}\tanh x = 1 - \tanh^2 x = 1 - y^2, \qquad
\frac{d}{dx}\sigma(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr) = y(1-y)
```

### 原点の折れ

ReLU は $x = 0$ で微分不可能である。コードはそこで劣勾配 $0$ を採用しており
（`v > 0.0 ? 1.0 : 0.0`）、PyTorch や TensorFlow と一致する。これは導出ではなく
規約であり、有限差分による検証が解析的勾配と正当に食い違いうる唯一の箇所でも
ある。[勾配チェック](#勾配チェック)を参照。

### Sigmoid が遅れる理由

$y(1-y)$ は $y = 1/2$ で最大となり $f'_{\max} = 1/4$ である。したがって Sigmoid
の層を 1 つ通るたびに backward の信号は最低でも 1/4 に縮み、$\ell$ 層の積は
$4^{-\ell}$ のオーダーで減衰する。ReLU の導関数は活性側でちょうど $1$ なので、
この種の係数を生まない。これが [experiments.md](experiments.md) の実験 2 で
Sigmoid が 90% に到達するのが目に見えて遅い理由である。

---

## softmax + cross-entropy の融合

**実装:** `src/loss.cpp` と `src/activations.cpp` の `softmax()`。

このライブラリの存在理由を最もよく示す導出である。素朴にやればサンプルごとに
$K \times K$ のヤコビアンが必要になるが、正しくやれば単なる引き算に collapse する。

### forward

ロジット $Z \in \mathbb{R}^{N \times K}$ と one-hot の教師 $Y$ に対して:

```math
p_{ij} = \frac{e^{z_{ij}}}{\sum_{k} e^{z_{ik}}}, \qquad
L = -\frac{1}{N} \sum_{i}\sum_{j} y_{ij} \log p_{ij}
```

### softmax のヤコビアン

$p_{ik}$ を $z_{ij}$ で微分する（同じ行 $i$ 内。異なる行は独立）。商の微分則から:

```math
\frac{\partial p_{ik}}{\partial z_{ij}} = p_{ik}\bigl(\delta_{kj} - p_{ij}\bigr)
```

### collapse

これを $\partial L/\partial z_{ij}$ に代入し、
$\partial \log p_{ik} / \partial z_{ij} = \delta_{kj} - p_{ij}$ を使うと:

```math
\frac{\partial L}{\partial z_{ij}}
  = -\frac{1}{N} \sum_{k} y_{ik} \bigl(\delta_{kj} - p_{ij}\bigr)
  = -\frac{1}{N} \Bigl( y_{ij} - p_{ij} \sum_{k} y_{ik} \Bigr)
  = \frac{p_{ij} - y_{ij}}{N}
```

最後の等号で $\sum_k y_{ik} = 1$ を使っている。教師が one-hot（より一般には
正規化された分布）だからである。$p$ について 2 次の項はすべて相殺する。

$$\frac{\partial L}{\partial Z} = \frac{P - Y}{N}$$

これが `SoftmaxCrossEntropy::backward` のすべてである。

```cpp
const double inv_n = 1.0 / static_cast<double>(probs_.rows());
return inv_n * (probs_ - targets_);
```

両者を分離したまま — `Softmax` レイヤーの後ろに `CrossEntropy` 損失を置く形 —
にすると、あの $K \times K$ ヤコビアンを構築して掛ける必要があり、$O(NK^2)$ の
コストがかかるうえ桁落ちの危険もある。融合すれば $O(NK)$ で、しかも厳密である。
主要なフレームワークがどれも同じことをしており、`nn.CrossEntropyLoss` が確率では
なく *ロジット* を受け取るのはこのためである。

### 数値安定性

`softmax()` は指数化の前に各行の最大値を引く。

$$p_{ij} = \frac{e^{z_{ij} - m_i}}{\sum_k e^{z_{ik} - m_i}}, \qquad m_i = \max_j z_{ij}$$

分子と分母がともに $e^{-m_i}$ 倍されるので、数学的には恒等変形である。しかし数値
的には必須で、最大の指数がちょうど $e^0 = 1$ になるためオーバーフローが起こり得ず、
分母が $1$ 以上になるためゼロ除算も起こらない。これがないと、学習率が高すぎる場合
に十分あり得る $800$ 程度のロジットで `inf/inf = NaN` になる。

forward の損失計算は $\log p$ ではなく $\log(p + 10^{-9})$ を使う。自信を持って
外した予測に対する損失値が $-\infty$ ではなく約 $20.7$ で下げ止まる。2 点補足する。

- $10^{-9}$ が影響するのは *報告される* 損失値だけである。勾配の経路は厳密に
  $P - Y$ を使うので、学習には影響しない。
- 有限差分による検証にはごくわずかな摂動（相対誤差 $\sim 10^{-9}/p$）を与えるが、
  そこで使う許容誤差 $10^{-4}$ よりはるかに小さい。

---

## im2col による Conv2D

**実装:** `src/conv2d.cpp`。最も長い導出であり、手書きの `backward()` が最も
間違いやすい箇所でもある。勾配チェックがここを確実に覆っているのはそのためである。

### 出力サイズ

入力 $(N, C, H, W)$、カーネルサイズ $K$、ストライド $S$、パディング $P$ に対して:

```math
H_{\mathrm{out}} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1, \qquad
W_{\mathrm{out}} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1
```

C++ の整数除算はゼロ方向に切り捨てられ、ここに現れる量はすべて符号なしなので、
`(H + 2*pad_ - k_) / stride_ + 1` は床関数そのものである。パディング後の入力より
大きいカーネルを与えるとエラーではなく `std::size_t` のアンダーフローが起きる。
前提条件は [API.md](API.md#conv2d) を参照。

### 行列積への落とし込み

畳み込みは、スライドする受容野の上での積和である。

```math
Y_{n,o,i,j} = \sum_{c}\sum_{k_y}\sum_{k_x} X_{n, c, iS + k_y - P, jS + k_x - P} \cdot W_{o,c,k_y,k_x} + b_o
```

内側の 3 重和は、入力の長さ $CK^2$ のパッチとカーネルの長さ $CK^2$ のスライスとの
内積である。im2col はそれを文字どおりの形にする。行 $(n, i, j)$ がそのパッチ
*そのもの* であるような行列 `col` を作る。

$$\texttt{col} \in \mathbb{R}^{(N H_{\mathrm{out}} W_{\mathrm{out}}) \times (CK^2)}$$

範囲外の座標は $0$ を寄与する（これがゼロパディングの実体であり、パディング済みの
入力コピーは一度も作られない）。カーネルを
$W_c \in \mathbb{R}^{(CK^2) \times C_{\mathrm{out}}}$ に変形すれば、レイヤー全体が

```math
\texttt{out\_mat} = \texttt{col} \cdot W_c + b
```

という 1 回の行列積になり、それを $(N, C_{\mathrm{out}}, H_{\mathrm{out}},
W_{\mathrm{out}})$ へ書き戻す。これは CS231n で知られる古典的な手法であり、
実体化を除けば cuDNN の `IMPLICIT_GEMM` が行っていることと同じである。

添字の順序に注意したい。`col` の行添字は $(n, i, j)$ で **バッチ優先**、列添字は
$(c, k_y, k_x)$ で **チャネル優先** である。この順序は `forward`（im2col）と
`backward`（col2im）で一致していなければならない。ずれると勾配が誤った位置へ
散らばり、もっともらしく見えるが誤った学習になる。

### backward

上流の勾配を `col` と同じ流儀で
$G_{\mathrm{mat}} \in \mathbb{R}^{(N H_{\mathrm{out}} W_{\mathrm{out}}) \times C_{\mathrm{out}}}$
に変形する。こうすると forward は *単なるアフィン写像* なので、Dense の導出が
そのまま使える。

```math
\frac{\partial L}{\partial b} = \mathbf{1}^{\top} G_{\mathrm{mat}}, \qquad
\frac{\partial L}{\partial W_c} = \texttt{col}^{\top} G_{\mathrm{mat}}, \qquad
\frac{\partial L}{\partial \texttt{col}} = G_{\mathrm{mat}} W_c^{\top}
```

コードが $\partial L/\partial W$ を転置した形
（`matmul(g_mat.transpose(), col_)`、形状 $(C_{\mathrm{out}}, CK^2)$）で計算するのは、
それが $W$ を $(C_{\mathrm{out}}, C, K, K)$ として見たときのメモリ配置そのもの
であり、`dW_` へのコピーが単純な線形走査で済むからである。

$b$ の勾配は $N$ 行ではなく $N \cdot H_{\mathrm{out}} \cdot W_{\mathrm{out}}$ 行に
わたって合計される点にも注意したい。1 つのバイアスをすべての空間位置が共有して
いるので、すべての位置が寄与する。

### col2im: なぜ scatter-add なのか

ここだけが本質的に新しい手順である。$\partial L / \partial \texttt{col}$ は
*パッチ* に対する勾配であり、ストライドが $K$ 未満だとパッチは重なり合う。入力の
1 ピクセルが `col` の複数の行に現れるのである。多変数の連鎖律に従えば、その
ピクセルに対する勾配はすべてのコピーにわたる **和** になる。

```math
\frac{\partial L}{\partial X_{n,c,h,w}} = \sum_{(i,j,k_y,k_x)\,:\,iS+k_y-P = h,\ jS+k_x-P = w} \frac{\partial L}{\partial \texttt{col}[(nH_{\mathrm{out}}+i)W_{\mathrm{out}}+j,\ (cK+k_y)K+k_x]}
```

そこで col2im は im2col と *同じ* ループ構造をたどり、読み出しを `+=` に置き換える。
ここを `+=` ではなく `=` と書くのが im2col 実装で最も多いバグである。ストライドが
$K$ 以上なら重なりがないので表面化せず、それ以外のすべての場合に静かに誤る。
入力の外に出た座標は単に飛ばす。パディングのゼロは定数なので、それに対する勾配は
どこにも行かない。

なお、この scatter-add は数学的には、上流の勾配と $180°$ 回転したカーネルとの
*full* 相関に等しい。「畳み込みの backward は転置畳み込みである」という説明は
まさにこれを指している。im2col を使えば、2 つ目の畳み込みカーネルを書かずに
そこへ到達できる。

---

## Flatten

**実装:** `src/layers.cpp`。

reshape はフラットなバッファ上では恒等写像なので、その勾配も恒等写像である。
backward は `forward` でキャッシュした元の形状に戻すだけでよい。このレイヤーが
存在するのは `Conv2D` が $(N, C, H, W)$ を話し、`Dense` が $(N, d)$ を話すから
であり、行優先の配置のおかげで変換のコストはゼロである。チャネル優先の平坦化
$(c, h, w) \mapsto (cHW + hW + w)$ は、バッファがすでに持っている並びそのもの
だからである。

---

## 重みの初期化

**実装:** `src/layers.cpp` の `init_std()`、`src/conv2d.cpp` の `conv_init_std()`。

問題になるのは分散の伝播である。独立で平均 0 の項からなる
$y = \sum_{p=1}^{n_{\mathrm{in}}} x_p w_p$ について
$\mathrm{Var}(y) = n_{\mathrm{in}} \cdot \mathrm{Var}(w) \cdot \mathrm{Var}(x)$
が成り立つ。層をまたいで信号の大きさが保たれるのは
$n_{\mathrm{in}} \mathrm{Var}(w) \approx 1$ のときだけである。

| 方式 | $\sigma$ | 根拠 |
|---|---|---|
| **He** | $\sqrt{2 / n_{\mathrm{in}}}$ | ReLU が入力の半分をゼロにして分散を半減させるので、係数 2 で補償する。ReLU と組み合わせる |
| **Xavier**（`Dense`） | $\sqrt{2 / (n_{\mathrm{in}} + n_{\mathrm{out}})}$ | forward の分散保存（$1/n_{\mathrm{in}}$）と backward の分散保存（$1/n_{\mathrm{out}}$）の折衷。活性化関数に中立で、Tanh/Sigmoid と組み合わせる |

バイアスはゼロから始める。重みがランダムであれば、破るべき対称性はもう残って
いない。

`Conv2D` では $n_{\mathrm{in}} = C_{\mathrm{in}} K^2$ である。fan-in はチャネル数
ではなく受容野の体積になる。

> **実装上の注意。** `Conv2D` の `Init::Xavier` は `Dense` が使う
> $\sqrt{2/(n_{\mathrm{in}} + n_{\mathrm{out}})}$ ではなく
> $\sqrt{1 / n_{\mathrm{in}}}$ を使う。ソースのコメントは、リファレンスでは
> 畳み込みが常に He を使うこと、Xavier の分岐は網羅性のために存在することを
> 明記している。Xavier で初期化した畳み込みに依存する場合は、`Dense` の式を
> 前提にせず `conv_init_std()` を先に読むこと。

---

## オプティマイザ

**実装:** `src/optimizer.cpp`。フレームワークごとの比較は
[experiments.md](experiments.md#実験-1最適化手法sgd-vs-momentum-vs-adam)
に詳しい。

1 つのパラメータテンソルについて $g = \partial L/\partial p$、学習率を $\eta$ と
書く。3 つとも要素ごとにその場で更新する。

### SGD

$$p \leftarrow p - \eta g$$

1 行である（`p->axpy(-lr_, *g)`）。

### Momentum

```math
v \leftarrow \mu v - \eta g, \qquad p \leftarrow p + v
```

$v$ は減衰率 $\mu$ で過去の勾配を指数重み付きに足し込んだものである。勾配が一貫
している方向では $v$ は $-\eta g /(1-\mu)$ に近づく。$\mu = 0.9$ なら実効的な
ステップは $1/(1-\mu) = 10$ 倍になる。`compare.cpp` が Momentum に 0.05、SGD に
0.2 の学習率を与えているのはこの係数のためで、*実効的な* ステップを揃えることが
公平な比較になる。

勾配が振動する方向では、連続する寄与が打ち消し合う。これが「ジグザグを抑える」と
いう幾何的な説明の中身である。

### Adam

```math
m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2
```

```math
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}, \qquad
p \leftarrow p - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}
```

移動平均が 2 つある。$m$ は勾配の平均を、$v$ は中心化していない分散を推定する。
$\sqrt{\hat v}$ で割ることで更新がスケール不変になり、すべての勾配を 1000 倍しても
更新量は変わらない。Adam の調整がほとんど要らないのはこのためである。

**バイアス補正。** どちらのモーメントも $0$ から始まるので、初期の平均はゼロ側へ
偏る。$g$ がほぼ定常だとして漸化式の期待値を取ると
$\mathbb{E}[m_t] \approx (1 - \beta_1^{t}) \mathbb{E}[g]$ となるので、
$1 - \beta_1^{t}$ で割れば、その係数がちょうど打ち消される。補正がなければ
$\beta_2 = 0.999$ での最初のステップは $\sqrt{1000} \approx 31$ 倍ほど小さくなって
しまう。補正は $t$ が大きくなるにつれ消えていく。`Adam` がステップカウンタ `t_` を
持ち、パラメータごとではなく `step()` の呼び出しごとに 1 増やすのはこのためである。

**$\varepsilon$ の位置。** このライブラリは平方根の外側、かつバイアス補正の *後* に
置く（$\eta \hat m / (\sqrt{\hat v} + \varepsilon)$）。PyTorch もまったく同じである。
一方 TensorFlow/Keras は補正前の 2 次モーメントに適用する。これは原論文の
「$\hat\varepsilon$」にあたり、初期のステップで $\varepsilon$ を
$1/\sqrt{1 - \beta_2^{t}}$ 倍に膨らませるのと等価である。既定の
$\varepsilon = 10^{-8}$ では差は $10^{-8}$ 程度で、学習には影響しない。これは推測では
なく実測で、`reference/check_optimizer_equivalence.py` が各フレームワークの更新を
閉じた式で再現している。[experiments.md](experiments.md) を参照。

### パラメータごとの状態とポインタ同一性

Momentum と Adam はパラメータごとに状態テンソルを 1 つ持ち、
`std::unordered_map<const Tensor*, Tensor>` にパラメータの **アドレス** をキーとして
保持する。これはリファレンス実装の `id(p)` に対応する C++ 版である。オプティマイザ
を使い回す前に知っておくべき帰結が 2 つある。

- 状態はポインタを初めて見たときに遅延生成される。そのため、オプティマイザは
  パラメータ一覧を事前に知らされなくても任意のモデルで動作する。
- キーが意味を持つのはモデルが生存している間だけである。モデルを破棄して新しく
  作ると同じヒープアドレスが再利用されうるので、持ち越したオプティマイザは古い
  モーメントを静かに引き継いでしまう。`compare.cpp` は実行ごとに新しい
  オプティマイザを構築している。同じようにすること。
  [ARCHITECTURE.md](ARCHITECTURE.md#所有権と生存期間) を参照。

---

## 勾配チェック

**実装:** `tests/test_gradcheck.cpp`。「from scratch」を願望ではなく検証可能な
主張にしているのがこのテストである。

### 推定量

解析的勾配を中心差分と比較する。

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$$

前進差分ではなく中心差分を使うのは、$L(w \pm \epsilon)$ のテイラー展開で 2 次の項が
相殺するからである。打ち切り誤差が前進差分の $O(\epsilon)$ ではなく
$O(\epsilon^2)$ になる。

### $\epsilon$ の選び方

2 つの誤差が逆方向に働く。

- **打ち切り誤差** $\approx \tfrac{1}{6} |L'''| \epsilon^2$ — $\epsilon$ とともに
  小さくなる。
- **丸め誤差** $\approx u |L| / \epsilon$（$u \approx 2.2 \times 10^{-16}$ は倍精度の
  単位丸め）— $\epsilon$ を小さくするほど *大きくなる*。$L(w+\epsilon)$ と
  $L(w-\epsilon)$ の上位桁がますます一致し、引き算でそれが消えてしまうためである。

和を最小化すると $\epsilon^{\ast} \sim u^{1/3} \approx 6 \times 10^{-6}$ となる。
テストが使う $10^{-5}$ はちょうどこの最適点付近で、6〜8 桁程度の一致が期待できる。
ライブラリ全体が `double` である理由もここにある。`float`（$u \approx 10^{-7}$）
では達成しうる一致がせいぜい $10^{-3}$ 程度で、わずかに誤った `backward()` を
捕まえるには緩すぎる。

### 許容誤差

```cpp
CHECK_CLOSE(numeric, analytic[i], 1e-4 * (1.0 + std::fabs(analytic[i])));
```

絶対誤差と相対誤差を混ぜた基準である。定数項がゼロ近傍の勾配に不可能な相対精度を
要求しないようにし、比例項が大きな勾配を形式的に通過させないようにする。

### コストと、サンプリングする理由

1 点の検証につき forward が 2 回必要なので、モデルの $P$ 個のパラメータをすべて
調べると $2P$ 回の forward になる。この小さな MLP でも数千回である。
テストは各パラメータテンソルを `stride = max(1, n/16)` で走査し、テンソルあたり
16 要素程度をサンプリングする。誤った `backward()` が全体の $1/16$ だけ誤っている
ということはまずないので、サンプリングによって検出力が実質的に落ちることはない。

### 折れ点に関する注意

ある probe が ReLU の折れ点をまたぐと — どこかのユニットで $|x| < \epsilon$ と
なると — $L(w+\epsilon)$ と $L(w-\epsilon)$ が異なる線形片の上に乗るため、有限差分は
どちらの劣勾配とも正当に食い違う。ランダムな入力ではその確率はごく小さく（活性化前
の値が $10^{-5}$ 以内にゼロへ近づく必要がある）、テストの固定シードはそれを避けて
いる。折れ点を持つレイヤーを追加して 1 箇所だけ失敗し、しかもシードを変えると
その位置が動く場合は、導出を疑う前にこれを疑うとよい。
[EXTENDING.md](EXTENDING.md#手順-4-勾配チェックで証明する) を参照。

### 何を証明し、何を証明しないか

このテストが証明するのは、各 `backward()` がそれ自身の `forward()` と整合している
ことである。自動微分なら無料で手に入る性質であり、手で導出したときに最も壊れやすい
性質でもある。一方、`forward()` が *意図したもの* を計算しているかは検証 **しない**。
添字の順序を取り違えた畳み込みはそれ自体としては整合しており、このテストを通過する。
`tests/test_tensor.cpp` が手計算の値で forward の正しさを検証しているのは、その
ためである。
