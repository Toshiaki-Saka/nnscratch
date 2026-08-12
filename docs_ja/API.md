# API リファレンス

すべて `namespace nn` に属する。公開 API 全体を取り込むには
`<nnscratch/nnscratch.hpp>` を include する。`<nnscratch/pgm.hpp>` は独立している
（デモ用のユーティリティであり、ネットワーク API の一部ではない）。

計算量は要素演算数で示し、$N$ はバッチサイズとする。値の型はすべて `double`。

| 節 | 内容 |
|---|---|
| [Tensor](#tensor) | 記憶域、形状の問い合わせ、要素アクセス、線形代数 |
| [Layer / ParamGrad](#layer--paramgrad) | すべてのレイヤーが実装するインターフェース |
| [レイヤー](#レイヤー) | `Dense`, `Flatten`, `Conv2D`, `Init` |
| [活性化関数](#活性化関数) | `ReLU`, `Tanh`, `Sigmoid`, `softmax()` |
| [損失関数](#損失関数) | `SoftmaxCrossEntropy`, `one_hot()` |
| [Model](#model) | 逐次コンテナ、予測、正解率 |
| [オプティマイザ](#オプティマイザ) | `SGD`, `Momentum`, `Adam` |
| [学習](#学習) | `train()`, `TrainConfig`, `History` |
| [データセット](#データセット) | `load_digits()`, `DigitsData` |
| [Rng](#rng) | 再現可能な乱数 |
| [PGM 出力](#pgm-出力) | `write_pgm()`, `write_pgm_grid()` |
| [バージョンマクロ](#バージョンマクロ) | `NNSCRATCH_VERSION_*` |

---

## Tensor

`<nnscratch/tensor.hpp>` — 形状ベクトルを持つ、密で行優先の `double` 配列。この
ライブラリが必要とするのはランク 2（`(rows, cols)`）とランク 4（`(N, C, H, W)`）
だけなので、意図的に汎用の n 次元配列型にはしていない。

### 構築

```cpp
Tensor();                                                  // 空、ランク 0
Tensor(std::size_t rows, std::size_t cols);                // ゼロ初期化のランク 2
explicit Tensor(std::vector<std::size_t> shape);           // 任意ランク、ゼロ初期化
static Tensor from(std::vector<std::size_t> shape, std::vector<double> data);
static Tensor zeros(std::vector<std::size_t> shape);
static Tensor zeros_like(const Tensor& t);
```

すべてのコンストラクタがゼロ初期化する。`from()` は既存のフラットなバッファ
（行優先）を受け取り、`prod(shape) != data.size()` なら
**`std::invalid_argument` を送出する**。

```cpp
Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});  // [[1,2,3],[4,5,6]]
Tensor z = Tensor::zeros({4, 1, 8, 8});
```

### 形状の問い合わせ

| シグネチャ | 戻り値 | 送出 |
|---|---|---|
| `const std::vector<std::size_t>& shape() const` | 形状全体 | — |
| `std::size_t rank() const` | 次元数 | — |
| `std::size_t size() const` | 総要素数 | — |
| `std::size_t dim(std::size_t i) const` | 軸 `i` の大きさ | `std::out_of_range`（`.at()` を使う） |
| `std::size_t rows() const` | `shape()[0]` | ランクが 2 でなければ `std::logic_error` |
| `std::size_t cols() const` | `shape()[1]` | ランクが 2 でなければ `std::logic_error` |

### 要素アクセス

```cpp
double& operator()(std::size_t i, std::size_t j);
double  operator()(std::size_t i, std::size_t j) const;
std::vector<double>& data();
const std::vector<double>& data() const;
```

> **境界検査はない。** `operator()` は `data_[i * shape_[1] + j]` を直接計算し、
> ランクの検証もしない。ランク 4 のテンソルに対して呼ぶと、誤ったストライドで
> 黙って添字を引く。ランク 4 の要素には `data()` とオフセットを使うこと。
> ```cpp
> // (N, C, H, W) テンソルの要素 (n, c, h, w)
> t.data()[((n * C + c) * H + h) * W + w];
> ```

### 形状変換

```cpp
Tensor reshape(std::vector<std::size_t> new_shape) const;  // サイズが違えば送出
Tensor flatten_batch() const;                               // (N, ...) -> (N, prod(残り))
```

どちらも **コピーを返す**。このライブラリにビューは存在しない。`flatten_batch` は
ランク 0 のテンソルに対して `std::out_of_range` を送出する（`shape_.at(0)` を
呼ぶため）。

### 線形代数

| シグネチャ | 結果 | 計算量 | 送出 |
|---|---|---|---|
| `Tensor transpose() const` | $A^\top$ | $O(mn)$ | ランクが 2 でなければ `std::logic_error` |
| `Tensor sum_rows() const` | 軸 0 で合計した $(1, \text{cols})$ | $O(mn)$ | ランクが 2 でなければ `std::logic_error` |
| `Tensor map(const std::function<double(double)>&) const` | 要素ごとの $f$ | $O(\text{size})$ | — |
| `void add_row_vector(const Tensor& bias)` | その場でブロードキャスト加算 | $O(mn)$ | ランク違いで `logic_error`、幅違いで `invalid_argument` |
| `void axpy(double alpha, const Tensor& other)` | その場で `*this += alpha * other` | $O(\text{size})$ | サイズ不一致で `std::invalid_argument` |

`add_row_vector` は形状 `(1, cols)` または `(cols)` のバイアスを受け付ける。検査
しているのは `bias.size() == cols` だけである。

`axpy` は形状ではなく総サイズを比較するので、`(1, 6)` を `(2, 3)` に足すことも
できてしまう。

### 自由関数

```cpp
Tensor matmul(const Tensor& a, const Tensor& b);   // (m,k) x (k,n) -> (m,n)

Tensor operator+(const Tensor& a, const Tensor& b);  // 要素ごと。形状は完全一致が必要
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);  // アダマール積
Tensor operator*(double s, const Tensor& a);         // スカラー倍（スカラーは左側）

std::ostream& operator<<(std::ostream& os, const Tensor& t);  // 形状だけを出力
```

`matmul` は `ikj` 順の素朴な $O(mkn)$ 3 重ループである
（[PERFORMANCE.md](PERFORMANCE.md#matmul) 参照）。どちらかがランク 2 でなければ
`std::logic_error` を、内側の次元が合わなければ `std::invalid_argument` を送出する。

要素ごとの演算子は **形状の完全一致** を要求する。ブロードキャストはない。バイアス
のようなブロードキャストには `add_row_vector` を使う。

`double * Tensor` はあるが `Tensor * double` はない。`Tensor * Tensor` がアダマール
積なので、その隣にスカラー倍を両方向で許すと混乱を招くからである。

`operator<<` は `Tensor(shape=[2,3])` のように形状だけを出力し、値は出力しない。

---

## Layer / ParamGrad

`<nnscratch/layer.hpp>`

```cpp
struct ParamGrad {
    Tensor* param;   // レイヤーが所有する。オプティマイザがその場で書き換える
    Tensor* grad;    // 対応する勾配。backward() のたびに上書きされる
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad_out) = 0;
    virtual std::vector<ParamGrad> params_and_grads() { return {}; }
};
```

`param` の **アドレス** が、オプティマイザにおけるパラメータごとの状態のキーに
なる。それに伴う生存期間の規則は
[ARCHITECTURE.md](ARCHITECTURE.md#所有権と生存期間) を参照。

独自のレイヤーを実装するには [EXTENDING.md](EXTENDING.md) を参照。

---

## レイヤー

`<nnscratch/layers.hpp>`

### Init

```cpp
enum class Init { He, Xavier };
```

| 値 | `Dense` の $\sigma$ | `Conv2D` の $\sigma$ | 組み合わせる相手 |
|---|---|---|---|
| `He` | $\sqrt{2/n_{\mathrm{in}}}$ | $\sqrt{2/n_{\mathrm{in}}}$ | ReLU |
| `Xavier` | $\sqrt{2/(n_{\mathrm{in}} + n_{\mathrm{out}})}$ | $\sqrt{1/n_{\mathrm{in}}}$ | Tanh, Sigmoid |

`Conv2D` では $n_{\mathrm{in}} = C_{\mathrm{in}} K^2$ である。2 つの `Xavier` の式が
異なる点に注意（[MATH.md](MATH.md#重みの初期化) の実装上の注意を参照）。バイアスは
常にゼロから始まる。

### Dense

```cpp
Dense(std::size_t n_in, std::size_t n_out, Rng& rng, Init init = Init::Xavier);

Tensor forward(const Tensor& x) override;         // (N, n_in)  -> (N, n_out)
Tensor backward(const Tensor& grad_out) override; // (N, n_out) -> (N, n_in)
std::vector<ParamGrad> params_and_grads() override;  // {{&W, &dW}, {&b, &db}}

Tensor& weight() noexcept;  // (n_in, n_out)
Tensor& bias()   noexcept;  // (1, n_out)
```

$Y = XW + b$。パラメータ数は
$n_{\mathrm{in}} n_{\mathrm{out}} + n_{\mathrm{out}}$。計算量は forward が
$O(N \cdot n_{\mathrm{in}} \cdot n_{\mathrm{out}})$、backward も定数倍を除いて同じ。
入力を値でキャッシュする。

`weight()` は可変参照を返す。`from_scratch.cpp` が第 1 層の学習済み特徴を描画して
いるのはこの経路である。書き込みも許されており、重みを手で設定する唯一の手段でも
ある（シリアライズは存在しない）。

### Flatten

```cpp
Tensor forward(const Tensor& x) override;         // (N, d1, ..., dk) -> (N, prod di)
Tensor backward(const Tensor& grad_out) override; // キャッシュした入力形状を復元
```

パラメータなし。$O(\text{size})$（コピー）。`Conv2D` と `Dense` の間に必須。

### Conv2D

```cpp
Conv2D(std::size_t in_c, std::size_t out_c, std::size_t k,
       std::size_t stride, std::size_t pad, Rng& rng, Init init = Init::He);

Tensor forward(const Tensor& x) override;   // (N,C,H,W) -> (N,out_c,OH,OW)
Tensor backward(const Tensor& grad_out) override;
std::vector<ParamGrad> params_and_grads() override;  // {{&W, &dW}, {&b, &db}}

Tensor& weight() noexcept;  // (out_c, in_c, k, k)
```

出力サイズ:

```math
OH = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1, \qquad
OW = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1
```

- パラメータ数: $C_{\mathrm{out}}(C_{\mathrm{in}}K^2 + 1)$。
- forward: 時間 $O(N \cdot OH \cdot OW \cdot C_{\mathrm{in}} K^2 \cdot
  C_{\mathrm{out}})$。加えて $N \cdot OH \cdot OW \cdot C_{\mathrm{in}} K^2$ 要素の
  im2col バッファを backward のためにキャッシュする。
- ゼロパディングのみ、正方カーネルのみ、両軸で同じストライドのみ。

**前提条件（すべてが強制されるわけではない）:**

| 条件 | 違反したときの挙動 |
|---|---|
| 入力がランク 4 | `std::logic_error` を送出 |
| `stride >= 1` | ゼロ除算 |
| `k <= H + 2*pad` かつ `k <= W + 2*pad` | **符号なしのアンダーフロー**。`H + 2P - K` が巨大な値に回り込み、異常な `OH` になって、明確なエラーではなく確保の失敗として現れる |
| `x.dim(1) == in_c` | 検査なし。誤った形状の `col_` ができ、`matmul` の内側次元エラーになる |

`backward()` の前に `forward()` を呼んでおく必要がある。backward は
キャッシュされた `col_`、`x_shape_`、`oh_`、`ow_` を読む。

---

## 活性化関数

`<nnscratch/activations.hpp>`

```cpp
Tensor softmax(const Tensor& logits);   // (N, K) -> (N, K)、各行の和は 1

class ReLU    final : public Layer { /* forward, backward */ };
class Tanh    final : public Layer { /* forward, backward */ };
class Sigmoid final : public Layer { /* forward, backward */ };
```

3 つとも要素ごとで形状を保存し、パラメータを持たず、任意のランクで動作する。
往復とも $O(\text{size})$。

| クラス | $f(x)$ | キャッシュ | backward |
|---|---|---|---|
| `ReLU` | $\max(0, x)$ | 0/1 マスク | $G \odot \text{mask}$ |
| `Tanh` | $\tanh x$ | 出力 $y$ | $G \odot (1 - y^2)$ |
| `Sigmoid` | $1/(1+e^{-x})$ | 出力 $y$ | $G \odot y(1-y)$ |

$x = 0$ において `ReLU` は劣勾配 $0$ を採用し、PyTorch と TensorFlow に一致する。

自由関数の `softmax()` はランク 2 を要求し（`rows()`/`cols()` を呼ぶ）、安定性の
ために各行の最大値を引く。直接使う場面は多くない。学習では損失と融合した
`SoftmaxCrossEntropy` を使う。学習済みモデルから較正された確率を取り出したいときに
呼ぶ。

```cpp
Tensor probs = softmax(model.forward(x_test));   // (N, 10)、各行の和は 1
```

---

## 損失関数

`<nnscratch/loss.hpp>`

```cpp
class SoftmaxCrossEntropy {
public:
    double forward(const Tensor& logits, const Tensor& targets_onehot);
    Tensor backward() const;   // (P - Y) / N
};

Tensor one_hot(const std::vector<int>& labels, std::size_t num_classes);
```

`forward` は確率ではなく **ロジット** を受け取り（softmax は内部で行う）、バッチ
平均の cross-entropy を返す。確率と教師をキャッシュするので、`backward()` は引数を
取らず、`forward` の後でのみ呼べる。往復とも $O(NK)$。融合された勾配が
$P - Y$ を $N$ で割ったものになる理由は
[MATH.md](MATH.md#softmax--cross-entropy-の融合) を参照。

報告される損失は $\log(p + 10^{-9})$ を使うので、自信を持って外した予測でも無限大
ではなく約 20.7 になる。勾配には影響しない。

`one_hot` は $(\text{labels.size}, K)$ のテンソルを作る。**ラベルは $[0, K)$ に
収まっていなければならない** — 境界検査はない。

```cpp
SoftmaxCrossEntropy loss_fn;
Tensor Y = one_hot(y_train, 10);
double L = loss_fn.forward(model.forward(x_train), Y);
model.backward(loss_fn.backward());
```

---

## Model

`<nnscratch/model.hpp>` — 逐次的な積み重ね。`nn.Sequential` に相当する。

```cpp
Model();
explicit Model(std::vector<std::unique_ptr<Layer>> layers);

template <class L, class... Args> L& add(Args&&... args);  // その場で構築
Layer& push(std::unique_ptr<Layer> layer);                 // 構築済みを追加

Tensor forward(const Tensor& x);
void   backward(const Tensor& grad);
std::vector<ParamGrad> params_and_grads();

std::vector<int> predict(const Tensor& x);                       // 行ごとの argmax
double accuracy(const Tensor& x, const std::vector<int>& labels);

Layer& layer(std::size_t i);          // std::out_of_range を送出
std::size_t size() const noexcept;
```

`add<L>(...)` は引数を `L` のコンストラクタへ転送し、構築したレイヤーへの参照を
返す。この参照はモデルの生存期間中は有効である。

```cpp
Rng rng(42);
Model net;
Dense& first = net.add<Dense>(64, 64, rng, Init::He);   // 参照を保持しておく
net.add<ReLU>();
net.add<Dense>(64, 10, rng, Init::He);
// ... 学習 ...
const Tensor& W = first.weight();                        // まだ有効
```

`push()` はレイヤーの型が実行時に決まる場合のためのものである。`compare.cpp` は
活性化関数のファクトリと組み合わせて使い、同じビルダーから ReLU / Tanh / Sigmoid の
各版を作っている。

`backward(grad)` はレイヤーを逆順にたどり、最後の勾配は捨てる（入力に対する
$\partial L/\partial X$ は誰も必要としない）。戻り値はない。

`predict` と `accuracy` はそれぞれ forward を 1 回走らせるので、両方呼べば 2 回に
なる。`accuracy` は `labels.size() == x.dim(0)` を前提とする。

`Model` は **ムーブ可能だがコピー不可** である（`Layer` に `clone()` がない）。

---

## オプティマイザ

`<nnscratch/optimizer.hpp>`

```cpp
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<ParamGrad>& pgs) = 0;
};

class SGD      final : public Optimizer { explicit SGD(double lr); };
class Momentum final : public Optimizer { explicit Momentum(double lr, double mu = 0.9); };
class Adam     final : public Optimizer { explicit Adam(double lr, double b1 = 0.9,
                                                        double b2 = 0.999, double eps = 1e-8); };
```

| | 更新式 | パラメータごとの状態 | このリポジトリでの `lr` の目安 |
|---|---|---|---|
| `SGD` | $p \mathrel{-}= \eta g$ | なし | 0.2 – 0.5 |
| `Momentum` | $v \leftarrow \mu v - \eta g$、$p \mathrel{+}= v$ | 速度テンソル 1 本 | 0.05 |
| `Adam` | [MATH.md](MATH.md#adam) 参照 | モーメント 2 本 + 共有のステップカウンタ | 0.01 |

Momentum の実効ステップは SGD のおよそ $1/(1-\mu)$ 倍になる。`compare.cpp` で公平な
比較になる学習率がおよそその係数だけ違うのはこのためである。

`step()` はすべての組に 1 回ずつ更新を適用する。唯一の書き換えを伴う呼び出しである。
パラメータごとの状態は `param` のアドレスをキーに遅延生成される。

> **`Momentum` や `Adam` のインスタンスを別のモデルに使い回さないこと。**
> アドレスをキーにした状態は、モデル破棄後に再利用されたヒープアドレスと衝突し、
> 古いモーメントを静かに新しい実行へ持ち込みうる。モデルごとに新しい
> オプティマイザを構築すること。

Adam のステップカウンタ `t_` はパラメータごとではなく `step()` の呼び出しごとに
1 増える。これがバイアス補正を正しくしている。

---

## 学習

`<nnscratch/training.hpp>`

```cpp
struct TrainConfig {
    int          epochs     = 40;
    std::size_t  batch_size = 32;
    std::uint64_t batch_seed = 123;   // ミニバッチの順序を固定する
    bool         verbose    = false;  // エポックごとの指標を標準出力へ
};

struct History {
    std::vector<int>    epoch;
    std::vector<double> loss;        // 学習データ「全体」の平均 CE
    std::vector<double> train_acc;
    std::vector<double> test_acc;
};

History train(Model& model, Optimizer& opt,
              const Tensor& x_train, const std::vector<int>& y_train,
              const Tensor& x_test,  const std::vector<int>& y_test,
              const TrainConfig& cfg);
```

損失を `SoftmaxCrossEntropy` に固定したミニバッチ学習。知っておくべき挙動:

- **`History` の行数は `epochs + 1` である。** エポック 0 が更新前の未学習
  ネットワークを記録し、「ランダム → 学習済み」の全軌跡を与える。
- **指標は各エポック後に学習データ全体で計算される。** ミニバッチの平均ではないので、
  `loss` は正真正銘のエポック終了時の損失である。その代わりエポックごとに全データの
  forward が 2 回余分に走り、すべてのレイヤーの forward キャッシュを上書きする。
  後に `backward()` が続かないので無害である。
- **`num_classes` は `max(y_train) + 1` として推論される。** 学習ラベルに現れない
  クラスがあると出力幅が縮むので、最終 `Dense` の形状と一致している必要がある。
- **エポック最後のミニバッチは `batch_size` より小さいことがある。**
- **シャッフルは呼び出しローカルな `Rng`**（`cfg.batch_seed` でシード）を使うので、
  バッチ順序はモデル初期化用の RNG と独立である。
- `x_train` はランク 2 でもランク 4 でもよい。バッチ切り出しが後続の次元を保つので、
  同じループで MLP も CNN も学習できる。

```cpp
TrainConfig cfg;
cfg.epochs = 60;
cfg.verbose = true;
History h = train(net, opt, data.train.flat, data.train.labels,
                  data.test.flat, data.test.labels, cfg);
std::printf("best test acc: %.1f%%\n",
            *std::max_element(h.test_acc.begin(), h.test_acc.end()) * 100.0);
```

---

## データセット

`<nnscratch/dataset.hpp>`

```cpp
struct DigitsData {
    struct Split {
        Tensor flat;              // (N, 64)      MLP 用
        Tensor img;               // (N, 1, 8, 8) CNN 用
        std::vector<int> labels;  // N 個、0..9
    };
    Split train;
    Split test;
};

DigitsData load_digits(const std::string& csv_path,
                       double train_frac = 0.8,
                       std::uint64_t split_seed = 0);
```

2 つのビューは *同じ* サンプルを保持するので、読み込み直さずに MLP と CNN を
切り替えられる。ピクセルは元の 0–16 から 16 で割って $[0, 1]$ に正規化される。

分割は `split_seed` でシードした Fisher–Yates の置換を行い、
`floor(N * train_frac)` で切る。同梱の 1797 レコードと既定値では
**train 1437 / test 360** になる。

ファイルを開けない場合、レコードが 1 つもない場合、65 フィールドでない行がある場合
に `std::runtime_error` を送出する。コメント行（`#`）と `p0,...` のヘッダ行は
読み飛ばされる。書式の詳細は [DATA_FORMATS.md](DATA_FORMATS.md) を参照。

---

## Rng

`<nnscratch/rng.hpp>` — `std::mt19937_64` のヘッダのみのラッパー。

```cpp
explicit Rng(std::uint64_t seed = 42);
void reseed(std::uint64_t seed);

double normal();                                              // N(0,1) を 1 つ
Tensor normal(std::vector<std::size_t> shape, double std_dev); // N(0, std^2) のテンソル
std::vector<std::size_t> permutation(std::size_t n);           // Fisher-Yates、O(n)
```

比較実験を公平にしているのが `reseed()` である。各モデルを構築する前に同じシードへ
巻き戻せば、初期重みが一致する。

```cpp
Rng rng(42);
rng.reseed(42);  Model a = build_mlp(rng, relu);
rng.reseed(42);  Model b = build_mlp(rng, tanh_);   // a と同じ初期重み
```

結果は特定のツールチェーンではビット単位で再現するが、**ツールチェーンをまたぐと
再現しない**。標準はエンジンを厳密に規定するが、`std::normal_distribution` や
`std::uniform_int_distribution` は規定しないからである。
[ARCHITECTURE.md](ARCHITECTURE.md#決定性) を参照。

---

## PGM 出力

`<nnscratch/pgm.hpp>` — ヘッダのみ。デモが使う。`nnscratch.hpp` には含まれない。

```cpp
void write_pgm(const std::string& path, const std::vector<double>& pixels,
               std::size_t width, std::size_t height);

void write_pgm_grid(const std::string& path,
                    const std::vector<std::vector<double>>& cells,
                    std::size_t cell, std::size_t cols, std::size_t pad = 1);
```

`write_pgm` はバイナリ PGM（`P5`）を書き出す。値を $[0,1]$ にクランプし、0–255 へ
量子化する。`write_pgm_grid` は同じ大きさの正方画像を 1 枚のキャンバスに並べ、
**セルごとに独立して min–max 正規化する**。重みの絶対的な大きさによらず構造が
見えるようにするためだが、その結果として明るさは 1 つのセルの *内部* でのみ比較でき、
セル *どうし* では比較できない。

どちらの関数も I/O エラーを報告しない。オープンに失敗すると黙って何も書かない。
配置の詳細と閲覧方法は
[DATA_FORMATS.md](DATA_FORMATS.md#pgm-画像) を参照。

---

## バージョンマクロ

```cpp
#define NNSCRATCH_VERSION_MAJOR 0
#define NNSCRATCH_VERSION_MINOR 1
#define NNSCRATCH_VERSION_PATCH 0
```

`<nnscratch/nnscratch.hpp>` が定義し、`CMakeLists.txt` の `project(VERSION)` と
同期している。インストールされる CMake パッケージのバージョンファイルは
`SameMajorVersion` 互換性を使うので、0.x ではマイナーバージョンの更新はすべて
非互換として扱われる。
