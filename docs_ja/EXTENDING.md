# nnscratch を拡張する

ここには自動微分がない。したがって新しいレイヤーを足すとは、`backward()` を手で
導出し、それが正しいことを勾配チェックで証明することを意味する。このドキュメントは
その手順を追い、続いてオプティマイザ、損失関数、データセットの追加を扱う。

前提として、[MATH.md](MATH.md) の連鎖律の規約と
[ARCHITECTURE.md](ARCHITECTURE.md#レイヤーの抽象) のレイヤー契約を読んでおくこと。

- [レイヤーを追加する](#レイヤーを追加する)
- [実装例: LeakyReLU](#実装例-leakyrelu)
- [実装例: MaxPool2D](#実装例-maxpool2d)
- [オプティマイザを追加する](#オプティマイザを追加する)
- [損失関数を追加する](#損失関数を追加する)
- [データセットを追加する](#データセットを追加する)
- [守るべき原則](#守るべき原則)

---

## レイヤーを追加する

5 つの手順がある。重要なのは手順 4 である。

### 手順 1: backward を紙の上で導出する

forward の写像を添字で書き、

$$\frac{\partial L}{\partial \theta} = \sum_k \frac{\partial L}{\partial Y_k}\frac{\partial Y_k}{\partial \theta}$$

を適用する。$\theta$ が影響を与えた **すべての** 出力要素について和を取ること。
その和を行列積、リダクション、または scatter-add へまとめる。最後に形状で検算する。
$\partial L/\partial W$ は $W$ と同じ形状でなければならず、たいていその形を作る
縮約は 1 通りしかない。

$1/N$ のバッチ平均は損失にあるので、レイヤー側で $N$ で割らないこと。

### 手順 2: 宣言する

パラメータを持つレイヤーは `include/nnscratch/layers.hpp` に、要素ごとのものは
`activations.hpp` にクラスを追加する。メンバにはパラメータ、その勾配、そして
backward が必要とするキャッシュを持たせる。

```cpp
class MyLayer final : public Layer {
public:
    MyLayer(std::size_t n_in, std::size_t n_out, Rng& rng, Init init = Init::He);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamGrad> params_and_grads() override;   // パラメータがなければ不要

private:
    Tensor W_, b_;     // パラメータ
    Tensor dW_, db_;   // 勾配。形状は同じ
    Tensor cache_;     // backward が必要とするもの
};
```

キャッシュするのは *必要最小限* のものにする。0/1 マスクは入力より軽く、導関数が
出力で書けるなら（Tanh や Sigmoid のように）出力のほうが入力より良い。

### 手順 3: 実装する

`src/layers.cpp`、`src/activations.cpp`、あるいは長くなるなら独立したファイルに書く。
`Conv2D` は `src/conv2d.cpp` を単独で持っている。新しい `.cpp` は `CMakeLists.txt`
の `add_library(nnscratch ...)` の一覧に追加する必要がある。

```cpp
Tensor MyLayer::forward(const Tensor& x) {
    if (x.rank() != 2) throw std::logic_error("MyLayer expects a rank-2 input");
    cache_ = /* ... */;   // その場での変更より「前に」キャッシュする
    return /* ... */;
}

Tensor MyLayer::backward(const Tensor& grad_out) {
    dW_ = /* ... */;      // 蓄積ではなく代入する
    db_ = /* ... */;
    return /* 入力に対する勾配。forward の x とまったく同じ形状 */;
}

std::vector<ParamGrad> MyLayer::params_and_grads() {
    return {{&W_, &dW_}, {&b_, &db_}};
}
```

譲れない点:

- **勾配は代入する。蓄積しない。** このライブラリに `zero_grad()` はないので、`+=`
  にするとステップをまたいで静かに積み上がる
  （[ARCHITECTURE.md](ARCHITECTURE.md#不変条件)）。
- **`forward()` の入力とまったく同じ形状のテンソルを返す。**
- **`params_and_grads()` からは安定したポインタを返す。** 常に `&member_` であり、
  ローカル変数や再確保されうるベクタの要素のアドレスであってはならない。
- **必要なランクを検証して送出する。** ストライドを取り違えた読み出しが黙って進む
  より、例外のほうがはるかにデバッグしやすい。

### 手順 4: 勾配チェックで証明する

**この手順は必須である**（[CONTRIBUTING.md](../CONTRIBUTING.md)）。
`tests/test_gradcheck.cpp` にブロックを追加する。

```cpp
{
    Rng rng(17);
    Model m;
    m.add<Dense>(6, 5, rng, Init::Xavier);
    m.add<MyLayer>(5, 4, rng, Init::He);   // 検証対象のレイヤー
    m.add<Dense>(4, 3, rng, Init::Xavier);

    Tensor X = rng.normal({4, 6}, 1.0);
    std::vector<int> y = {0, 2, 1, 1};
    grad_check(m, X, y, 3);
}
```

指針:

- **`Dense` で挟む。** そうすれば、最初の `Dense` に非ゼロの勾配が届くためには自作
  レイヤーを *通って* 流れる必要があるので、パラメータの勾配と入力に対する勾配の
  両方が検証される。
- **小さく保つ。** ユニット 4〜6、サンプル 3〜5 程度。probe 1 点ごとに forward が
  2 回走る。
- **扱いにくい設定こそカバーする。** ストライド、パディング、レートなどを持つ
  レイヤーなら既定値以外も試す。バグが隠れるのはたいてい既定値のほうである
  （[TESTING.md](TESTING.md#カバレッジの穴) の `Conv2D` の項を参照）。
- **折れ点を持つレイヤーなら**（ReLU のような区分的なもの）、活性化前の値が折れ点の
  $10^{-5}$ 以内に来ないシードを選ぶ。さもないと有限差分が 2 つの線形片をまたぎ、
  正当に食い違う（[MATH.md](MATH.md#折れ点に関する注意)）。

そのうえで実行し、[TESTING.md](TESTING.md#失敗の切り分け) の対応表を読む。解析値と
数値微分の比が、たいていそのままバグの名前になる。

### 手順 5: 対応関係を文書化する

トップレベルの [README.md](../README.md#numpy--c--framework-correspondence) にある
`numpy → C++ → フレームワーク` の表に行を足す。各部品をフレームワークの対応物へ
結びつけることがこのプロジェクトの目的そのものなので、その行がないレイヤーは
半分しか届いていない。

---

## 実装例: LeakyReLU

最小の完全な例。要素ごとで、パラメータを持たない。

**数式。** $x > 0$ なら $f(x) = x$、そうでなければ $\alpha x$。したがって
$f'(x)$ は $1$ か $\alpha$ である。要素ごとなのでヤコビアンは対角行列になり、
backward はアダマール積になる。ReLU の構造で $0$ を $\alpha$ に置き換えただけである。

`include/nnscratch/activations.hpp`:

```cpp
/// LeakyReLU: x > 0 なら x、そうでなければ alpha * x。負側にもわずかな勾配を
/// 残すので、ユニットが恒久的に死ぬことがない。
class LeakyReLU final : public Layer {
public:
    explicit LeakyReLU(double alpha = 0.01) : alpha_(alpha) {}
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;

private:
    double alpha_;
    Tensor slope_;   // x > 0 なら 1、そうでなければ alpha
};
```

`src/activations.cpp`:

```cpp
Tensor LeakyReLU::forward(const Tensor& x) {
    const double a = alpha_;
    slope_ = x.map([a](double v) { return v > 0.0 ? 1.0 : a; });
    return x * slope_;          // アダマール積: どちらの分岐でも厳密
}

Tensor LeakyReLU::backward(const Tensor& grad_out) {
    return grad_out * slope_;
}
```

`params_and_grads()` はオーバーライドしない。基底クラスが空のリストを返し、学習可能
な状態を持たないレイヤーにとってそれが正しい。

`tests/test_gradcheck.cpp` のテストブロック:

```cpp
{
    Rng rng(23);
    Model m;
    m.add<Dense>(6, 5, rng, Init::He);
    m.add<LeakyReLU>(0.1);      // alpha を大きめにすると分岐の誤りが目立つ
    m.add<Dense>(5, 3, rng, Init::Xavier);

    Tensor X = rng.normal({4, 6}, 1.0);
    std::vector<int> y = {1, 0, 2, 1};
    grad_check(m, X, y, 3);
}
```

既定の 0.01 ではなく $\alpha = 0.1$ を使うのは意図的である。backward が誤った分岐を
取った場合の差が信号に対して 100 倍ではなく 10 倍下になる、という以上に重要なのは、
$\alpha$ が小さすぎると負側の勾配が有限差分のノイズ床に埋もれて誤りを隠しかねない
ことである。

---

## 実装例: MaxPool2D

もう少し難しい場合のスケッチ。ランク 4、パラメータなし、しかし backward が
*経路の振り分け* になる。

**数式。** forward は $K \times K$ の窓ごとに最大値を取る。出力に寄与するのは
argmax だけなので、

```math
\frac{\partial L}{\partial X_{n,c,h,w}} =
\sum_{\text{argmax が } (h,w) \text{ である窓 } (i,j)} G_{n,c,i,j}
```

となり、最大でない要素の勾配はちょうどゼロになる。したがって:

- `forward` は出力位置ごとに **argmax のフラット添字** をキャッシュする（出力要素
  あたり `std::size_t` 1 つで済み、入力をキャッシュするよりずっと安い）。
- `backward` は入力と同じ形状のゼロテンソルを確保し、上流の勾配を記録しておいた
  argmax の位置へ scatter-add する。窓が重なる（ストライド < K）とたんに `+=` が
  効いてくる。理由は col2im と同じである
  （[MATH.md](MATH.md#col2im-なぜ-scatter-add-なのか)）。

プーリング特有の罠が 2 つある。

- **同値。** 窓の中に厳密に等しい要素が 2 つあると、真の関数はそこで微分不可能に
  なる。フレームワークと同様に決定的に一方（先に現れたほう）を選び、勾配チェックでは
  シードを固定した乱数入力を使って同値が起きないようにする。
- **折れ。** max は区分的に線形なので、窓内の 2 要素が $\epsilon$ 以内に近づくと
  勾配チェックが境界をまたぐ。対処は ReLU と同じである。

`layers.hpp` の `Conv2D` の隣に置き、画面 1 枚を超えるなら独立した `.cpp` に実装し、
そのファイルを `CMakeLists.txt` に登録し、ストライド $=K$（重なりなし）と
ストライド $< K$（重なりあり）の両方で勾配チェックすること。この 2 つは実際に別の
コード経路である。

---

## オプティマイザを追加する

こちらはずっと簡単である。導出は不要で、勾配チェックの対象にもならない。

```cpp
// include/nnscratch/optimizer.hpp
class RMSProp final : public Optimizer {
public:
    explicit RMSProp(double lr, double rho = 0.9, double eps = 1e-8)
        : lr_(lr), rho_(rho), eps_(eps) {}
    void step(const std::vector<ParamGrad>& pgs) override;

private:
    double lr_, rho_, eps_;
    std::unordered_map<const Tensor*, Tensor> s_;   // パラメータの「アドレス」をキーに
};
```

```cpp
// src/optimizer.cpp
void RMSProp::step(const std::vector<ParamGrad>& pgs) {
    for (const auto& [p, g] : pgs) {
        auto it = s_.find(p);
        if (it == s_.end()) it = s_.emplace(p, Tensor::zeros_like(*p)).first;
        Tensor& s = it->second;
        for (std::size_t i = 0; i < s.size(); ++i) {
            const double gi = g->data()[i];
            s.data()[i] = rho_ * s.data()[i] + (1.0 - rho_) * gi * gi;
            p->data()[i] -= lr_ * gi / (std::sqrt(s.data()[i]) + eps_);
        }
    }
}
```

守るべき点:

- **状態はパラメータのアドレスをキーに遅延生成する。** 上のパターンをそのまま
  写すこと。これがあるおかげで、オプティマイザはパラメータ一覧を事前に知らされ
  なくても任意のモデルで動く。
- **バイアス補正のためのステップカウンタは、パラメータではなくオプティマイザが持つ。**
  `Adam` と同様に `step()` の呼び出しごとに 1 増やす。
- **その場で更新する。** 単純な `p += alpha * g` には `axpy()` が使える。要素ごとに
  非線形なものは明示的なループが必要になる。

そのうえで `tests/test_optimizer.cpp` に $(w-3)^2$ を最小化するブロックを追加し、
[experiments.md](experiments.md) のオプティマイザ表に行を足す。

---

## 損失関数を追加する

`SoftmaxCrossEntropy` は意図的に `Layer` **ではない**。教師を受け取ってスカラーを
返すので、インターフェースの形が違う。

```cpp
double forward(const Tensor& predictions, const Tensor& targets);
Tensor backward() const;   // forward でキャッシュした d(loss)/d(predictions)
```

新しい損失も同じ形に従うこと。また、最終段の活性化関数を損失に融合すると勾配が
簡単にならないかを検討するとよい。softmax + cross-entropy が $P - Y$ に collapse
するのは、まさにそれによる（[MATH.md](MATH.md#collapse)）。

なお `train()` は `SoftmaxCrossEntropy` をハードコードしている。別の損失を使うには
自分でループを書くことになる。`src/training.cpp` は 86 行で、コピーして手を入れる
ことを想定している。

---

## データセットを追加する

`load_digits` と同じ形にそろえる。同じサンプルを **2 つのビュー** — ランク 2
（MLP 用）とランク 4（CNN 用） — で返し、`std::vector<int>` のラベルを添え、シード
付きの置換で分割して再現性を確保する。

```cpp
struct MyData {
    struct Split { Tensor flat, img; std::vector<int> labels; };
    Split train, test;
};
MyData load_mine(const std::string& path, double train_frac = 0.8,
                 std::uint64_t split_seed = 0);
```

入力はおおむね $[0,1]$ か、平均 0・分散 1 に正規化すること。初期化の式は入力が
1 のオーダーであることを前提にしている（[MATH.md](MATH.md#重みの初期化)）。I/O や
書式の問題では、`load_digits` と同じくパスをメッセージに含めた
`std::runtime_error` を送出する。

依存関係を増やさないこと。同梱のローダは `std::ifstream` と `std::stod` だけで
できている。

---

## 守るべき原則

[CONTRIBUTING.md](../CONTRIBUTING.md) からの再掲である。受け入れられる拡張の形を
決めているので、ここにも書いておく。

1. **実行時依存関係を持たない。** ライブラリもテストもデモも C++20 標準ライブラリ
   だけで完結すること。
2. **明快さは巧妙さに優る。** これは教材のライブラリである。数式をそのまま写した
   読めるループのほうが、それを覆い隠す速いループより価値がある。その速度の代償が
   どれほどかは [PERFORMANCE.md](PERFORMANCE.md) が定量化している。
3. **新しいレイヤーは必ず `test_gradcheck` で検証する。** 有限差分の検証を通らない
   `backward()` は正しくない。
4. **コミット前に整形する:**
   `clang-format -i $(git ls-files '*.cpp' '*.hpp')`。CI は未整形のコードで失敗する。
5. **警告はエラーである**（`-Wall -Wextra -Wpedantic -Wshadow -Wconversion
   -Wsign-conversion -Werror`、MSVC では `/W4 /WX`）。明示的な
   `static_cast<std::size_t>` や `static_cast<double>` を書くことになる。既存の
   コードもそうしており、だからこそ 4 つのツールチェーンで警告ゼロを保てている。
