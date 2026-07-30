# 実験ガイド：最適化手法 / 活性化関数 / ネットワーク構造

nnscratch が実装する3つの比較実験の概念・コードの動き、および
オリジナル実装（nnscratch C++）・PyTorch・TensorFlow の関係を説明する。

---

## 実験の設計方針

各実験は「**比べる対象以外の条件を完全に揃える**」ことで、1 要素の差だけを公平に測定する。

- 同一データ（8×8 手書き数字、`data/digits.csv`）
- 同一初期重み（`Rng` を固定シードで巻き戻してから各モデルを構築）
- 同一ミニバッチ順序（バッチシードを別途固定）

---

## 実験 1：最適化手法（SGD vs Momentum vs Adam）

### 概念

重みを「どの方向にどれだけ動かすか」のルールだけを変える。
ネットワーク構造・活性化関数・初期重みはすべて同一。

| 手法 | 仕組み | 特性 |
|---|---|---|
| **SGD** | 勾配の逆方向に固定歩幅で進む | 最も単純。収束が遅いことがある |
| **Momentum** | 過去の進行方向（速度）を引き継ぐ | 谷を転がるように加速。SGD より速く収束しやすい |
| **Adam** | 各パラメータの学習率を 1 次・2 次モーメントで自動調整 | 現代の標準。チューニングが少なくて済む |

```
SGD      : 毎回ゼロから「今の傾き」だけ見て進む → ジグザグしやすい
Momentum : 「今の傾き」＋「これまでの勢い」を合わせて進む → 直進しやすい
Adam     : さらに「急な傾きには小さく、緩やかな傾きには大きく」と歩幅を自動調整
```

### アルゴリズムの親子関係

3 つは独立したものではなく、拡張関係にある。

```
SGD
 └─ + 速度 v ──────────────────────────→ Momentum
                    └─ + v を m（1次モーメント）に置き換え、
                         さらに v（2次モーメント）を追加 → Adam
```

Adam は「Momentum の速度を 2 本に増やし、勾配の大きさでも正規化する」ものと理解できる。

### 数式

**SGD**

$$p \leftarrow p - \eta \cdot g$$

$p$：パラメータ（重み）、 $g$：勾配、 $\eta$：学習率

---

**Momentum**

$$v \leftarrow \mu v - \eta \cdot g$$

$$p \leftarrow p + v$$

$v$：速度（前ステップの進行方向を引き継ぐ）、 $\mu$：慣性係数（通常 $0.9$）

---

**Adam**

$$m \leftarrow \beta_1 m + (1 - \beta_1) g$$

$$v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$

$$\hat{m} = \frac{m}{1 - \beta_1^t}, \qquad \hat{v} = \frac{v}{1 - \beta_2^t}$$

$$p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$$

$t$：ステップ数、 $\beta_1 = 0.9$、 $\beta_2 = 0.999$、 $\varepsilon = 10^{-8}$

$\hat{m}$, $\hat{v}$ はバイアス補正項（初期ステップでモーメントが 0 に引っ張られることを補う）。

### コード（nnscratch C++）

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

// Adam: m, v の 2 モーメントを管理してパラメータを更新
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

## 実験 2：活性化関数（ReLU vs Tanh vs Sigmoid）

### 概念

活性化関数は各層の出力に「非線形性」を与える。
これがないと何層重ねても線形変換と同じになり、曲がった決定境界が必要な画像認識はできない。
実験ではネットワーク構造・SGD・初期重みを固定し、活性化関数だけを差し替える。

| 関数 | 数式 | 出力範囲 | 特性 |
|---|---|---|---|
| **ReLU** | $\max(0,\, x)$ | $[0,\, +\infty)$ | 勾配消失が起きにくい。現代の主流 |
| **Tanh** | $\dfrac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1,\, +1)$ | 0 を中心とした出力。Sigmoid より勾配が大きい |
| **Sigmoid** | $\dfrac{1}{1 + e^{-x}}$ | $(0,\, 1)$ | 深いネットでは勾配消失が起きやすい |

**勾配消失**：Sigmoid の最大勾配は $\frac{1}{4}$ しかなく、層を遡るたびに誤差信号が減衰する。
ReLU は正の領域で勾配が $1$ のまま伝わるため消失しにくい。

### 逆伝播の微分

各関数の $\text{backward}$ では、上流から来た勾配 $g$ に対して以下を掛ける。

```math
\text{ReLU}': \quad \frac{\partial}{\partial x}\max(0,x) = \begin{cases} 1 & (x > 0) \\ 0 & (x \le 0) \end{cases}
```

$$\text{Tanh}': \quad \frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

$$\text{Sigmoid}': \quad \frac{d}{dx}\sigma(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)$$

### コード（nnscratch C++）

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

## 実験 3：ネットワーク構造（浅い線形モデル vs 深い MLP vs CNN）

### 概念

最適化手法を Adam に固定し、ネットワークの「形」だけを変える。

| 構造 | 層の構成 | 何ができるか |
|---|---|---|
| **浅い（隠れ層なし）** | $64 \to 10$ | 線形な境界のみ。ロジスティック回帰と等価 |
| **深い MLP** | $64 \to 64 \to 32 \to 10$ | 非線形変換により複雑なパターンを学習可能 |
| **CNN** | $\text{Conv}(1 \to 8,\; 3\times3) \to \text{Flatten} \to \text{Dense}(10)$ | 画像の局所的な模様（エッジ・曲線）を捉える |

**MLP と CNN の違い**：

```
MLP : 64 ピクセル全部を一度に見て学習 → 位置がずれると別の特徴として扱う
CNN : 3×3 の小窓をスライドさせて局所パターンを探す → 位置がずれても同じ特徴を検出
```

### コード（nnscratch C++）

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

## nnscratch / PyTorch / TensorFlow の関係

### 3 実装の立ち位置

PyTorch・TensorFlow は nnscratch と**まったく同じ数式を動かしている**。
違いは「その計算を誰が書いたか（自動か手書きか）」と「どこまで機能が付随するか」。

```
nnscratch        PyTorch          TensorFlow（高水準）
──────────────   ──────────────   ──────────────────────
backward() 手書  loss.backward()  model.fit() 一行
optimizer 手書き torch.optim.*    model.compile() + fit
学習ループ手書き 学習ループ手書き 学習ループ自動
CPU のみ         GPU: .to(device) GPU: 自動検出
double (64bit)   float32          float32
```

### 差の 4 層

#### ① 自動微分の有無（最大の差）

```
nnscratch
  各層の backward() を連鎖律から人間が手書きする。

PyTorch
  loss.backward() 一行で計算グラフを自動で遡る（autograd）。

TensorFlow
  tf.GradientTape がグラフを記録して自動微分。
  model.fit() はさらにループごと隠蔽。
```

オプティマイザ自体の数式は同一でも、**その入力（勾配 $g$）を誰が用意するかが根本的に異なる**。

#### ② 数値型

| 実装 | 型 | 備考 |
|---|---|---|
| nnscratch | `double`（64bit） | 精度優先・教材向き |
| PyTorch | `float32`（デフォルト） | GPU で高速 |
| TensorFlow | `float32`（デフォルト） | 同上 |

同じ学習率でも精度差で収束曲線が微妙にずれることがある。

#### ③ Momentum の内部式の微妙な差

数学的には等価だが、内部の速度テンソル $v$ の持ち方が異なる。

$$\text{nnscratch / TF Keras}: \quad v \leftarrow \mu v - \eta g, \quad p \leftarrow p + v$$

$$\text{PyTorch SGD}: \quad v \leftarrow \mu v + g, \quad p \leftarrow p - \eta v$$

$v$ のスケールが $\eta$ 倍ずれているだけで結果は同じ。
PyTorch には `dampening` オプションがあり初回ステップの挙動を変えられる（nnscratch にはない）。

#### ④ 付随機能の有無

| 機能 | nnscratch | PyTorch | TensorFlow |
|---|---|---|---|
| Weight decay（L2 正則化） | なし | `weight_decay=` | `decay=` |
| 学習率スケジューラ | なし | `lr_scheduler.*` | `LearningRateSchedule` |
| パラメータグループ別 lr | なし | `param_groups` 対応 | なし |
| Adam AMSGrad 変種 | なし | `amsgrad=True` | なし |
| 勾配クリッピング | なし | `clip_grad_norm_()` | `clipnorm=` |

### SGD / Momentum / Adam が関係する差の層

| 差の層 | SGD | Momentum | Adam |
|---|---|---|---|
| ① 自動微分の有無 | ○ | ○ | ○ |
| ② float64 vs float32 | ○ | ○ | ○ |
| ③ 内部式の微妙な差 | **なし** | ○（式の形が異なる） | △（$`\varepsilon`$ の適用位置が異なる） |
| ④ 付随機能 | weight_decay / clipping | 同左 | + AMSGrad |

**SGD に③の差がない理由**：状態（速度・モーメント）を持たないため実装の揺れようがない。

**Adam の③について**：骨格とデフォルト値（$`\beta_1=0.9,\;\beta_2=0.999,\;\varepsilon=10^{-8}`$）は揃っているが、 $\varepsilon$ の適用位置が異なる。

$$\text{nnscratch / 論文 / TF}: \quad p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$$

$$\text{PyTorch（デフォルト）}: \quad p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v} + \varepsilon}}$$

$\hat{v}$ が極端に小さい場合に挙動が変わるが、通常の学習では差はほぼ出ない。

---

## 実行方法

### ビルド

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### デモアプリの実行

```bash
# 実験1〜3（最適化手法 / 活性化関数 / 構造の比較）
./build/compare

# 未学習→学習済みの全過程
./build/from_scratch
```

出力は `output/` ディレクトリに CSV と PGM 形式で書き出される。

### テスト

```bash
ctest --test-dir build --output-on-failure
```

| テストファイル | 内容 |
|---|---|
| `tests/test_tensor.cpp` | Tensor の演算・reshape・matmul |
| `tests/test_optimizer.cpp` | SGD / Momentum / Adam の更新値検証 |
| `tests/test_gradcheck.cpp` | 全層の analytic 勾配 vs 数値微分の一致確認 |
