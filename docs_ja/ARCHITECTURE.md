# アーキテクチャ

各部品がどう組み合わさっているか。何が何に依存し、学習の 1 ステップで何が起き、
どのメモリを誰が所有し、どの不変条件を壊してはならないのか。

*なぜ* この形なのかは [DESIGN.md](DESIGN.md)、各レイヤーの数式は
[MATH.md](MATH.md)、シグネチャは [API.md](API.md) を参照。

- [モジュール構成](#モジュール構成)
- [レイヤーの抽象](#レイヤーの抽象)
- [学習 1 ステップの流れ](#学習-1-ステップの流れ)
- [形状の契約](#形状の契約)
- [所有権と生存期間](#所有権と生存期間)
- [不変条件](#不変条件)
- [エラー処理](#エラー処理)
- [決定性](#決定性)
- [意図的に存在しないもの](#意図的に存在しないもの)

---

## モジュール構成

ヘッダ 10 本、翻訳単位 9 本、循環依存なし。すべては `Tensor` に行き着く。

```
                        tensor.hpp
             （唯一の真に基礎的な型）
                            |
        +-------------------+--------------------+---------+
        |                   |                    |         |
     rng.hpp            layer.hpp             loss.hpp   pgm.hpp
   （シード付き PRNG）(Layer, ParamGrad)    (softmax+CE)  （ヘッダのみ、
        |                   |                    |         デモ専用）
        |          +--------+--------+           |
        |          |                 |           |
        +----> layers.hpp     activations.hpp    |
        |    (Dense, Conv2D,   (ReLU, Tanh,      |
        |     Flatten)          Sigmoid,         |
        |          |            softmax())       |
        |          |                 |           |
        |          +--------+--------+           |
        |                   |                    |
        |               model.hpp                |
        |        （レイヤー列を所有）             |
        |                   |                    |
        |            optimizer.hpp               |
        |         (SGD/Momentum/Adam)            |
        |                   |                    |
        +-------------> training.hpp <-----------+
        |          （エポックのループ）
        |
    dataset.hpp
  (digits.csv ローダ)

  nnscratch.hpp  =  総括ヘッダ。pgm.hpp 以外のすべてを include する
```

| ヘッダ | 翻訳単位 | 役割 |
|---|---|---|
| `tensor.hpp` | `src/tensor.cpp` | フラットな行優先の記憶域、`matmul`、要素ごとの演算、reshape/transpose |
| `rng.hpp` | ヘッダのみ | `std::mt19937_64` のラッパー。正規乱数、置換、再シード |
| `layer.hpp` | ヘッダのみ | `Layer` インターフェースと `ParamGrad` |
| `layers.hpp` | `src/layers.cpp`, `src/conv2d.cpp` | `Dense`, `Flatten`, `Conv2D`, `Init` |
| `activations.hpp` | `src/activations.cpp` | `ReLU`, `Tanh`, `Sigmoid`, 自由関数 `softmax()` |
| `loss.hpp` | `src/loss.cpp` | `SoftmaxCrossEntropy`, `one_hot()` |
| `model.hpp` | `src/model.cpp` | 逐次コンテナ、`predict`、`accuracy` |
| `optimizer.hpp` | `src/optimizer.cpp` | `Optimizer` 基底クラスと 3 つの実装 |
| `training.hpp` | `src/training.cpp` | `train()`, `TrainConfig`, `History` |
| `dataset.hpp` | `src/dataset.cpp` | `load_digits()`, `DigitsData` |
| `pgm.hpp` | ヘッダのみ | グレースケール PGM の書き出し。デモ専用 |

`Conv2D` は `layers.hpp` で宣言されているのに独立した `.cpp` を持つ。実装が
群を抜いて長く、分離しておくことで `layers.cpp` が一度に読み切れる長さに収まる
からである。

`pgm.hpp` は意図的に総括ヘッダに含めていない。これはデモ用のユーティリティで
あってニューラルネットワーク API の一部ではないので、デモ側が明示的に include する。

---

## レイヤーの抽象

forward/backward の流れに参加するものはすべて、3 つのメソッドを実装する
（`include/nnscratch/layer.hpp`）。

```cpp
virtual Tensor forward(const Tensor& x) = 0;
virtual Tensor backward(const Tensor& grad_out) = 0;
virtual std::vector<ParamGrad> params_and_grads() { return {}; }
```

契約は次のとおり。

1. **`forward` は `backward` が必要とするものをキャッシュする。** テープもグラフも
   存在せず、キャッシュ *こそが* グラフである。`Dense` は入力を、`ReLU` は 0/1
   マスクを、`Tanh`/`Sigmoid` は出力を、`Conv2D` は im2col 行列と入力形状を、
   `Flatten` は入力形状をキャッシュする。
2. **`backward` は $\partial L/\partial\text{出力}$ を受け取り
   $\partial L/\partial\text{入力}$ を返す。** その過程でパラメータの勾配を自身の
   メンバに書き込む。
3. **`params_and_grads` はオプティマイザに安定したポインタを渡す。** 学習可能な
   状態を持たないレイヤーは、既定の空実装をそのまま継承する。

`SoftmaxCrossEntropy` は意図的に `Layer` を実装 *しない*。教師データを受け取って
スカラーを返すので、シグネチャの形が違うからである
（`forward(logits, targets) -> double` と引数なしの `backward()`）。`Layer` に
してしまうと教師データを裏口から渡すことになる。分離しておくほうが、学習ループの
構造が素直に見える。

---

## 学習 1 ステップの流れ

`train()` が 1 つのミニバッチに対して行うこと（`src/training.cpp`）。ネットワークを
`Dense → ReLU → Dense` とする。

```
 1. gather_rows(x_train, batch_indices)          -> xb  (B, 64)
    gather_rows(Y_train, batch_indices)          -> yb  (B, 10)
         （Y_train はエポックループの前に一度だけ one-hot 化されている）

 2. model.forward(xb)
       Dense[0].forward   xb -> h1     xb をキャッシュ
       ReLU[1].forward    h1 -> a1     mask(h1 > 0) をキャッシュ
       Dense[2].forward   a1 -> logits a1 をキャッシュ
                                                 -> logits (B, 10)

 3. loss_fn.forward(logits, yb)
       softmax(logits) -> probs、yb とともにキャッシュ
       平均 cross-entropy を返す               （ここでは値を捨てる。
                                                 エポックの指標は全データで
                                                 計算し直す）

 4. loss_fn.backward()  ->  (probs - yb) / B     -> g (B, 10)

 5. model.backward(g)      ［レイヤーを逆順にたどる］
       Dense[2].backward  g  -> g2    dW_, db_ を書き込む
       ReLU[1].backward   g2 -> g3    (g2 * mask)
       Dense[0].backward  g3 -> g4    dW_, db_ を書き込む
                                                 (g4 は捨てられる。入力に
                                                  対する勾配は誰も必要としない)

 6. opt.step(model.params_and_grads())
       [{&W0,&dW0}, {&b0,&db0}, {&W2,&dW2}, {&b2,&db2}] に平坦化し、
       すべてのパラメータをその場で更新する
```

そしてエポックごとに、全ミニバッチを処理し終えた後で:

```
 7. model.forward(x_train)  学習データ全体に対して -> 指標
    loss_fn.forward(...)  -> train_loss
    model.accuracy(x_test, y_test) -> test_acc
    History に積む
```

手順 7 について、見落としやすい点が 3 つある。

- **すべてのレイヤーの forward キャッシュを全データの活性値で上書きする。** この後
  `backward()` が続かないから安全なだけである。指標計算の後に `backward()` を呼ぶ
  コールバックを足すと、まったく別のものを微分することになる。
- 無料ではない。エポックごとに全データの forward が 2 回余分に走る。このデータ
  セットでは小さな定数だが、大きなデータセットではサンプリングすることになる。
- **エポック 0 は更新前に実行される。** これが `History` に「未学習 → 学習済み」の
  出発点を与えている。したがって `cfg.epochs = 60` なら 61 行になる。

勾配をゼロクリアする呼び出しはどこにもない。勾配は蓄積ではなく *代入* される
（`dW_ = matmul(...)` が上書きする）からである。PyTorch の `optimizer.zero_grad()`
に相当するものが存在しないのはこのためであり、同時に、`backward()` を変更せずに
マイクロバッチ間の勾配蓄積を表現できない理由でもある。

---

## 形状の契約

| レイヤー | 入力 | 出力 | 備考 |
|---|---|---|---|
| `Dense(n_in, n_out)` | $(N, n_{\mathrm{in}})$ | $(N, n_{\mathrm{out}})$ | ランク 2 のみ |
| `ReLU` / `Tanh` / `Sigmoid` | 任意 | 同じ形状 | 要素ごと。ランク非依存 |
| `Flatten` | $(N, d_1, \ldots, d_k)$ | $(N, \prod d_i)$ | backward でランクを復元 |
| `Conv2D(C_in, C_out, K, S, P)` | $(N, C_{\mathrm{in}}, H, W)$ | $(N, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})$ | ランク 4 必須。[MATH.md](MATH.md#出力サイズ) 参照 |
| `SoftmaxCrossEntropy` | ロジット $(N, K)$、教師 $(N, K)$ | スカラー。backward は $(N, K)$ | 教師は one-hot |

したがって CNN では `Conv2D` と `Dense` の間に `Flatten` が必要で、入力には
`data.train.flat` ではなく `data.train.img` を与える。`load_digits` が同じサンプルの
2 つのビューを返すのは、まさにこの選択にコストをかけないためである
（`dataset.hpp`）。

形状の誤りは黙って誤動作するのではなく `Tensor` からの例外として現れる。
[エラー処理](#エラー処理)を参照。

---

## 所有権と生存期間

実際に鋭い角があるのはここで、すべては `ParamGrad` が生ポインタを保持することに
由来する。

```
Model                            所有: std::vector<std::unique_ptr<Layer>>
  └── Dense                      所有: W_, b_, dW_, db_, x_   （値としての Tensor）
        └── Tensor               所有: std::vector<double>    （ヒープ上のバッファ）

ParamGrad {Tensor* param, Tensor* grad}
  └── レイヤーのメンバを指す、所有しないビュー

Optimizer (Momentum/Adam)
  └── std::unordered_map<const Tensor*, Tensor>
        └── パラメータ Tensor の「アドレス」をキーとする
```

**アドレスが安定である理由。** レイヤーは `unique_ptr` 経由で保持されるので、
`add()` を重ねて `std::vector` が再確保されても `Layer` オブジェクト自体は固定した
ヒープアドレスに留まる。メンバの `Tensor` もそれとともに動くので、`&W_` はレイヤー
の生存期間を通じて安定である。アドレスをキーにしたオプティマイザの状態が健全で
あるのはこのためである。（`std::vector<Dense>` ならこうはいかない。再確保が
オプティマイザの保持するポインタをすべて無効化してしまう。）

**そこから導かれる規則:**

| 規則 | 理由 |
|---|---|
| `Model` は、そこから取り出した `ParamGrad` と、それを更新したオプティマイザより長く生存しなければならない。 | さもないとポインタがぶら下がる。 |
| `Momentum`/`Adam` のインスタンスを、作り直したモデルに使い回さない。 | 状態はアドレスでキーづけされている。新しいモデルが解放済みのアドレスに載ると、古いモーメントを引き継いでしまう。クラッシュはせず、ただ静かに誤った学習になる。`compare.cpp` は実行ごとに新しいオプティマイザを構築している。 |
| `params_and_grads()` は安価だが無料ではない。呼ぶたびに新しいベクタを構築する。 | `train()` はミニバッチごとに 1 回呼ぶ。この規模なら問題ない。プロファイルして気になるなら外へ括り出せばよい。 |
| `Model::add<L>()` / `Model::layer(i)` が返す参照は、モデルの生存期間中は有効である。 | 同じく `unique_ptr` の間接参照のおかげである。`from_scratch.cpp` は `add()` が返した `Dense&` を数百エポックにわたって保持し、最後に重みを読み出している。 |

`Model` はムーブ可能だが **コピー不可** である（`Layer` に `clone()` がない）。
`compare.cpp` はムーブ代入（`m = build_mlp(...)`）でモデルをその場で作り直している。

---

## 不変条件

いずれも、破ってもコンパイルは通り、エラーではなく誤った数値が出る。だからこそ
明示しておく価値がある。

1. **同じデータに対して `forward()` を呼んでから `backward()` を呼ぶ。** すべての
   `backward()` は対応する `forward()` が書いたキャッシュを読む。先に `backward()`
   を呼ぶと、既定構築された空のキャッシュを読むことになる。
2. **`forward()` 1 回につき `backward()` は 1 回。** 勾配は蓄積ではなく代入される。
   2 回呼んでも加算されず、2 回目が上書きする。
3. **`SoftmaxCrossEntropy::forward` を `backward()` より先に呼ぶ。** 同じ理由で、
   `backward()` はキャッシュされた確率と教師を読む。
4. **バッチサイズは呼び出しごとに変わってよい。** $N$ を跨いで保持するものはなく、
   エポック末尾の半端なミニバッチが毎回このことを実証している。
5. **教師は行ごとに正規化された分布であること。** 融合された勾配の導出は
   $\sum_k y_{ik} = 1$ を使う。正規化されていない教師を与えると、最小化している
   損失が静かに変わる（[MATH.md](MATH.md#collapse)）。
6. **ラベルの値は $[0, K)$ に収まること。** `one_hot()` はラベルをそのまま添字に使い、
   `Tensor::operator()` は境界検査をしない。

---

## エラー処理

例外を使うが、対象はプログラマのミスと I/O に限られる。

| 例外 | 送出元 | 原因 |
|---|---|---|
| `std::invalid_argument` | `matmul`, `operator+/-/*`, `reshape`, `Tensor::from`, `add_row_vector`, `axpy` | 形状またはサイズの不一致 |
| `std::logic_error` | `rows`, `cols`, `transpose`, `sum_rows`, `add_row_vector`, `matmul`, `Conv2D::forward` | 演算に対してランクが誤っている |
| `std::runtime_error` | `load_digits` | ファイルがない、行が壊れている、データが空 |

内側のループの速度と単純さのために、意図的に検査 *しない* もの:

- `Tensor::operator()(i, j)` — 境界検査なし（`data_[i * cols + j]` を直接引く）。
  `dim(i)` と `shape().at(i)` は検査 *される* が、要素アクセスはされない。
- `Conv2D` の形状 — パディング後の入力より大きいカーネルを与えると
  `H + 2P - K` が符号なしで回り込み、異常な `oh_` になって、明確なメッセージでは
  なく確保の失敗として現れる。
- `one_hot()` のラベル範囲。

それ以外も同様で、標準ライブラリがやること以上のメモリ確保失敗の処理はなく、
エラーコードもどこにも存在しない。

---

## 決定性

独立した 3 つのシードを意図的に分離している。

| シード | 設定場所 | 制御対象 |
|---|---|---|
| モデルのシード | アプリ側の `Rng rng(42)` | 初期重み |
| バッチのシード | `TrainConfig::batch_seed`（既定 123） | ミニバッチの並び順。`train()` 内のローカルな `Rng` が駆動する |
| 分割のシード | `load_digits(..., split_seed)`（既定 0） | どのサンプルが train / test に入るか |

これらを分離しておくことが、比較実験を公平にしている。モデル用の RNG を各実行の
直前に再シードすれば競合する設定が同一の初期重みから始まり、バッチ順序はそれとは
独立に固定されたままになる。こうして 2 つの実行は、調べたい軸 *だけ* で異なる。

**プラットフォーム間の注意。** `Rng::permutation` は `std::uniform_int_distribution`
を使って Fisher–Yates を直接実装しており、`Rng::normal` は
`std::normal_distribution` を使う。*エンジン*（`mt19937_64`）はビット単位まで標準で
規定されているが、*分布* はそうではない。libstdc++、libc++、MSVC で同じエンジン出力
が異なる値に写像されうる。したがって結果は特定のツールチェーンでは再現するが、
ツールチェーンをまたぐと数十分の 1 パーセント程度ぶれる。README が単一の数値では
なく範囲を示しているのはそのためである。

---

## 意図的に存在しないもの

何が *ない* かを知ることは、何があるかを知るのと同じくらい役に立つ。

| 存在しないもの | 帰結 | 理由 |
|---|---|---|
| 自動微分のテープ | 新しいレイヤーには手で導いた `backward()` が必要 | その導出こそがこのプロジェクトの目的そのものだから |
| BLAS / SIMD / スレッド | `matmul` は素朴な `ikj` の 3 重ループ | 可読性。[PERFORMANCE.md](PERFORMANCE.md) 参照 |
| `float` / 混合精度 | 全体を通じて 1 要素 8 バイト | 勾配チェックを鋭く保っているのが `double` である（[MATH.md](MATH.md#epsilon-の選び方)） |
| ビュー / ストライド / 遅延評価 | すべての演算が新しい `Tensor` を確保する | コストモデルが 1 つで明快、エイリアシングの問題も生じない |
| シリアライズ | 重みの保存・読み込みができない | 1 回の実行が数秒で、シードから再現できる |
| 正則化、Dropout、Batch Normalization、学習率スケジュール | 新しいレイヤーなしには表現できない | 移植元のリファレンスの範囲外 |
| 非逐次のトポロジー | `Model` は一直線の積み重ね。分岐もスキップ接続もない | DAG にはグラフ表現、すなわちテープが必要になる |

レイヤーやオプティマイザの追加は既存の構造にきれいに収まる
（[EXTENDING.md](EXTENDING.md) を参照）。分岐の追加はそうではない。
