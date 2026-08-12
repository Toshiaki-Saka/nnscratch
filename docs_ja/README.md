# nnscratch ドキュメント（日本語）

The English version lives in [`docs_en/`](../docs_en/README.md).

10 本のドキュメントがある。知りたいことがあるところから読めばよい。

| ドキュメント | 答えること |
|---|---|
| [DESIGN.md](DESIGN.md) | *なぜ* この作りなのか。自明でない設計判断 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | *どう* 組み合わさっているか。モジュール構成、学習 1 ステップの流れ、所有権、不変条件 |
| [MATH.md](MATH.md) | 各 `backward()` はどこから来たのか。すべての勾配を連鎖律から導出する |
| [API.md](API.md) | 何を呼べるのか。公開されている型と関数のすべて。形状、前提条件、計算量つき |
| [experiments.md](experiments.md) | デモは何を示しているのか。3 つの比較実験と、PyTorch/TensorFlow との対応 |
| [EXTENDING.md](EXTENDING.md) | レイヤー、オプティマイザ、損失関数、データセットをどう追加するか |
| [TESTING.md](TESTING.md) | 何がテストされ、何がされていないか。勾配チェックの失敗をどう読むか |
| [BUILD.md](BUILD.md) | ビルド、インストール、利用、トラブルシューティング |
| [DATA_FORMATS.md](DATA_FORMATS.md) | `digits.csv`、出力 CSV、PGM 画像の正確な中身 |
| [PERFORMANCE.md](PERFORMANCE.md) | 可読性の代償はいくらか。計算量、実測値、既知の非効率 |

---

## 読む順序の例

**誤差逆伝播が実際にどう動くのかを学ぶ** — プロジェクトの存在理由そのもの:

1. [MATH.md](MATH.md) — まず `Dense`、次に
   [softmax + cross-entropy の融合](MATH.md#softmax--cross-entropy-の融合)、
   そして [Conv2D](MATH.md#im2col-による-conv2d)
2. 各導出の隣で対応するソースを読む: `src/layers.cpp`、`src/loss.cpp`、
   `src/conv2d.cpp`
3. [TESTING.md の test_gradcheck](TESTING.md#test_gradcheck) — その導出が正しいことを
   どう証明しているか
4. [experiments.md](experiments.md) — 学習曲線に現れる帰結を見る

**ライブラリとして使う:**

1. [BUILD.md のライブラリを利用する](BUILD.md#ライブラリを利用する)
2. [API.md](API.md)
3. [ARCHITECTURE.md の所有権と生存期間](ARCHITECTURE.md#所有権と生存期間) —
   `ParamGrad` に伴うポインタの規則
4. 独自データを使うなら [DATA_FORMATS.md](DATA_FORMATS.md)

**コントリビュートする:**

1. [CONTRIBUTING.md](../CONTRIBUTING.md) — 基本ルール
2. [ARCHITECTURE.md](ARCHITECTURE.md) — レイヤー契約と不変条件
3. [EXTENDING.md](EXTENDING.md) — 実装例つきの手順
4. [TESTING.md](TESTING.md) — 勾配チェックがカバーするまで変更は完了しない

**自分の問題に合うか判断する:**

1. [PERFORMANCE.md のスケールの限界](PERFORMANCE.md#スケールの限界)
2. [ARCHITECTURE.md の意図的に存在しないもの](ARCHITECTURE.md#意図的に存在しないもの)

---

## 全体で使う記法

- $N$ はバッチサイズで、常に先頭軸である。ランク 2 テンソルは
  $(\text{rows}, \text{cols})$、ランク 4 は $(N, C, H, W)$ で、いずれも行優先。
- $G = \partial L/\partial Y$ は上流の勾配であり、すべての `backward()` の
  `grad_out` 引数にあたる。
- $1/N$ のバッチ平均は損失にのみ存在するので、どのレイヤーも $N$ で割らない。
- コードへの参照はリポジトリのルートからの相対パスで書く（`src/conv2d.cpp`、
  `include/nnscratch/tensor.hpp`）。
