# デモの実行と、その読み方

このプロジェクトを動かすすべての方法と、たいてい説明が抜けている「出力の何に
注目すればよいか」をまとめる。

コマンドはすべてリポジトリのルートから実行する。先にビルドすること。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

| コマンド | 得られるもの | 所要時間 |
|---|---|---|
| [`./build/from_scratch`](#1-from_scratch--1-つのネットワークが学習する) | 1 つのネットワークがランダムから学習済みになる過程を端末で | 約 1 秒 |
| [`./build/compare`](#2-compare--3-つの制御された実験) | 3 つの制御された実験の結果表 | 約 6 秒 |
| [`output/*.html` を開く](#3-html-レポート) | 同じ実行をブラウザで対話的に | 即時 |
| [`python reference/gui_demo.py`](#4-gui-デモ--学習を眺める) | 目の前で学習するウィンドウ | 1 回約 20 秒 |
| [`.\run_demo.ps1`](#5-run_demops1--1-コマンドで全部) | ビルド + テスト + 両デモをプログレスバーと ASCII チャート付きで | 約 15 秒 |
| [`./build/export_reference`](#6-参照実装との比較) | numpy/PyTorch/TensorFlow 比較用のデータ | 約 10 秒 |

Windows では実行ファイルは `build/Release/` にある
（`.\build\Release\from_scratch.exe`）。

---

## 1. `from_scratch` — 1 つのネットワークが学習する

```bash
./build/from_scratch                        # 同梱データ、出力は .
./build/from_scratch data/digits.csv output # パスを明示
```

ReLU を使う 64→64→32→10 の MLP を、素の SGD で 60 エポック学習させる。

### 出力の読み方

```
Untrained test accuracy: 5.3%  (≈10% = random for 10 classes)

epoch   0 | loss 2.5534 | train_acc   7.5% | test_acc   5.3%
epoch   1 | loss 0.3813 | train_acc  91.1% | test_acc  91.4%
epoch   2 | loss 0.6214 | train_acc  79.6% | test_acc  82.5%
epoch   3 | loss 0.1171 | train_acc  97.0% | test_acc  95.3%
...
epoch  60 | loss 0.0010 | train_acc 100.0% | test_acc  97.8%
```

**エポック 0 は未学習のネットワーク**で、更新前に記録される。60 エポックの実行で
履歴が 61 行になるのはそのためである。損失 2.55 は 10 クラスをほぼ一様に当てずっぽう
した場合の値で、$-\log(1/10) = 2.30$ に、一様よりやや外している分が乗っている。

**エポック 1 での飛躍は本物であり、バグではない。** 学習画像 1437 枚をバッチ 32 で
回せば 45 回の更新になり、しかもこの問題は易しい。学習の大半が最初の 1 周で実際に
起きる。

**エポック 2 の落ち込みは理解しておく価値がある。** 損失が *上がり*（0.38 → 0.62）、
正解率が 11 ポイント落ちてから回復する。学習率 0.3 の素の SGD が行き過ぎているので
ある。ステップが大きく、方向によっては最小値を飛び越える。これは欠陥ではなく、
「記憶を持たず歩幅も固定の最適化手法」がどう見えるかそのものであり、まさに
[実験 1](#2-compare--3-つの制御された実験) で Momentum と Adam が均してみせる現象で
ある。60 エポックの間に何度か起きるのが普通である。

**train が 100% に達しても test は 97.8% あたりで落ち着く。** この 2% の差は学習
データの暗記である。重要なのは差が広がり続けないことで、test の正解率が下がりながら
差が開き続けるようなら、それは対処すべき過学習である。

**手元の数値は少しずれる。** 未学習の値はおよそ 5〜9%、最終値は 97〜98% の幅で
コンパイラによって変わる。`std::normal_distribution` は同じシードから同じ値を返すと
規定されていないためである（[ARCHITECTURE.md](ARCHITECTURE.md#決定性)）。ただし
*大きな* 逸脱 — 最終正解率 60%、回復しないまま増え続ける損失 — は本物の問題なので
報告してほしい。

### 出力されるファイル

| ファイル | 内容 |
|---|---|
| `learning_curve.csv` | 上の表を機械可読にしたもの |
| `learned_features.pgm` | 第 1 層 64 ユニットを 8×8 画像として |
| `from_scratch.html` | 以上すべてを対話的に。[後述](#3-html-レポート) |

---

## 2. `compare` — 3 つの制御された実験

```bash
./build/compare data/digits.csv output
```

学習を 9 回行う。各実験は **1 つの** 軸だけを変え、データ・初期重み・バッチ順序を
固定するので、差が出ればそれは調べている対象によるものであって運ではない。概念の
説明は [experiments.md](experiments.md) にある。

### 表の読み方

```
=== Experiment 1: optimizers (SGD vs Momentum vs Adam) ===
name                  final test acc  best test acc     epochs to 90%
---------------------------------------------------------------------
Adam                          97.8%         97.8%                 1
Momentum                      97.5%         98.1%                 2
SGD                           97.2%         97.5%                 3
```

**話を担っているのは `epochs to 90%` の列である。** 最終正解率では 3 つはほとんど
分離しない。このデータセットは易しく、どれも最後にはたどり着くからである。*どれだけ
速く* たどり着くかが実際の差であり、それこそが最適化手法の主題である。

**`best` が `final` より高いのは、ピークを打って戻ったということ。** Momentum が
最良 98.1%、最終 97.5% になるのは、テスト画像 360 枚における学習後半の揺らぎとして
正常である。0.6 ポイントは 2 枚分にすぎない。

実験 2 はこのプロジェクトで最も明快な結果である。

```
ReLU                          97.2%         97.5%                 3
Sigmoid                       96.7%         96.7%                 9
Tanh                          97.2%         97.5%                 1
```

**Sigmoid は 90% 到達に 3 倍のエポックを要する。** その導関数は最大でも 1/4 なので、
通過する層ごとに逆伝播の信号が最低 4 分の 1 に縮む。勾配消失問題が 9 秒のデモで
目に見える。ReLU の導関数は活性側でちょうど 1 で、この種の係数を生まない
（[MATH.md](MATH.md#sigmoid-が遅れる理由)）。

実験 3 は結論ではなく留保が必要である。

```
1_shallow                     96.4%         96.7%                 2
2_deep_mlp                    97.8%         97.8%                 1
3_cnn                         97.2%         97.5%                 1
```

**深さが線形モデルに勝つことは確かだが、CNN と MLP の順序づけはそうではない。**
10 シードで見ると深い MLP と CNN の差は平均 0.31 ポイントで、これは各々のシード間の
ばらつきより小さく、テスト画像およそ 1 枚分である。1 回の実行から勝者を読み取っては
いけない。[experiments.md](experiments.md#実際に何が起きるか) を参照し、
`python reference/architecture_trials.py --seeds 10` で自分で確かめられる。

### 出力されるファイル

`cmp_optimizers.csv`、`cmp_activations.csv`、`cmp_architecture.csv`（long 形式、
1 行が 1 実行 1 エポック）、`cnn_filters.pgm`、`compare.html`。

---

## 3. HTML レポート

どちらのデモも自己完結したページを書き出す。そのまま開けばよく、サーバも
インストールも要らない。CSS・JavaScript・全データ点がインライン展開されているので、
USB メモリからでもメールの添付からでも動く。

```powershell
start output\from_scratch.html      # Windows
open  output/from_scratch.html      # macOS
xdg-open output/from_scratch.html   # Linux
```

### 何ができるか

- **チャートをホバー**するとクロスヘアとそのエポックの値が出る。「正確にはいつ 90% を
  超えたか」を調べるのが最も速い。
- **凡例をクリック**すると系列を隠して下にあるものを見られる。色は実行に紐づくので、
  1 つ隠しても他の色は変わらない。
- **Show data table** でチャートの背後の数値を表示する。
- 損失チャートの **Linear scale / Log scale**。対数から始めること。線形軸ではエポック 5
  以降の曲線がすべてゼロに張り付いて何も見えない。対数なら着実な進捗は直線になり、
  *停滞*（対数で平らな区間）が探すべきものになる。
- **Theme** で明暗を切り替える。OS の設定にも追従する。

重み画像は **タイルごとに** min–max 正規化されている。明るさは 1 つのタイルの内部では
比較できるが、タイル *どうし* では比較できない。構造を読み取るために使い、大きさを
読み取ってはいけない。

---

## 4. GUI デモ — 学習を眺める

```bash
python -m pip install -r reference/requirements.txt   # 初回のみ
python reference/gui_demo.py                          # 起動して Train を押す
python reference/gui_demo.py --autostart              # 起動と同時に学習開始
```

4 つのパネルがリアルタイムに動くウィンドウ。ドキュメントが説明していることを実際に
*見たい* ならこれを使う。

| パネル | 注目点 |
|---|---|
| Training loss（対数） | 最初の数エポックの急降下、その後の長い直線的な低下。凹凸は SGD の行き過ぎ |
| Accuracy | train と test が並走しているか。暗記が始まると離れる |
| First-layer weights | 面白いのはここ。エポック 0 ではノイズ、数エポックでストローク状・エッジ状の構造に解けていく |
| Held-out digits | テスト画像 10 枚と現在の予測。誤答は `予測→正解` を赤の太字で表示 |

プルダウンは同じシードから作り直すので、切り替えは再シャッフルではなく制御された
比較になる。試す価値があるのは次の 3 つ。

1. 深い MLP での **Sigmoid vs ReLU**。ReLU が落ちるところで損失曲線が目に見えて
   這う。勾配消失をその場で観察できる。
2. **CNN**。重みパネルが 3×3 のカーネル 8 枚に切り替わり、明暗の対比 — エッジ検出器 —
   がノイズから現れるのを見られる。
3. **Shallow（64→10）**。ほぼ即座に収束し、そこで改善が止まる。線形モデルの限界が
   5 秒ほどで分かる。

ネットワークは `reference/` の numpy 実装であり、C++ ライブラリと 2e-15 で一致する
ので、見ているのは nnscratch の計算そのものである。学習は UI のイベントループ内で
1 ティック 1 エポックずつ進む。MLP の 60 エポックで約 20 秒。

---

## 5. `run_demo.ps1` — 1 コマンドで全部

PowerShell 7 以降、Windows:

```powershell
.\run_demo.ps1                # configure、ビルド、ctest、そして両デモ
.\run_demo.ps1 -SkipBuild     # デモだけ
.\run_demo.ps1 > demo.log     # 非対話: プロンプトもアニメーションもなし
```

対話実行ではアニメーション付きのプログレスバー、ASCII の学習曲線、比較の棒グラフを
描き、パートの区切りで「Press Enter」で止まる。標準出力がコンソールでない場合は
自動的にアニメーションをやめ、1 エポック 1 行を出力するので、リダイレクトもパイプも
動作する（`-NonInteractive` を渡せば端末上でも同じ動作を強制できる）。ここで行うことは
すべて素の CMake コマンドで到達でき、ライブラリ側がこれに依存することはない。

---

## 6. 参照実装との比較

同じネットワークを numpy・PyTorch・TensorFlow で、*同じ* 初期重みとバッチ順序から
学習させ、手書きの逆伝播と `loss.backward()` が同じ計算であることを示す。詳細・
結果・留保は [reference/README.md](../reference/README.md) にある。

```bash
./build/export_reference                       # 分割・重み・バッチ順序
python reference/numpy_reference.py            # 手書きの backward
python reference/pytorch_reference.py          # autograd
python reference/tensorflow_reference.py       # GradientTape
python reference/compare_curves.py             # 重ね合わせと差分表
```

差分表の読み方: **2e-15 は丸め誤差**であり「同一」を意味する。**2e-08 も同一である。**
これはこのライブラリが `log(p + 1e-9)` を報告し、フレームワークが `log p` を報告する
ことによるオフセットで、エポック 0 の時点で既に存在し、勾配は通らない。それを超える
ものは本物の差であり、該当するのはちょうど 1 つ、TensorFlow の CNN である。Keras が
Adam の $\varepsilon$ をバイアス補正の前に適用するためである。

単独で実行できる検証が 2 つある。

```bash
python reference/check_optimizer_equivalence.py   # 各フレームワークの更新式を同定
python reference/architecture_trials.py --seeds 10 # 実験 3 の順位は再現するか
```

---

## おかしいと思ったとき

| 症状 | 意味 |
|---|---|
| 最終正解率がここの数値と 0.数 % 違う | コンパイラをまたげば想定内。分布は移植性を持たない（[ARCHITECTURE.md](ARCHITECTURE.md#決定性)） |
| 損失が 1〜2 エポック上がってから回復する | 学習率 0.3 の素の SGD では正常。上がり続けるなら異常 |
| 未学習の正解率が 10% ではなく 5% | これも問題ない。未学習の argmax は一様ではなく任意なので、偶然の水準を下回りうる |
| `load_digits: cannot open ...` | リポジトリのルートから実行するか、CSV のパスを明示する |
| 正解率が 10% 付近から何エポックも動かない | 本物の問題。まず `ctest` を実行すること。勾配チェックが壊れた `backward()` を捕まえる（[TESTING.md](TESTING.md)） |
| HTML ページが真っ白 | ブラウザのコンソールを確認する。ネットワークは不要だが JavaScript は必要である |
