# ビルドと統合

CMake 3.16 以上と C++20 コンパイラだけあればよい。このドキュメントはビルドオプション、
ライブラリを取り込む 3 通りの方法、インストール構成、そして CI の構成を扱う。

- [必要なもの](#必要なもの)
- [手早くビルドする](#手早くビルドする)
- [オプション](#オプション)
- [ターゲット](#ターゲット)
- [コンパイラフラグ](#コンパイラフラグ)
- [ライブラリを利用する](#ライブラリを利用する)
- [インストール構成](#インストール構成)
- [デモ](#デモ)
- [Windows: run_demo.ps1](#windows-run_demops1)
- [継続的インテグレーション](#継続的インテグレーション)
- [トラブルシューティング](#トラブルシューティング)

---

## 必要なもの

| | 最低要件 | 備考 |
|---|---|---|
| CMake | 3.16 | 3.21 以降は `PROJECT_IS_TOP_LEVEL` を標準で持つ。それ未満のためにプロジェクト側で補っている |
| GCC | 10 | |
| Clang | 12 | |
| MSVC | 19.29 (VS 2019 16.10) | |
| 依存関係 | *なし* | ビルド時も実行時も C++20 標準ライブラリのみ |

C++20 は `target_compile_features(nnscratch PUBLIC cxx_std_20)` で要求し、
`CMAKE_CXX_EXTENSIONS OFF` を設定しているので、利用側もこの要件を継承し、
`-std=gnu++20` ではなく標準の `-std=c++20` が使われる。

使っている C++20 の機能はごく一部である（range-for での構造化束縛、`[[nodiscard]]`
など）。古いツールチェーンで回避すべき特殊な機能は、標準バージョンの要件そのものを
除けば何もない。

---

## 手早くビルドする

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

単一構成ジェネレータで `CMAKE_BUILD_TYPE` を指定しなかった場合、`CMakeLists.txt` が
`Release` を強制する。最適化なしが既定だと、理由もなくデモが目に見えて遅くなる
からである。

マルチ構成ジェネレータ（Visual Studio, Xcode）では、構成はビルド時に選ぶ。

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

---

## オプション

| オプション | 既定値 | 効果 |
|---|---|---|
| `NNSCRATCH_BUILD_APPS` | トップレベルなら `ON`、そうでなければ `OFF` | `from_scratch` と `compare` をビルドする |
| `NNSCRATCH_BUILD_TESTS` | トップレベルなら `ON`、そうでなければ `OFF` | テスト 3 本をビルドし、CTest を有効にする |
| `NNSCRATCH_WARNINGS_AS_ERRORS` | `ON` | `-Werror` / `/WX` を付ける |

この既定値のおかげで、親プロジェクトから `add_subdirectory(nnscratch)` すると
ライブラリ *だけ* がビルドされ、デモの実行ファイルもテストターゲットもビルドを
汚さない。一方、単独でチェックアウトした場合はすべてビルドされる。これが意図した
挙動であり、設定は不要である。

新しいコンパイラが未知の診断を出すようになった場合は
`NNSCRATCH_WARNINGS_AS_ERRORS` を切るとよい。

```bash
cmake -S . -B build -DNNSCRATCH_WARNINGS_AS_ERRORS=OFF
```

---

## ターゲット

| ターゲット | 種類 | 備考 |
|---|---|---|
| `nnscratch` | ライブラリ | 本体。`BUILD_SHARED_LIBS=ON` でなければ `STATIC` |
| `nnscratch::nnscratch` | エイリアス | `target_link_libraries` ではこちらを使う。ツリー内ビルドでも `find_package` でも同じ名前で通る |
| `nnscratch_warnings` | `INTERFACE` | 警告フラグ一式。`PRIVATE` かつ `$<BUILD_INTERFACE:...>` で包んであるので、インストール後の利用側には漏れない |
| `from_scratch`, `compare`, `export_reference` | 実行ファイル | デモと参照データのエクスポータ。`NNSCRATCH_BUILD_APPS` に依存 |
| `test_tensor`, `test_gradcheck`, `test_optimizer` | 実行ファイル | `NNSCRATCH_BUILD_TESTS` に依存。CTest に登録される |

`VERSION` と `SOVERSION` は `project(VERSION 0.1.0)` から設定されるので、共有
ライブラリとしてビルドすれば適切にバージョン付けされた `libnnscratch.so.0` ができる。

---

## コンパイラフラグ

`nnscratch_warnings` インターフェースターゲット経由で適用される。

| ツールチェーン | フラグ |
|---|---|
| MSVC | `/W4 /utf-8`（+ `/WX`） |
| GCC / Clang | `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion`（+ `-Werror`） |

厳しいのは `-Wconversion` と `-Wsign-conversion` だが、これは意図的である。この
コードは `std::size_t` で添字計算をしながら `double` で計算するので、暗黙の縮小変換
によるバグがまさに生じやすい。新しいコードでは明示的な `static_cast` を書くことに
なる。既存のソースも一貫してそうしており、だからこそ GCC、Clang、AppleClang、MSVC の
4 つで警告ゼロを保てている。

`/utf-8` が必要なのは、デモが文字列リテラル中に非 ASCII 文字（`≈`、`→`）を出力する
ためである。これがないと MSVC がソースのエンコーディングを取り違える。

---

## ライブラリを利用する

### 1. インストール済みパッケージ

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build
cmake --install build
```

```cmake
find_package(nnscratch REQUIRED)
target_link_libraries(your_app PRIVATE nnscratch::nnscratch)
```

標準外のプレフィックスへインストールした場合は
`-DCMAKE_PREFIX_PATH=/your/prefix` で場所を教える。

パッケージのバージョンファイルは `SameMajorVersion` 互換性を使う。0.x では *マイナー*
の更新がすべて非互換になるが、これは 1.0 前のライブラリにおけるセマンティック
バージョニングの保守的な解釈である。`find_package(nnscratch 0.1 REQUIRED)` は 0.2 を
受け付けない。

### 2. `add_subdirectory`

```cmake
add_subdirectory(third_party/nnscratch)
target_link_libraries(your_app PRIVATE nnscratch::nnscratch)
```

この場合、デモとテストは自動的に無効になる。

### 3. FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(nnscratch
    GIT_REPOSITORY https://github.com/Toshiaki-Saka/nnscratch.git
    GIT_TAG        main)
FetchContent_MakeAvailable(nnscratch)
target_link_libraries(your_app PRIVATE nnscratch::nnscratch)
```

再現可能なビルドのためには `GIT_TAG` をリリースタグかコミット SHA に固定すること。

### CMake を使わない場合

設定するものは何もない。`include/` をインクルードパスに加え、`src/` の 9 ファイルを
コンパイルすればよい。

```bash
g++ -std=c++20 -O2 -Iinclude src/*.cpp your_app.cpp -o your_app
```

生成ヘッダも、configure ステップも、機能検出も存在しない。

---

## インストール構成

```
${CMAKE_INSTALL_PREFIX}/
├── include/nnscratch/*.hpp                   公開ヘッダすべて（pgm.hpp を含む）
├── lib/                                      libnnscratch.a （または .so / .lib）
└── lib/cmake/nnscratch/
    ├── nnscratchConfig.cmake                 cmake/nnscratchConfig.cmake.in から生成
    ├── nnscratchConfigVersion.cmake          SameMajorVersion
    └── nnscratchTargets.cmake                エクスポートされたターゲット。nnscratch:: 名前空間
```

パスは `GNUInstallDirs` に従うので、それを使うディストリビューションでは `lib` が
`lib64` になる。デモとデータセットは **インストールされない**。成果物ではなく実演
だからである。

---

## デモ

```bash
./build/from_scratch          # Part 1: 未学習 -> 学習済み
./build/compare               # Part 2: 3 つの制御された実験
```

どちらも省略可能な引数を取る。

```
from_scratch [digits.csv] [output_dir]
compare      [digits.csv] [output_dir]
```

既定の CSV パスは configure 時にコンパイル定義として埋め込まれる。

```cmake
target_compile_definitions(${demo} PRIVATE
    NNSCRATCH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
```

そのため作業ディレクトリによらず同梱データセットを見つけられる。ただしこれは
*ソースツリー* を指す絶対パスなので、別のマシンにバイナリだけコピーした場合は CSV
のパスを明示的に渡す必要がある。

出力（CSV の学習曲線と PGM 画像）は出力ディレクトリ（既定は `.`）に置かれる。
書式は [DATA_FORMATS.md](DATA_FORMATS.md) を参照。

実行時間は数秒である。Release ビルドで `from_scratch`（61 エポック）が約 1 秒、
`compare`（9 回の学習）が約 6 秒。[PERFORMANCE.md](PERFORMANCE.md) を参照。

3 つ目の実行ファイル `export_reference` はデモではない。分割・初期重み・
ミニバッチ順序を書き出し、[`reference/`](../reference/README.md) にある
PyTorch/TensorFlow 版が実行を厳密に再現できるようにするためのものである。

```bash
./build/export_reference          # output/reference/ に書き出す
```

---

## Windows: run_demo.ps1

PowerShell 7 以降。configure、ビルド、`ctest`、そして 2 つのデモをプログレスバーと
ASCII チャート付きで実行するところまでを 1 コマンドで行う。

```powershell
.\run_demo.ps1              # フル実行
.\run_demo.ps1 -SkipBuild   # ビルドとテストを飛ばし、デモだけ実行
.\run_demo.ps1 > demo.log   # 非対話実行
```

`Clear-Host` とプログレスバーはコンソールのカーソルを直接操作し、さらにこの
スクリプトは `Read-Host` で 2 回停止する。標準出力がパイプやファイルの場合、
どちらも機能しない。カーソル操作は「ハンドルが無効です」を送出し、
`$ErrorActionPreference = "Stop"` によってそれが実行の中断になる。そこで起動時に
実コンソールの有無を判定し（`[Console]::IsOutputRedirected` を見たうえで
`$Host.UI.RawUI.CursorPosition` の読み取りを試す）、コンソールがなければ
プロンプトを省略し、ブロックを再描画する代わりに 1 エポック 1 行を出力する。
`-NonInteractive` を渡せば端末上でも同じ動作を強制できる。

見せ方のためのラッパーである。ここで行うことはすべて素の CMake コマンドで到達でき、
ライブラリ側がこれに依存することはない。

---

## 継続的インテグレーション

`.github/workflows/ci.yml`。`main` への push とすべてのプルリクエストで走る。

**`build-and-test`** — configure、ビルド、`ctest` を Release で。`fail-fast: false`
なので、1 つのツールチェーンの失敗が他を隠さない。

| OS | コンパイラ |
|---|---|
| ubuntu-latest | GCC |
| ubuntu-latest | Clang |
| macos-latest | AppleClang |
| windows-latest | MSVC |

CI では警告がエラーなので、4 つのいずれか 1 つで警告が出ればビルドは失敗する。

**`format-check`** — `git ls-files '*.cpp' '*.hpp'` に対する
`clang-format --dry-run --Werror`。push 前に同じ検査をローカルで実行できる。

```bash
clang-format -i $(git ls-files '*.cpp' '*.hpp')
```

スタイルはリポジトリの `.clang-format` に従う。

---

## トラブルシューティング

| 症状 | 原因と対処 |
|---|---|
| `error: 'concept' does not name a type` のような構文エラー | コンパイラが C++20 に対して古い。[必要なもの](#必要なもの)の表を確認する |
| 新しいコンパイラで `-Werror` が失敗する | 新しい診断が原因。`-DNNSCRATCH_WARNINGS_AS_ERRORS=OFF` で configure し、修正して報告する |
| `load_digits: cannot open ...` | リポジトリのルートから実行するか、CSV のパスを明示する: `./build/from_scratch data/digits.csv` |
| `add_subdirectory` 後にテストターゲットがない | 仕様。トップレベルでないとき `NNSCRATCH_BUILD_TESTS` は `OFF` になる。必要なら `ON` にする |
| `find_package(nnscratch)` が見つからない | `-DCMAKE_PREFIX_PATH=<インストール先>` を足す |
| `find_package(nnscratch 0.1)` が 0.2 を拒否する | 0.x での `SameMajorVersion` の挙動。インストールしたバージョンを正確に要求する |
| Windows でデモの出力が文字化けする | ビルドではなくコンソールのコードページが原因。`/utf-8` は設定済みなので、`chcp 65001` か Windows Terminal を試す |
| 正解率が README と 0.数 % 違う | ツールチェーンをまたぐと想定内。`std::normal_distribution` は移植性を持たない（[ARCHITECTURE.md](ARCHITECTURE.md#決定性)） |
