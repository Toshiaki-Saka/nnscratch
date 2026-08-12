# Contributing

Thanks for your interest in nnscratch.

## Ground rules

- The core library must stay **dependency-free** (C++20 standard library only).
  Optional tooling (tests, demos) may not add runtime dependencies either.
- Clarity beats cleverness. This is a teaching library; readable code and honest
  math matter more than micro-optimizations.
- Every new layer must come with an entry exercised by the gradient-check test
  (`tests/test_gradcheck.cpp`). If `backward()` can't pass a finite-difference
  check, it isn't correct.

## Workflow

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Format with the bundled `.clang-format` before opening a pull request. CI pins
clang-format to an exact version, so install the same one rather than relying on
whatever your system package manager provides — different releases disagree
about line breaking, and an unpinned binary will fight with CI:

```bash
python -m pip install clang-format==22.1.8
clang-format -i $(git ls-files '*.cpp' '*.hpp')
```

The version lives in `.github/workflows/ci.yml`; keep the two in step when
bumping it.

Warnings are treated as errors (`-Werror`). Keep the build clean.

## Where to look next

- [docs_en/EXTENDING.md](docs_en/EXTENDING.md) — step-by-step guide to adding a
  layer, optimizer, loss or dataset, with worked examples.
- [docs_en/ARCHITECTURE.md](docs_en/ARCHITECTURE.md) — the layer contract, the
  ownership rules around `ParamGrad`, and the invariants you must not break.
- [docs_en/MATH.md](docs_en/MATH.md) — how each `backward()` is derived, which is
  what you will be doing for any new layer.
- [docs_en/TESTING.md](docs_en/TESTING.md) — how to add a test, and how to read a
  gradient-check failure.

日本語版は [docs_ja/](docs_ja/README.md) にあります。
