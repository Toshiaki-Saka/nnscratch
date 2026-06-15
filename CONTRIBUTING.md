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

Format with the bundled `.clang-format` before opening a pull request:

```bash
clang-format -i $(git ls-files '*.cpp' '*.hpp')
```

Warnings are treated as errors (`-Werror`). Keep the build clean.
