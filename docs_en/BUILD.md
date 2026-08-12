# Build and integration

CMake ≥ 3.16, a C++20 compiler, nothing else. This document covers the build
options, the three ways to consume the library, the install layout, and the CI
matrix.

- [Requirements](#requirements)
- [Quick build](#quick-build)
- [Options](#options)
- [Targets](#targets)
- [Compiler flags](#compiler-flags)
- [Consuming the library](#consuming-the-library)
- [Install layout](#install-layout)
- [The demos](#the-demos)
- [Windows: run_demo.ps1](#windows-run_demops1)
- [Continuous integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

---

## Requirements

| | Minimum | Notes |
|---|---|---|
| CMake | 3.16 | 3.21+ provides `PROJECT_IS_TOP_LEVEL` natively; the project polyfills it below that |
| GCC | 10 | |
| Clang | 12 | |
| MSVC | 19.29 (VS 2019 16.10) | |
| Dependencies | *none* | C++20 standard library only, at build time and at runtime |

C++20 is requested through `target_compile_features(nnscratch PUBLIC cxx_std_20)`
with `CMAKE_CXX_EXTENSIONS OFF`, so consumers inherit the requirement and get
standard `-std=c++20`, not `-std=gnu++20`.

Only a modest slice of C++20 is used — structured bindings in range-for,
`[[nodiscard]]`, designated-ish aggregate init. There is nothing exotic to
work around on an older toolchain except the standard-version requirement
itself.

---

## Quick build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

`CMAKE_BUILD_TYPE` defaults to `Release` if you leave it unset on a
single-config generator — the project forces it in `CMakeLists.txt` — because an
unoptimised default would make the demos noticeably slow for no reason.

On a multi-config generator (Visual Studio, Xcode), the type is chosen at build
time instead:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

---

## Options

| Option | Default | Effect |
|---|---|---|
| `NNSCRATCH_BUILD_APPS` | `ON` when top-level, else `OFF` | Builds `from_scratch` and `compare` |
| `NNSCRATCH_BUILD_TESTS` | `ON` when top-level, else `OFF` | Builds the three tests and enables CTest |
| `NNSCRATCH_WARNINGS_AS_ERRORS` | `ON` | Adds `-Werror` / `/WX` |

The top-level defaults mean that `add_subdirectory(nnscratch)` from a parent
project builds *only* the library — no demo executables, no test targets
polluting your build — while a standalone checkout builds everything. That is
the intended behaviour and needs no configuration from you.

Turn off `NNSCRATCH_WARNINGS_AS_ERRORS` if a newer compiler introduces a
diagnostic the code has not seen yet:

```bash
cmake -S . -B build -DNNSCRATCH_WARNINGS_AS_ERRORS=OFF
```

---

## Targets

| Target | Kind | Notes |
|---|---|---|
| `nnscratch` | library | The core. `STATIC` unless `BUILD_SHARED_LIBS=ON` |
| `nnscratch::nnscratch` | alias | Use this name in `target_link_libraries` — it works identically whether the library is built in-tree or found with `find_package` |
| `nnscratch_warnings` | `INTERFACE` | The warning flag set. Linked `PRIVATE` and wrapped in `$<BUILD_INTERFACE:...>`, so it never leaks into installed consumers |
| `from_scratch`, `compare` | executables | Demos, gated on `NNSCRATCH_BUILD_APPS` |
| `test_tensor`, `test_gradcheck`, `test_optimizer` | executables | Gated on `NNSCRATCH_BUILD_TESTS`, registered with CTest |

`VERSION` and `SOVERSION` are set from `project(VERSION 0.1.0)`, so a shared
build produces a properly versioned `libnnscratch.so.0`.

---

## Compiler flags

Applied through the `nnscratch_warnings` interface target:

| Toolchain | Flags |
|---|---|
| MSVC | `/W4 /utf-8` (+ `/WX`) |
| GCC / Clang | `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion` (+ `-Werror`) |

`-Wconversion` and `-Wsign-conversion` are the demanding ones, and they are
deliberate: this code does index arithmetic with `std::size_t` while computing
in `double`, which is precisely where silent narrowing bugs live. Expect to
write explicit `static_cast`s in new code — the existing sources do throughout,
which is why the build is warning-clean on GCC, Clang, AppleClang and MSVC.

`/utf-8` is needed because the demos print non-ASCII characters (`≈`, `→`) in
string literals; without it MSVC misinterprets the source encoding.

---

## Consuming the library

### 1. Installed package

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build
cmake --install build
```

```cmake
find_package(nnscratch REQUIRED)
target_link_libraries(your_app PRIVATE nnscratch::nnscratch)
```

If you installed to a non-standard prefix, point CMake at it with
`-DCMAKE_PREFIX_PATH=/your/prefix`.

The package version file uses `SameMajorVersion` compatibility. At 0.x that
makes every *minor* bump incompatible, which is the conservative reading of
semantic versioning for a pre-1.0 library — `find_package(nnscratch 0.1 REQUIRED)`
will not accept 0.2.

### 2. `add_subdirectory`

```cmake
add_subdirectory(third_party/nnscratch)
target_link_libraries(your_app PRIVATE nnscratch::nnscratch)
```

Apps and tests switch themselves off automatically in this mode.

### 3. FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(nnscratch
    GIT_REPOSITORY https://github.com/Toshiaki-Saka/nnscratch.git
    GIT_TAG        main)
FetchContent_MakeAvailable(nnscratch)
target_link_libraries(your_app PRIVATE nnscratch::nnscratch)
```

Pin `GIT_TAG` to a release tag or commit SHA for reproducible builds.

### Without CMake

There is nothing to configure — add `include/` to the include path and compile
the nine files in `src/`:

```bash
g++ -std=c++20 -O2 -Iinclude src/*.cpp your_app.cpp -o your_app
```

No generated headers, no config step, no feature detection.

---

## Install layout

```
${CMAKE_INSTALL_PREFIX}/
├── include/nnscratch/*.hpp                   all public headers, incl. pgm.hpp
├── lib/                                      libnnscratch.a  (or .so / .lib)
└── lib/cmake/nnscratch/
    ├── nnscratchConfig.cmake                 generated from cmake/nnscratchConfig.cmake.in
    ├── nnscratchConfigVersion.cmake          SameMajorVersion
    └── nnscratchTargets.cmake                exported target, nnscratch:: namespace
```

Paths follow `GNUInstallDirs`, so `lib` becomes `lib64` on distributions that
use it. The demos and the dataset are **not** installed — they are
demonstrations, not deliverables.

---

## The demos

```bash
./build/from_scratch          # Part 1: untrained -> trained
./build/compare               # Part 2: three controlled experiments
```

Both accept optional arguments:

```
from_scratch [digits.csv] [output_dir]
compare      [digits.csv] [output_dir]
```

The default CSV path is baked in at configure time via a compile definition:

```cmake
target_compile_definitions(${demo} PRIVATE
    NNSCRATCH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
```

so the demos find the bundled dataset regardless of the working directory —
but that absolute path points into the *source tree*. A copied binary on another
machine needs the CSV path passed explicitly.

Outputs (CSV learning curves and PGM images) land in the output directory,
default `.`. Formats: [DATA_FORMATS.md](DATA_FORMATS.md).

Runtimes are seconds: about 1 s for `from_scratch` (61 epochs) and about 6 s for
`compare` (nine training runs) in a Release build. See
[PERFORMANCE.md](PERFORMANCE.md).

---

## Windows: run_demo.ps1

PowerShell 7+, does everything in one command — configure, build, `ctest`, then
both demos with progress bars and ASCII charts:

```powershell
.\run_demo.ps1              # full run
.\run_demo.ps1 -SkipBuild   # skip build/test, just run the demos
```

It is a presentation wrapper. Everything it does is reachable through plain
CMake commands; nothing in the library depends on it.

---

## Continuous integration

`.github/workflows/ci.yml`, on every push to `main` and every pull request.

**`build-and-test`** — configure, build and `ctest`, Release, `fail-fast: false`
so one broken toolchain does not hide the others:

| OS | Compiler |
|---|---|
| ubuntu-latest | GCC |
| ubuntu-latest | Clang |
| macos-latest | AppleClang |
| windows-latest | MSVC |

Warnings are errors in CI, so a warning on any one of the four fails the build.

**`format-check`** — `clang-format --dry-run --Werror` over
`git ls-files '*.cpp' '*.hpp'`. Run the same check locally before pushing:

```bash
clang-format -i $(git ls-files '*.cpp' '*.hpp')
```

The style comes from the repository's `.clang-format`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: 'concept' does not name a type` or similar syntax errors | Compiler too old for C++20. Check the [requirements](#requirements) table |
| `-Werror` failure on a new compiler | A newer diagnostic. Configure with `-DNNSCRATCH_WARNINGS_AS_ERRORS=OFF`, then fix and report the warning |
| `load_digits: cannot open ...` | Run from the repository root, or pass the CSV path explicitly: `./build/from_scratch data/digits.csv` |
| No test targets after `add_subdirectory` | By design — `NNSCRATCH_BUILD_TESTS` defaults to `OFF` when not top-level. Set it `ON` if you want them |
| `find_package(nnscratch)` not found | Add `-DCMAKE_PREFIX_PATH=<install prefix>` |
| `find_package(nnscratch 0.1)` rejects 0.2 | `SameMajorVersion` at 0.x. Request the exact version you installed |
| Mojibake in the demo output on Windows | Console code page, not the build. `/utf-8` is already set; try `chcp 65001` or Windows Terminal |
| Accuracy differs by a few tenths from the README | Expected across toolchains — `std::normal_distribution` is not portable ([ARCHITECTURE.md](ARCHITECTURE.md#determinism)) |
