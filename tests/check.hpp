// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
//
// A deliberately tiny test harness: no external dependency, integrates with
// CTest via the process exit code (0 = pass, non-zero = fail count).
#ifndef NNSCRATCH_TEST_CHECK_HPP
#define NNSCRATCH_TEST_CHECK_HPP

#include <cmath>
#include <cstdio>
#include <string>

namespace nntest {

inline int& failures() {
    static int f = 0;
    return f;
}

inline void report(bool ok, const char* expr, const char* file, int line) {
    if (!ok) {
        ++failures();
        std::printf("FAIL  %s:%d  %s\n", file, line, expr);
    }
}

inline void report_close(double a, double b, double tol, const char* expr, const char* file,
                         int line) {
    const bool ok = std::fabs(a - b) <= tol;
    if (!ok) {
        ++failures();
        std::printf("FAIL  %s:%d  %s   (|%.6g - %.6g| = %.3g > %.3g)\n", file, line, expr, a,
                    b, std::fabs(a - b), tol);
    }
}

inline int summary(const char* suite) {
    if (failures() == 0) {
        std::printf("PASS  %s\n", suite);
        return 0;
    }
    std::printf("%d failure(s) in %s\n", failures(), suite);
    return 1;
}

}  // namespace nntest

#define CHECK(expr) ::nntest::report((expr), #expr, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol) ::nntest::report_close((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)

#endif  // NNSCRATCH_TEST_CHECK_HPP
