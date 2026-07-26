// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#ifndef NNSCRATCH_PGM_HPP
#define NNSCRATCH_PGM_HPP

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "nnscratch/tensor.hpp"

namespace nn {

/// Write a single-channel grayscale image (values in [0,1]) as a binary PGM.
/// Kept header-only and dependency-free so the demos can emit visual artifacts
/// without dragging in an image or plotting library.
inline void write_pgm(const std::string& path, const std::vector<double>& pixels,
                      std::size_t width, std::size_t height) {
    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << width << " " << height << "\n255\n";
    for (double v : pixels) {
        const int q = static_cast<int>(std::clamp(v, 0.0, 1.0) * 255.0 + 0.5);
        out.put(static_cast<char>(q));
    }
}

/// Tile a set of equally-sized square images into one grid PGM, each cell
/// independently min-max normalised so structure is visible. Useful for
/// rendering learned weights/filters.
inline void write_pgm_grid(const std::string& path, const std::vector<std::vector<double>>& cells,
                           std::size_t cell, std::size_t cols, std::size_t pad = 1) {
    const std::size_t rows = (cells.size() + cols - 1) / cols;
    const std::size_t W = cols * cell + (cols + 1) * pad;
    const std::size_t H = rows * cell + (rows + 1) * pad;
    std::vector<double> canvas(W * H, 0.5);  // gray background

    for (std::size_t idx = 0; idx < cells.size(); ++idx) {
        const std::size_t cr = idx / cols, cc = idx % cols;
        const auto& src = cells[idx];
        double lo = src[0], hi = src[0];
        for (double v : src) lo = std::min(lo, v), hi = std::max(hi, v);
        const double range = (hi - lo) > 1e-12 ? (hi - lo) : 1.0;
        const std::size_t y0 = pad + cr * (cell + pad);
        const std::size_t x0 = pad + cc * (cell + pad);
        for (std::size_t y = 0; y < cell; ++y)
            for (std::size_t x = 0; x < cell; ++x)
                canvas[(y0 + y) * W + (x0 + x)] = (src[y * cell + x] - lo) / range;
    }
    write_pgm(path, canvas, W, H);
}

}  // namespace nn

#endif  // NNSCRATCH_PGM_HPP
