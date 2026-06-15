// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nnscratch contributors
#include "nnscratch/dataset.hpp"

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nnscratch/rng.hpp"

namespace nn {

namespace {

// One parsed CSV record: 64 pixels (0..16) + label.
struct Record {
    std::array<double, 64> pixels;
    int label;
};

std::vector<Record> read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_digits: cannot open " + path);

    std::vector<Record> records;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;            // comment lines
        if (line.rfind("p0,", 0) == 0 || line.rfind("p0", 0) == 0)
            continue;                                            // header row
        std::stringstream ss(line);
        std::string cell;
        Record rec{};
        std::size_t col = 0;
        while (std::getline(ss, cell, ',')) {
            const double val = std::stod(cell);
            if (col < 64) {
                rec.pixels[col] = val / 16.0;  // normalise to [0,1]
            } else {
                rec.label = static_cast<int>(val);
            }
            ++col;
        }
        if (col != 65) throw std::runtime_error("load_digits: malformed row in " + path);
        records.push_back(rec);
    }
    if (records.empty()) throw std::runtime_error("load_digits: no data in " + path);
    return records;
}

DigitsData::Split make_split(const std::vector<Record>& recs,
                             const std::vector<std::size_t>& idx) {
    DigitsData::Split s;
    const std::size_t n = idx.size();
    s.flat = Tensor(n, 64);
    s.img = Tensor({n, 1, 8, 8});
    s.labels.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
        const Record& rec = recs[idx[r]];
        for (std::size_t p = 0; p < 64; ++p) {
            s.flat(r, p) = rec.pixels[p];
            s.img.data()[r * 64 + p] = rec.pixels[p];  // (n,1,8,8) is row-major flat
        }
        s.labels[r] = rec.label;
    }
    return s;
}

}  // namespace

DigitsData load_digits(const std::string& csv_path, double train_frac,
                       std::uint64_t split_seed) {
    const std::vector<Record> recs = read_csv(csv_path);
    const std::size_t n = recs.size();

    Rng rng(split_seed);
    const std::vector<std::size_t> perm = rng.permutation(n);
    const std::size_t n_train = static_cast<std::size_t>(static_cast<double>(n) * train_frac);

    const auto split = static_cast<std::ptrdiff_t>(n_train);
    std::vector<std::size_t> tr(perm.begin(), perm.begin() + split);
    std::vector<std::size_t> te(perm.begin() + split, perm.end());

    DigitsData data;
    data.train = make_split(recs, tr);
    data.test = make_split(recs, te);
    return data;
}

}  // namespace nn
