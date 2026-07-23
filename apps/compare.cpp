// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 nnscratch contributors
//
// Part 2 — compare algorithmic choices on the same data and the same starting
// weights, varying one axis at a time:
//   1) optimizer    : SGD vs Momentum vs Adam
//   2) activation   : ReLU vs Tanh vs Sigmoid
//   3) architecture : shallow vs deep MLP vs CNN
//
// Usage: compare [digits.csv] [output_dir]
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "nnscratch/nnscratch.hpp"
#include "nnscratch/pgm.hpp"

#ifndef NNSCRATCH_DATA_DIR
#define NNSCRATCH_DATA_DIR "."
#endif

namespace {

constexpr std::uint64_t kSeed = 42;

// Factory for an activation layer, so a model can be rebuilt identically.
using ActFactory = std::function<std::unique_ptr<nn::Layer>()>;

nn::Model build_mlp(nn::Rng& rng, const ActFactory& act) {
    nn::Model m;
    m.add<nn::Dense>(64, 64, rng, nn::Init::Xavier);
    m.push(act());
    m.add<nn::Dense>(64, 32, rng, nn::Init::Xavier);
    m.push(act());
    m.add<nn::Dense>(32, 10, rng, nn::Init::Xavier);
    return m;
}

nn::Model build_shallow(nn::Rng& rng) {
    nn::Model m;
    m.add<nn::Dense>(64, 10, rng, nn::Init::Xavier);
    return m;
}

nn::Model build_cnn(nn::Rng& rng) {
    nn::Model m;
    m.add<nn::Conv2D>(1, 8, 3, 1, 0, rng, nn::Init::He);  // 1ch -> 8ch, 8x8 -> 6x6
    m.add<nn::ReLU>();
    m.add<nn::Flatten>();
    m.add<nn::Dense>(8 * 6 * 6, 10, rng, nn::Init::Xavier);
    return m;
}

void write_curve(const std::string& path, const std::map<std::string, nn::History>& runs) {
    std::ofstream out(path);
    out << "name,epoch,loss,test_acc\n";
    for (const auto& [name, h] : runs)
        for (std::size_t i = 0; i < h.epoch.size(); ++i)
            out << name << ',' << h.epoch[i] << ',' << h.loss[i] << ',' << h.test_acc[i]
                << '\n';
}

int epochs_to_reach(const nn::History& h, double threshold) {
    for (std::size_t i = 0; i < h.epoch.size(); ++i)
        if (h.test_acc[i] >= threshold) return h.epoch[i];
    return -1;
}

void print_table(const std::map<std::string, nn::History>& runs, double threshold = 0.90) {
    std::printf("\n%-20s%16s%15s%18s\n", "name", "final test acc", "best test acc",
                "epochs to 90%");
    std::puts("---------------------------------------------------------------------");
    for (const auto& [name, h] : runs) {
        double best = 0.0;
        for (double a : h.test_acc) best = std::max(best, a);
        const int e = epochs_to_reach(h, threshold);
        char ebuf[16];
        if (e >= 0)
            std::snprintf(ebuf, sizeof ebuf, "%d", e);
        else
            std::snprintf(ebuf, sizeof ebuf, "%s", "-");
        std::printf("%-20s%14.1f%%%13.1f%%%18s\n", name.c_str(), h.test_acc.back() * 100.0,
                    best * 100.0, ebuf);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string csv =
        argc > 1 ? argv[1] : std::string(NNSCRATCH_DATA_DIR) + "/digits.csv";
    const std::string out_dir = argc > 2 ? argv[2] : ".";

    const nn::DigitsData data = nn::load_digits(csv);
    std::printf("train %zu / test %zu\n", data.train.labels.size(),
                data.test.labels.size());

    nn::Rng rng(kSeed);
    const auto reseed = [&] { rng.reseed(kSeed); };

    // --- Experiment 1: optimizers (same MLP, same ReLU, same init) ----------
    std::puts("\n=== Experiment 1: optimizers (SGD vs Momentum vs Adam) ===");
    std::map<std::string, nn::History> exp1;
    {
        nn::TrainConfig cfg;
        cfg.epochs = 40;
        const ActFactory relu = [] { return std::make_unique<nn::ReLU>(); };

        reseed();
        nn::Model m = build_mlp(rng, relu);
        nn::SGD sgd(0.2);
        exp1["SGD"] = nn::train(m, sgd, data.train.flat, data.train.labels,
                                data.test.flat, data.test.labels, cfg);

        reseed();
        m = build_mlp(rng, relu);
        nn::Momentum mom(0.05, 0.9);
        exp1["Momentum"] = nn::train(m, mom, data.train.flat, data.train.labels,
                                     data.test.flat, data.test.labels, cfg);

        reseed();
        m = build_mlp(rng, relu);
        nn::Adam adam(0.01);
        exp1["Adam"] = nn::train(m, adam, data.train.flat, data.train.labels,
                                 data.test.flat, data.test.labels, cfg);
    }
    print_table(exp1);
    write_curve(out_dir + "/cmp_optimizers.csv", exp1);

    // --- Experiment 2: activations (same MLP, same SGD, same init) ----------
    std::puts("\n=== Experiment 2: activations (ReLU vs Tanh vs Sigmoid) ===");
    std::map<std::string, nn::History> exp2;
    {
        nn::TrainConfig cfg;
        cfg.epochs = 40;
        const std::vector<std::pair<std::string, ActFactory>> acts = {
            {"ReLU", [] { return std::make_unique<nn::ReLU>(); }},
            {"Tanh", [] { return std::make_unique<nn::Tanh>(); }},
            {"Sigmoid", [] { return std::make_unique<nn::Sigmoid>(); }},
        };
        for (const auto& [name, act] : acts) {
            reseed();
            nn::Model m = build_mlp(rng, act);
            nn::SGD sgd(0.5);
            exp2[name] = nn::train(m, sgd, data.train.flat, data.train.labels,
                                   data.test.flat, data.test.labels, cfg);
        }
    }
    print_table(exp2);
    write_curve(out_dir + "/cmp_activations.csv", exp2);

    // --- Experiment 3: architecture (same Adam optimizer) -------------------
    std::puts("\n=== Experiment 3: architecture (shallow vs deep MLP vs CNN) ===");
    std::map<std::string, nn::History> exp3;
    nn::Model cnn;
    {
        nn::TrainConfig cfg;
        cfg.epochs = 25;
        const ActFactory relu = [] { return std::make_unique<nn::ReLU>(); };

        reseed();
        nn::Model shallow = build_shallow(rng);
        nn::Adam a1(0.01);
        exp3["1_shallow"] = nn::train(shallow, a1, data.train.flat, data.train.labels,
                                      data.test.flat, data.test.labels, cfg);

        reseed();
        nn::Model deep = build_mlp(rng, relu);
        nn::Adam a2(0.01);
        exp3["2_deep_mlp"] = nn::train(deep, a2, data.train.flat, data.train.labels,
                                       data.test.flat, data.test.labels, cfg);

        reseed();
        cnn = build_cnn(rng);
        nn::Adam a3(0.01);
        exp3["3_cnn"] = nn::train(cnn, a3, data.train.img, data.train.labels,
                                  data.test.img, data.test.labels, cfg);
    }
    print_table(exp3);
    write_curve(out_dir + "/cmp_architecture.csv", exp3);

    // Render the CNN's learned 3x3 filters.
    {
        auto& conv = dynamic_cast<nn::Conv2D&>(cnn.layer(0));
        const nn::Tensor& W = conv.weight();  // (8, 1, 3, 3)
        std::vector<std::vector<double>> cells;
        for (std::size_t oc = 0; oc < W.dim(0); ++oc) {
            std::vector<double> cell(9);
            for (std::size_t i = 0; i < 9; ++i) cell[i] = W.data()[oc * 9 + i];
            cells.push_back(std::move(cell));
        }
        nn::write_pgm_grid(out_dir + "/cnn_filters.pgm", cells, /*cell=*/3, /*cols=*/8,
                           /*pad=*/1);
    }

    std::printf("\nWrote CSV curves and cnn_filters.pgm to %s\n", out_dir.c_str());
    return 0;
}
