// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
//
// Part 1 — "untrained -> trained": watch a 3-layer MLP go from random guessing
// to reading 8x8 handwritten digits, implemented entirely with this library.
//
// Usage: from_scratch [digits.csv] [output_dir]
#include <cstdio>
#include <fstream>
#include <string>

#include "nnscratch/nnscratch.hpp"
#include "nnscratch/pgm.hpp"

#ifndef NNSCRATCH_DATA_DIR
#define NNSCRATCH_DATA_DIR "."
#endif

namespace {

void write_learning_curve(const std::string& path, const nn::History& h) {
    std::ofstream out(path);
    out << "epoch,train_loss,train_acc,test_acc\n";
    for (std::size_t i = 0; i < h.epoch.size(); ++i) {
        out << h.epoch[i] << ',' << h.loss[i] << ',' << h.train_acc[i] << ','
            << h.test_acc[i] << '\n';
    }
}

// Render the 64 first-layer neurons' weights as a grid of 8x8 images.
void write_learned_features(const std::string& path, nn::Dense& first) {
    const nn::Tensor& W = first.weight();  // (64 in, 64 out)
    std::vector<std::vector<double>> cells;
    for (std::size_t neuron = 0; neuron < W.cols(); ++neuron) {
        std::vector<double> cell(64);
        for (std::size_t in = 0; in < 64; ++in) cell[in] = W(in, neuron);
        cells.push_back(std::move(cell));
    }
    nn::write_pgm_grid(path, cells, /*cell=*/8, /*cols=*/8);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string csv =
        argc > 1 ? argv[1] : std::string(NNSCRATCH_DATA_DIR) + "/digits.csv";
    const std::string out_dir = argc > 2 ? argv[2] : ".";

    std::puts("======================================================================");
    std::puts(" Loading data");
    std::puts("======================================================================");
    const nn::DigitsData data = nn::load_digits(csv);
    std::printf("train: %zu images / test: %zu images\n\n",
                data.train.labels.size(), data.test.labels.size());

    // Build an untrained network: 64 -> 64 -> 32 -> 10, ReLU hidden layers.
    nn::Rng rng(42);
    nn::Model net;
    nn::Dense& first = net.add<nn::Dense>(64, 64, rng, nn::Init::He);
    net.add<nn::ReLU>();
    net.add<nn::Dense>(64, 32, rng, nn::Init::He);
    net.add<nn::ReLU>();
    net.add<nn::Dense>(32, 10, rng, nn::Init::He);

    std::printf("Untrained test accuracy: %.1f%%  (≈10%% = random for 10 classes)\n\n",
                net.accuracy(data.test.flat, data.test.labels) * 100.0);

    std::puts("======================================================================");
    std::puts(" Training");
    std::puts("======================================================================");
    nn::SGD opt(0.3);
    nn::TrainConfig cfg;
    cfg.epochs = 60;
    cfg.batch_size = 32;
    cfg.verbose = true;
    const nn::History hist = nn::train(net, opt, data.train.flat, data.train.labels,
                                       data.test.flat, data.test.labels, cfg);

    std::printf("\nFinal test accuracy: %.1f%%\n",
                net.accuracy(data.test.flat, data.test.labels) * 100.0);

    write_learning_curve(out_dir + "/learning_curve.csv", hist);
    write_learned_features(out_dir + "/learned_features.pgm", first);
    std::printf("\nWrote: %s/learning_curve.csv, %s/learned_features.pgm\n",
                out_dir.c_str(), out_dir.c_str());
    return 0;
}
