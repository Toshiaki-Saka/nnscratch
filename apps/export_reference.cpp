// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
//
// Export everything a framework port needs to reproduce an nnscratch run
// *exactly*: the train/test split, the initial weights, the per-epoch
// mini-batch order, and nnscratch's own learning curve.
//
// This exists because two of those cannot be reconstructed from a seed on the
// other side. `Rng` wraps std::mt19937_64, whose output the standard pins down
// bit for bit, but `Rng::normal` and `Rng::permutation` go through
// std::normal_distribution / std::uniform_int_distribution, whose mapping from
// engine output to values is implementation-defined. So the weights and the
// shuffles have to travel as data, not as a seed.
//
// Usage: export_reference [digits.csv] [output_dir]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nnscratch/nnscratch.hpp"

#ifndef NNSCRATCH_DATA_DIR
#define NNSCRATCH_DATA_DIR "."
#endif

namespace {

// --- output helpers --------------------------------------------------------

std::ofstream open_out(const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    return out;
}

// 17 significant digits round-trips an IEEE double exactly.
void write_tensor(std::ostream& out, const std::string& name, const nn::Tensor& t) {
    out << "tensor " << name << ' ' << t.rank();
    for (std::size_t d : t.shape()) out << ' ' << d;
    out << '\n' << std::setprecision(17);
    const std::vector<double>& d = t.data();
    for (std::size_t i = 0; i < d.size(); ++i) {
        const bool eol = (i + 1) % 8 == 0 || i + 1 == d.size();
        out << d[i] << (eol ? '\n' : ' ');
    }
    if (d.empty()) out << '\n';
}

void write_indices(std::ostream& out, const std::vector<std::size_t>& idx) {
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const bool eol = (i + 1) % 20 == 0 || i + 1 == idx.size();
        out << idx[i] << (eol ? '\n' : ' ');
    }
    if (idx.empty()) out << '\n';
}

void write_curve(const std::string& path, const nn::History& h) {
    std::ofstream out = open_out(path);
    out << "epoch,train_loss,train_acc,test_acc\n" << std::setprecision(17);
    for (std::size_t i = 0; i < h.epoch.size(); ++i) {
        out << h.epoch[i] << ',' << h.loss[i] << ',' << h.train_acc[i] << ',' << h.test_acc[i]
            << '\n';
    }
}

// --- CSV re-parse, purely to validate the exported split -------------------

// Mirrors src/dataset.cpp's parser. The point is not to load data -- that is
// load_digits()'s job -- but to prove that "parse the CSV, then apply the
// exported permutation" reproduces load_digits() element for element, which is
// exactly what the Python side will do.
std::vector<std::vector<double>> reparse_csv(const std::string& path,
                                             std::vector<int>& labels_out) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("p0", 0) == 0) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> px;
        std::size_t col = 0;
        int label = 0;
        while (std::getline(ss, cell, ',')) {
            const double v = std::stod(cell);
            if (col < 64) {
                px.push_back(v / 16.0);
            } else {
                label = static_cast<int>(v);
            }
            ++col;
        }
        if (col != 65) throw std::runtime_error("malformed row in " + path);
        rows.push_back(std::move(px));
        labels_out.push_back(label);
    }
    return rows;
}

double check_split(const std::vector<std::vector<double>>& rows,
                   const std::vector<int>& labels, const std::vector<std::size_t>& idx,
                   const nn::DigitsData::Split& split) {
    if (idx.size() != split.labels.size()) {
        throw std::runtime_error("split size mismatch");
    }
    double worst = 0.0;
    for (std::size_t r = 0; r < idx.size(); ++r) {
        if (labels[idx[r]] != split.labels[r]) {
            throw std::runtime_error("split label mismatch at row " + std::to_string(r));
        }
        for (std::size_t p = 0; p < 64; ++p) {
            worst = std::max(worst, std::fabs(rows[idx[r]][p] - split.flat(r, p)));
        }
    }
    return worst;
}

// --- the two reference configurations --------------------------------------

constexpr std::uint64_t kModelSeed = 42;
constexpr double kMlpLr = 0.3;   // from_scratch.cpp
constexpr int kMlpEpochs = 60;   // from_scratch.cpp
constexpr double kCnnLr = 0.01;  // compare.cpp, experiment 3
constexpr int kCnnEpochs = 25;   // compare.cpp, experiment 3

void write_mlp_config(const std::string& path, const nn::TrainConfig& cfg) {
    std::ofstream out = open_out(path);
    out << "# nnscratch reference configuration -- mirrors apps/from_scratch.cpp\n"
        << "model mlp\n"
        << "input flat\n"
        << "model_seed " << kModelSeed << "\n"
        << "optimizer sgd\n"
        << "lr " << kMlpLr << "\n"
        << "epochs " << cfg.epochs << "\n"
        << "batch_size " << cfg.batch_size << "\n"
        << "batch_seed " << cfg.batch_seed << "\n"
        << "num_classes 10\n"
        << "layers 5\n"
        << "layer 0 dense 64 64 he\n"
        << "layer 1 relu\n"
        << "layer 2 dense 64 32 he\n"
        << "layer 3 relu\n"
        << "layer 4 dense 32 10 he\n";
}

void write_cnn_config(const std::string& path, const nn::TrainConfig& cfg) {
    std::ofstream out = open_out(path);
    out << "# nnscratch reference configuration -- mirrors experiment 3 of apps/compare.cpp\n"
        << "model cnn\n"
        << "input img\n"
        << "model_seed " << kModelSeed << "\n"
        << "optimizer adam\n"
        << "lr " << kCnnLr << "\n"
        << "beta1 0.9\n"
        << "beta2 0.999\n"
        << "eps 1e-8\n"
        << "epochs " << cfg.epochs << "\n"
        << "batch_size " << cfg.batch_size << "\n"
        << "batch_seed " << cfg.batch_seed << "\n"
        << "num_classes 10\n"
        << "layers 4\n"
        << "layer 0 conv2d 1 8 3 1 0 he\n"
        << "layer 1 relu\n"
        << "layer 2 flatten\n"
        << "layer 3 dense 288 10 xavier\n";
}

// Replays the permutation sequence train() draws internally: one per epoch,
// for epochs 1..N, from a generator seeded with cfg.batch_seed.
void write_batch_order(const std::string& path, const nn::TrainConfig& cfg, std::size_t n) {
    std::ofstream out = open_out(path);
    out << "# per-epoch mini-batch order: a permutation of [0, n_train), one block per\n"
        << "# epoch, in the order train() draws them from a generator seeded with\n"
        << "# batch_seed. Epoch 0 records metrics only and draws nothing.\n"
        << "epochs " << cfg.epochs << "\n"
        << "n_train " << n << "\n"
        << "batch_size " << cfg.batch_size << "\n";
    nn::Rng batch_rng(cfg.batch_seed);
    for (int ep = 1; ep <= cfg.epochs; ++ep) {
        out << "epoch " << ep << '\n';
        write_indices(out, batch_rng.permutation(n));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string csv =
            argc > 1 ? argv[1] : std::string(NNSCRATCH_DATA_DIR) + "/digits.csv";
        const std::string dir = argc > 2 ? argv[2] : std::string("output/reference");
        std::filesystem::create_directories(dir);

        std::puts("======================================================================");
        std::puts(" Exporting nnscratch reference data");
        std::puts("======================================================================");

        const nn::DigitsData data = nn::load_digits(csv);
        const std::size_t n_train = data.train.labels.size();
        const std::size_t n_test = data.test.labels.size();
        const std::size_t n = n_train + n_test;
        std::printf("dataset: %zu records -> train %zu / test %zu\n", n, n_train, n_test);

        // load_digits() splits on a permutation seeded with split_seed; replay it.
        nn::Rng split_rng(0);
        const std::vector<std::size_t> perm = split_rng.permutation(n);
        const std::vector<std::size_t> train_idx(
            perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(n_train));
        const std::vector<std::size_t> test_idx(
            perm.begin() + static_cast<std::ptrdiff_t>(n_train), perm.end());

        // Prove the replay is faithful before writing it out.
        std::vector<int> labels;
        const std::vector<std::vector<double>> rows = reparse_csv(csv, labels);
        if (rows.size() != n) throw std::runtime_error("record count mismatch");
        const double e_tr = check_split(rows, labels, train_idx, data.train);
        const double e_te = check_split(rows, labels, test_idx, data.test);
        std::printf("split replay verified: max |pixel diff| = %.1e (train), %.1e (test)\n",
                    e_tr, e_te);
        if (e_tr != 0.0 || e_te != 0.0) throw std::runtime_error("split replay is not exact");

        {
            std::ofstream out = open_out(dir + "/split.txt");
            out << "# train/test split as indices into digits.csv record order (0-based,\n"
                << "# comment and header lines excluded). Verified to reproduce\n"
                << "# load_digits() exactly.\n"
                << "n_records " << n << "\ntrain_frac 0.8\nsplit_seed 0\n"
                << "n_train " << n_train << "\nn_test " << n_test << "\ntrain\n";
            write_indices(out, train_idx);
            out << "test\n";
            write_indices(out, test_idx);
        }

        // --- MLP: the from_scratch.cpp network --------------------------------
        {
            nn::Rng rng(kModelSeed);
            nn::Model net;
            nn::Dense& d0 = net.add<nn::Dense>(64, 64, rng, nn::Init::He);
            net.add<nn::ReLU>();
            nn::Dense& d1 = net.add<nn::Dense>(64, 32, rng, nn::Init::He);
            net.add<nn::ReLU>();
            nn::Dense& d2 = net.add<nn::Dense>(32, 10, rng, nn::Init::He);

            {
                std::ofstream out = open_out(dir + "/mlp_init_weights.txt");
                out << "# initial parameters, before any update. Row-major, 17 digits.\n"
                    << "# Dense W is (n_in, n_out) and y = x . W + b.\n";
                write_tensor(out, "dense0.W", d0.weight());
                write_tensor(out, "dense0.b", d0.bias());
                write_tensor(out, "dense2.W", d1.weight());
                write_tensor(out, "dense2.b", d1.bias());
                write_tensor(out, "dense4.W", d2.weight());
                write_tensor(out, "dense4.b", d2.bias());
            }

            nn::TrainConfig cfg;
            cfg.epochs = kMlpEpochs;
            cfg.batch_size = 32;
            write_mlp_config(dir + "/mlp_config.txt", cfg);
            write_batch_order(dir + "/mlp_batch_order.txt", cfg, n_train);

            nn::SGD opt(kMlpLr);
            const nn::History h = nn::train(net, opt, data.train.flat, data.train.labels,
                                            data.test.flat, data.test.labels, cfg);
            write_curve(dir + "/mlp_curve_nnscratch.csv", h);
            std::printf("mlp : final test acc %.4f (%d epochs)\n", h.test_acc.back(),
                        cfg.epochs);
        }

        // --- CNN: experiment 3 of compare.cpp ---------------------------------
        {
            nn::Rng rng(kModelSeed);
            nn::Model net;
            nn::Conv2D& c0 = net.add<nn::Conv2D>(1, 8, 3, 1, 0, rng, nn::Init::He);
            net.add<nn::ReLU>();
            net.add<nn::Flatten>();
            nn::Dense& d0 = net.add<nn::Dense>(8 * 6 * 6, 10, rng, nn::Init::Xavier);

            {
                std::ofstream out = open_out(dir + "/cnn_init_weights.txt");
                out << "# initial parameters, before any update. Row-major, 17 digits.\n"
                    << "# Conv2D W is (out_c, in_c, k, k), cross-correlation (no kernel "
                       "flip).\n"
                    << "# Flatten uses channel-major order: (c, h, w) -> c*H*W + h*W + w.\n";
                write_tensor(out, "conv0.W", c0.weight());
                write_tensor(out, "conv0.b", *c0.params_and_grads()[1].param);
                write_tensor(out, "dense3.W", d0.weight());
                write_tensor(out, "dense3.b", d0.bias());
            }

            nn::TrainConfig cfg;
            cfg.epochs = kCnnEpochs;
            cfg.batch_size = 32;
            write_cnn_config(dir + "/cnn_config.txt", cfg);
            write_batch_order(dir + "/cnn_batch_order.txt", cfg, n_train);

            nn::Adam opt(kCnnLr);
            const nn::History h = nn::train(net, opt, data.train.img, data.train.labels,
                                            data.test.img, data.test.labels, cfg);
            write_curve(dir + "/cnn_curve_nnscratch.csv", h);
            std::printf("cnn : final test acc %.4f (%d epochs)\n", h.test_acc.back(),
                        cfg.epochs);
        }

        std::printf("\nWrote the reference export to %s\n", dir.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "export_reference: %s\n", e.what());
        return 1;
    }
}
