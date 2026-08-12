# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""Is "CNN beats deep MLP" a real result on this dataset, or one lucky seed?

``compare`` reports a single number per architecture from a single seed, which
is not enough to rank three models that finish within a point of each other.
This repeats experiment 3 across many seeds and reports the spread.

    python reference/architecture_trials.py --seeds 10

The design follows compare.cpp: the data is held fixed, and the initialisation
and batch order are varied *together across all three architectures* within each
trial, so every trial is a fair three-way comparison. It uses the numpy
reference layers, which agree with nnscratch to 2e-15 (see README.md).

Requires the export first, for the train/test split::

    ./build/export_reference
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
import numpy_reference as nr                                         # noqa: E402

EPOCHS, BATCH, LR = 25, 32, 0.01          # experiment 3 of apps/compare.cpp
ARCHS = ['shallow', 'deep_mlp', 'cnn']


def he(rng, shape, fan_in):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)


def xavier(rng, shape, fan_in, fan_out):
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), shape)


def build(name, rng):
    """The three networks of compare.cpp, with nnscratch's init formulas."""
    if name == 'shallow':
        return [nr.Dense(xavier(rng, (64, 10), 64, 10), np.zeros(10))]
    if name == 'deep_mlp':
        return [nr.Dense(xavier(rng, (64, 64), 64, 64), np.zeros(64)), nr.ReLU(),
                nr.Dense(xavier(rng, (64, 32), 64, 32), np.zeros(32)), nr.ReLU(),
                nr.Dense(xavier(rng, (32, 10), 32, 10), np.zeros(10))]
    return [nr.Conv2D(he(rng, (8, 1, 3, 3), 1 * 3 * 3), np.zeros(8), stride=1, pad=0),
            nr.ReLU(), nr.Flatten(),
            nr.Dense(xavier(rng, (288, 10), 288, 10), np.zeros(10))]


def train_one(name, seed, data):
    x_flat_tr, x_img_tr, y_tr, x_flat_te, x_img_te, y_te = data
    layers = build(name, np.random.default_rng(seed))
    x_tr = x_img_tr if name == 'cnn' else x_flat_tr
    x_te = x_img_te if name == 'cnn' else x_flat_te

    params = [pg for layer in layers for pg in layer.params()]
    loss_fn = nr.SoftmaxCrossEntropy()
    opt = nr.Adam(LR)
    y1h = np.eye(10)[y_tr]
    # Offset so the batch order is decorrelated from the weights but still
    # identical across the three architectures in this trial.
    order_rng = np.random.default_rng(seed + 10_000)

    acc = best = 0.0
    for _ in range(EPOCHS):
        for idx in common.batches(order_rng.permutation(len(y_tr)), BATCH):
            logits = nr.forward(layers, x_tr[idx])
            loss_fn.forward(logits, y1h[idx])
            nr.backward(layers, loss_fn.backward())
            opt.step(params)
        acc = float((nr.forward(layers, x_te).argmax(1) == y_te).mean())
        best = max(best, acc)
    return acc, best


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--export-dir', default=common.DEFAULT_EXPORT_DIR)
    p.add_argument('--csv', default=common.DEFAULT_CSV)
    args = p.parse_args()

    pixels, labels = common.load_digits_csv(args.csv)
    tr, te = common.load_split(os.path.join(args.export_dir, 'split.txt'))
    data = (pixels[tr], pixels[tr].reshape(-1, 1, 8, 8), labels[tr],
            pixels[te], pixels[te].reshape(-1, 1, 8, 8), labels[te])
    n_test = len(te)

    final = {n: [] for n in ARCHS}
    best = {n: [] for n in ARCHS}
    for seed in range(1, args.seeds + 1):
        cells = []
        for name in ARCHS:
            f, b = train_one(name, seed, data)
            final[name].append(f)
            best[name].append(b)
            cells.append(f'{name} {f * 100:.1f}')
        print(f'seed {seed:2d}: ' + '  '.join(cells), flush=True)

    print(f'\n{args.seeds} seeds, final test accuracy (%)')
    print(f'  {"architecture":<14}{"mean":>8}{"sd":>7}{"min":>7}{"max":>7}{"mean best":>11}')
    for name in ARCHS:
        a, b = np.array(final[name]) * 100, np.array(best[name]) * 100
        sd = a.std(ddof=1) if len(a) > 1 else 0.0
        print(f'  {name:<14}{a.mean():>8.2f}{sd:>7.2f}{a.min():>7.1f}{a.max():>7.1f}'
              f'{b.mean():>11.2f}')

    cnn, deep, sh = (np.array(final[n]) for n in ARCHS[::-1])
    print(f'\n  deep_mlp > shallow  : {int((deep > sh).sum())}/{args.seeds} seeds'
          f'   mean {np.mean(deep - sh) * 100:+.2f} pt')
    print(f'  cnn      > shallow  : {int((cnn > sh).sum())}/{args.seeds} seeds'
          f'   mean {np.mean(cnn - sh) * 100:+.2f} pt')
    print(f'  cnn     >= deep_mlp : {int((cnn >= deep).sum())}/{args.seeds} seeds'
          f'   mean {np.mean(cnn - deep) * 100:+.2f} pt')
    print(f'\n  For scale: one image of the {n_test}-sample test set is '
          f'{100 / n_test:.2f} pt.')


if __name__ == '__main__':
    main()
