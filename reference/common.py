# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""Shared loading code for the nnscratch reference implementations.

Every script in this directory trains the *same* network on the *same* data
from the *same* initial weights in the *same* mini-batch order as the C++ demo.
None of that can be reconstructed from a seed -- ``std::normal_distribution``
and ``std::uniform_int_distribution`` are not portable -- so it all travels as
data, written by ``export_reference``:

    cmake --build build --config Release
    ./build/export_reference                 # -> output/reference/

This module reads that export. It depends only on numpy.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_EXPORT_DIR = os.path.join('output', 'reference')
DEFAULT_CSV = os.path.join('data', 'digits.csv')


# --------------------------------------------------------------------------
# the export
# --------------------------------------------------------------------------

def load_config(path):
    """Parse a ``*_config.txt`` into {key: value}, numbers converted.

    Repeated ``layer`` lines collect into cfg['layer'] as a list of token lists.
    """
    cfg, layers = {}, []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, _, rest = line.partition(' ')
            if key == 'layer':
                layers.append(rest.split())
                continue
            try:
                cfg[key] = int(rest)
            except ValueError:
                try:
                    cfg[key] = float(rest)
                except ValueError:
                    cfg[key] = rest
    cfg['layer'] = layers
    return cfg


def load_weights(path):
    """Parse a ``*_init_weights.txt`` into {name: ndarray} (float64, row-major)."""
    out = {}
    with open(path, encoding='utf-8') as fh:
        tokens = [t for line in fh if not line.startswith('#') for t in line.split()]
    i = 0
    while i < len(tokens):
        if tokens[i] != 'tensor':
            raise ValueError(f'{path}: expected "tensor" at token {i}, got {tokens[i]!r}')
        name = tokens[i + 1]
        rank = int(tokens[i + 2])
        shape = [int(t) for t in tokens[i + 3:i + 3 + rank]]
        i += 3 + rank
        n = int(np.prod(shape)) if shape else 0
        out[name] = np.array(tokens[i:i + n], dtype=np.float64).reshape(shape)
        i += n
    return out


def load_split(path):
    """Return (train_idx, test_idx) as int arrays of record positions."""
    train, test = [], []
    target = None
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line == 'train':
                target = train
                continue
            if line == 'test':
                target = test
                continue
            head = line.split()[0]
            if not head.isdigit() or target is None:
                continue          # a "key value" header line
            target.extend(int(t) for t in line.split())
    return np.array(train, dtype=np.int64), np.array(test, dtype=np.int64)


def load_batch_order(path):
    """Return a list of per-epoch index arrays, epoch 1 first."""
    epochs, current = [], None
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('epoch '):
                current = []
                epochs.append(current)
                continue
            head = line.split()[0]
            if not head.isdigit() or current is None:
                continue          # a header line such as "n_train 1437"
            current.extend(int(t) for t in line.split())
    return [np.array(e, dtype=np.int64) for e in epochs]


# --------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------

def load_digits_csv(path):
    """Read data/digits.csv exactly as src/dataset.cpp does.

    Skips comments and the header, divides the 0..16 pixels by 16, and returns
    (pixels (N, 64) float64, labels (N,) int64) in file order -- the order the
    exported split indexes into.
    """
    pixels, labels = [], []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('p0'):
                continue
            cells = line.split(',')
            if len(cells) != 65:
                raise ValueError(f'{path}: malformed row with {len(cells)} fields')
            pixels.append([float(c) / 16.0 for c in cells[:64]])
            labels.append(int(float(cells[64])))
    return np.array(pixels, dtype=np.float64), np.array(labels, dtype=np.int64)


@dataclass
class Reference:
    """Everything one reference run needs, already split and shaped."""
    name: str
    cfg: dict
    weights: dict
    batch_order: list
    x_train: np.ndarray          # (n_train, 64) or (n_train, 1, 8, 8)
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    nnscratch_curve: np.ndarray  # (epochs+1, 4): epoch, loss, train_acc, test_acc


def load_reference(model, export_dir=DEFAULT_EXPORT_DIR, csv_path=DEFAULT_CSV):
    """Load the full export for 'mlp' or 'cnn'."""
    cfg = load_config(os.path.join(export_dir, f'{model}_config.txt'))
    weights = load_weights(os.path.join(export_dir, f'{model}_init_weights.txt'))
    order = load_batch_order(os.path.join(export_dir, f'{model}_batch_order.txt'))
    train_idx, test_idx = load_split(os.path.join(export_dir, 'split.txt'))

    pixels, labels = load_digits_csv(csv_path)
    x_train, y_train = pixels[train_idx], labels[train_idx]
    x_test, y_test = pixels[test_idx], labels[test_idx]
    if cfg['input'] == 'img':                      # (N, 64) -> (N, 1, 8, 8)
        x_train = x_train.reshape(-1, 1, 8, 8)
        x_test = x_test.reshape(-1, 1, 8, 8)

    curve = np.loadtxt(os.path.join(export_dir, f'{model}_curve_nnscratch.csv'),
                       delimiter=',', skiprows=1)

    if len(order) != cfg['epochs']:
        raise ValueError(f'{model}: {len(order)} batch orders for {cfg["epochs"]} epochs')
    return Reference(model, cfg, weights, order, x_train, y_train, x_test, y_test, curve)


# --------------------------------------------------------------------------
# shared reporting
# --------------------------------------------------------------------------

def batches(order, batch_size):
    """Slice one epoch's index permutation into mini-batches, ragged tail included."""
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def write_curve(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('epoch,train_loss,train_acc,test_acc\n')
        for ep, loss, tr, te in rows:
            fh.write(f'{int(ep)},{loss:.17g},{tr:.17g},{te:.17g}\n')
    print(f'wrote {path}')


def report(name, rows, ref):
    """Print the epoch-by-epoch agreement with the nnscratch run."""
    got = np.array([[r[1], r[2], r[3]] for r in rows])
    want = ref.nnscratch_curve[:, 1:4]
    n = min(len(got), len(want))
    d_loss = np.abs(got[:n, 0] - want[:n, 0])
    d_acc = np.abs(got[:n, 2] - want[:n, 2])
    print(f'\n{name} vs nnscratch ({ref.name}):')
    print(f'  final test acc : {got[-1, 2]:.4f}  (nnscratch {want[-1, 2]:.4f})')
    print(f'  max |loss diff|      : {d_loss.max():.3e}')
    print(f'  max |test acc diff|  : {d_acc.max():.3e}')
    print(f'  epoch-0 loss diff    : {d_loss[0]:.3e}   '
          f'(untrained: pure forward-pass agreement)')


def common_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--model', choices=['mlp', 'cnn', 'both'], default='both')
    p.add_argument('--export-dir', default=DEFAULT_EXPORT_DIR)
    p.add_argument('--csv', default=DEFAULT_CSV)
    p.add_argument('--out-dir', default=DEFAULT_EXPORT_DIR)
    p.add_argument('--dtype', choices=['float64', 'float32'], default='float64',
                   help='float64 matches nnscratch; float32 is the framework default '
                        'and shows how much of the gap is precision')
    return p


def models_from(args):
    return ['mlp', 'cnn'] if args.model == 'both' else [args.model]
