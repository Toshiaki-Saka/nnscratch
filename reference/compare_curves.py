# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""Overlay every implementation's learning curve and quantify the gaps.

    python reference/compare_curves.py

Reads whatever ``*_curve_*.csv`` files exist in the export directory, prints a
divergence table, and -- if matplotlib is installed -- writes an overlay plot.
Run the implementations first::

    ./build/export_reference
    python reference/numpy_reference.py
    python reference/pytorch_reference.py
    python reference/tensorflow_reference.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

BASELINE = 'nnscratch'
# nnscratch reports log(p + 1e-9); the frameworks do not. On this data that is a
# ~1e-8 offset in the loss and nothing else, so gaps at that scale mean "equal".
FLOOR_ARTIFACT = 5e-8


def discover(export_dir, model):
    """{implementation name: curve array} for one model."""
    out = {}
    for path in sorted(glob.glob(os.path.join(export_dir, f'{model}_curve_*.csv'))):
        name = re.sub(rf'^{model}_curve_', '', os.path.basename(path))[:-len('.csv')]
        out[name] = np.loadtxt(path, delimiter=',', skiprows=1)
    return out


def table(model, curves):
    if BASELINE not in curves:
        print(f'{model}: no {BASELINE} curve in the export directory -- skipping')
        return
    base = curves[BASELINE]
    print(f'\n{model.upper()}  ({len(base) - 1} epochs, baseline = {BASELINE})')
    print(f'  {"implementation":<22}{"final test acc":>15}{"max |loss gap|":>16}'
          f'{"max |acc gap|":>15}  verdict')
    print('  ' + '-' * 74)
    for name, c in curves.items():
        n = min(len(c), len(base))
        d_loss = float(np.abs(c[:n, 1] - base[:n, 1]).max())
        d_acc = float(np.abs(c[:n, 3] - base[:n, 3]).max())
        if name == BASELINE:
            verdict = '(baseline)'
        elif d_loss <= FLOOR_ARTIFACT and d_acc == 0.0:
            verdict = 'identical'
        elif d_acc == 0.0:
            verdict = 'same predictions, loss drifts'
        else:
            verdict = 'diverges'
        print(f'  {name:<22}{c[-1, 3]:>15.4f}{d_loss:>16.3e}{d_acc:>15.3e}  {verdict}')


def plot(path, models, curves_by_model):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nmatplotlib not installed -- skipping the plot')
        return

    models = [m for m in models if curves_by_model.get(m)]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        curves = curves_by_model[model]
        base = curves.get(BASELINE)
        # Baseline goes down first as a thick pale line; the others are dashed on
        # top of it. Otherwise the agreement is so exact that only one curve shows.
        ordered = ([(BASELINE, base)] if base is not None else []) + \
                  [(k, v) for k, v in sorted(curves.items()) if k != BASELINE]
        for name, c in ordered:
            style = dict(lw=5, alpha=0.25, color='k', zorder=1) if name == BASELINE \
                else dict(lw=1.4, alpha=0.9, ls='--', zorder=2)
            axes[row][0].plot(c[:, 0], c[:, 1], label=name, **style)
            axes[row][1].plot(c[:, 0], c[:, 3] * 100, label=name, **style)
            if base is None or name == BASELINE:
                continue
            n = min(len(c), len(base))
            gap = np.abs(c[:n, 1] - base[:n, 1])
            axes[row][2].plot(c[:n, 0], np.maximum(gap, 1e-17), label=name,
                              lw=1.4, alpha=0.9)
        axes[row][0].set(xlabel='epoch', ylabel='train loss', yscale='log',
                         title=f'{model} — loss (all implementations)')
        axes[row][1].set(xlabel='epoch', ylabel='test accuracy (%)',
                         title=f'{model} — test accuracy')
        axes[row][2].axhline(FLOOR_ARTIFACT, color='k', ls=':', lw=1,
                             label='log(p+1e-9) floor')
        axes[row][2].set(xlabel='epoch', ylabel='|loss − nnscratch|', yscale='log',
                         title=f'{model} — divergence from nnscratch')
        for ax in axes[row]:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle('nnscratch vs numpy / PyTorch / TensorFlow — same data, same initial '
                 'weights, same batch order\n(left and middle overlap exactly; the '
                 'right panel is where the differences live)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f'\nwrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--export-dir', default=os.path.join('output', 'reference'))
    p.add_argument('--model', choices=['mlp', 'cnn', 'both'], default='both')
    p.add_argument('--plot-path', default=os.path.join('reference', 'curves.png'),
                   help='where to write the overlay plot; the default is the copy '
                        'embedded in reference/README.md')
    args = p.parse_args()

    models = ['mlp', 'cnn'] if args.model == 'both' else [args.model]
    curves_by_model = {m: discover(args.export_dir, m) for m in models}
    for model in models:
        if curves_by_model[model]:
            table(model, curves_by_model[model])
        else:
            print(f'{model}: no curves found in {args.export_dir}')
    print(f'\n"identical" means the loss gap is within {FLOOR_ARTIFACT:g}, which is the\n'
          "log(p + 1e-9) floor in nnscratch's reported loss, and every prediction agrees.")
    plot(args.plot_path, models, curves_by_model)


if __name__ == '__main__':
    main()
