# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""Watch the network learn, live, in a window.

    python reference/gui_demo.py

The demos in apps/ print numbers and write files; this shows the same training
happening. Mini-batches run inside the Tk event loop, so the curves, the
first-layer weights and the sample predictions all update as the network learns
rather than after it has finished.

The network is the numpy reference in this directory -- the same hand-derived
backward passes as the C++ library, agreeing with it to 2e-15 (see README.md).
So what you are watching is nnscratch's arithmetic, driven from Python because
that is where a window is cheap.

Needs only numpy, matplotlib and tkinter (tkinter ships with Python). Reads
data/digits.csv directly; nothing needs to be exported first.
"""
from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg               # noqa: E402
from matplotlib.figure import Figure                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                                 # noqa: E402
import numpy_reference as nr                                                  # noqa: E402

# Series colours: slots 1-3 of the validated categorical palette, and the
# matching dark-surface steps. Chosen for CVD separation, not by eye.
LIGHT = {'surface': '#fcfcfb', 'plane': '#f9f9f7', 'ink': '#0b0b0b', 'ink2': '#52514e',
         'muted': '#898781', 'grid': '#e1e0d9', 'axis': '#c3c2b7',
         's1': '#2a78d6', 's2': '#eb6834', 's3': '#1baf7a'}
ARCHS = ['deep MLP (64-64-32-10)', 'shallow (64-10)', 'CNN (conv 3x3 -> dense)']
OPTS = ['Adam (lr 0.01)', 'Momentum (lr 0.05)', 'SGD (lr 0.3)']


def he(rng, shape, fan_in):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)


def xavier(rng, shape, fan_in, fan_out):
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), shape)


def build_layers(arch, rng):
    """The three architectures of apps/compare.cpp, with nnscratch's init."""
    if arch.startswith('shallow'):
        return [nr.Dense(xavier(rng, (64, 10), 64, 10), np.zeros(10))]
    if arch.startswith('CNN'):
        return [nr.Conv2D(he(rng, (8, 1, 3, 3), 9), np.zeros(8), stride=1, pad=0),
                nr.ReLU(), nr.Flatten(),
                nr.Dense(xavier(rng, (288, 10), 288, 10), np.zeros(10))]
    return [nr.Dense(he(rng, (64, 64), 64), np.zeros(64)), nr.ReLU(),
            nr.Dense(he(rng, (64, 32), 64), np.zeros(32)), nr.ReLU(),
            nr.Dense(he(rng, (32, 10), 32), np.zeros(10))]


def build_opt(name):
    """Learning rates matched to each rule's effective step, as in compare.cpp."""
    if name.startswith('Adam'):
        return nr.Adam(0.01)
    if name.startswith('Momentum'):
        return nr.Momentum(0.05, 0.9)
    return nr.SGD(0.3)


class Demo:
    def __init__(self, root, csv, seed, batch):
        self.root, self.csv, self.seed, self.batch = root, csv, seed, batch
        self.running = False
        self._load_data()
        self._build_ui()
        self.reset()

    # -- data ---------------------------------------------------------------
    def _load_data(self):
        pixels, labels = common.load_digits_csv(self.csv)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(labels))
        cut = int(len(labels) * 0.8)
        tr, te = perm[:cut], perm[cut:]
        self.x_tr, self.y_tr = pixels[tr], labels[tr]
        self.x_te, self.y_te = pixels[te], labels[te]
        self.img_tr = self.x_tr.reshape(-1, 1, 8, 8)
        self.img_te = self.x_te.reshape(-1, 1, 8, 8)
        self.y1h = np.eye(10)[self.y_tr]
        # A fixed handful of test digits to show predictions for.
        self.sample = np.arange(10) * (len(self.y_te) // 10)

    # -- ui -----------------------------------------------------------------
    def _build_ui(self):
        self.root.title('nnscratch — watch it learn')
        self.root.configure(bg=LIGHT['plane'])

        bar = tk.Frame(self.root, bg=LIGHT['plane'], padx=12, pady=10)
        bar.pack(fill='x')

        self.arch = tk.StringVar(value=ARCHS[0])
        self.optn = tk.StringVar(value=OPTS[2])
        for label, var, values, width in (('Architecture', self.arch, ARCHS, 22),
                                          ('Optimizer', self.optn, OPTS, 18)):
            tk.Label(bar, text=label, bg=LIGHT['plane'], fg=LIGHT['muted'],
                     font=('Segoe UI', 9)).pack(side='left', padx=(0, 6))
            box = ttk.Combobox(bar, textvariable=var, values=values, width=width,
                               state='readonly', takefocus=False)
            box.pack(side='left', padx=(0, 16))
            box.bind('<<ComboboxSelected>>', lambda _e: self.reset())
            # A focused readonly Combobox changes value on scroll and on the arrow
            # keys. Over a window you are scrolling anyway, that silently swaps the
            # experiment out from under the run -- swallow both.
            box.bind('<MouseWheel>', lambda _e: 'break')
            box.bind('<Up>', lambda _e: 'break')
            box.bind('<Down>', lambda _e: 'break')

        self.btn = tk.Button(bar, text='▶  Train', command=self.toggle, width=10,
                             relief='flat', bg=LIGHT['s1'], fg='white',
                             font=('Segoe UI', 10, 'bold'), cursor='hand2')
        self.btn.pack(side='left')
        tk.Button(bar, text='Reset', command=self.reset, relief='flat',
                  bg=LIGHT['surface'], fg=LIGHT['ink2'], cursor='hand2',
                  font=('Segoe UI', 10)).pack(side='left', padx=8)

        self.status = tk.Label(bar, text='', bg=LIGHT['plane'], fg=LIGHT['ink'],
                               font=('Consolas', 10))
        self.status.pack(side='right')

        self.fig = Figure(figsize=(11.5, 6.4), dpi=100, facecolor=LIGHT['plane'])
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1],
                                   hspace=0.42, wspace=0.22,
                                   left=0.07, right=0.98, top=0.93, bottom=0.08)
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_acc = self.fig.add_subplot(gs[1, 0])
        self.ax_w = self.fig.add_subplot(gs[0, 1])
        self.ax_p = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax_loss, self.ax_acc):
            ax.set_facecolor(LIGHT['surface'])
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(LIGHT['axis'])
            ax.tick_params(colors=LIGHT['muted'], labelsize=8)
            ax.grid(True, color=LIGHT['grid'], linewidth=1, alpha=1)
            ax.set_axisbelow(True)
        for ax in (self.ax_w, self.ax_p):
            ax.set_facecolor(LIGHT['surface'])
            ax.set_xticks([])
            ax.set_yticks([])
            # imshow keeps its aspect, so without this the image floats in the
            # middle of the cell and drifts away from its own title.
            ax.set_anchor('N')
            for s in ax.spines.values():
                s.set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=12, pady=(0, 12))

    # -- training -----------------------------------------------------------
    def reset(self):
        self.running = False
        self.btn.config(text='▶  Train', bg=LIGHT['s1'])
        rng = np.random.default_rng(self.seed)
        self.layers = build_layers(self.arch.get(), rng)
        self.params = [pg for layer in self.layers for pg in layer.params()]
        self.opt = build_opt(self.optn.get())
        self.loss_fn = nr.SoftmaxCrossEntropy()
        self.order_rng = np.random.default_rng(self.seed + 1)
        self.epoch = 0
        self.hist = {'loss': [], 'train': [], 'test': []}
        self._metrics()
        self.draw()

    def toggle(self):
        self.running = not self.running
        self.btn.config(text='❚❚  Pause' if self.running else '▶  Train',
                        bg=LIGHT['s2'] if self.running else LIGHT['s1'])
        if self.running:
            self.root.after(10, self.step)

    def _x(self, train=True):
        cnn = self.arch.get().startswith('CNN')
        if train:
            return self.img_tr if cnn else self.x_tr
        return self.img_te if cnn else self.x_te

    def _metrics(self):
        logits = nr.forward(self.layers, self._x(True))
        self.hist['loss'].append(self.loss_fn.forward(logits, self.y1h))
        self.hist['train'].append(float((logits.argmax(1) == self.y_tr).mean()))
        te = nr.forward(self.layers, self._x(False))
        self.hist['test'].append(float((te.argmax(1) == self.y_te).mean()))
        self.pred = te.argmax(1)

    def train_one_epoch(self):
        x = self._x(True)
        for idx in common.batches(self.order_rng.permutation(len(self.y_tr)), self.batch):
            logits = nr.forward(self.layers, x[idx])
            self.loss_fn.forward(logits, self.y1h[idx])
            nr.backward(self.layers, self.loss_fn.backward())
            self.opt.step(self.params)
        self.epoch += 1
        self._metrics()

    def step(self):
        """One epoch per tick, then hand control back so the window stays alive."""
        if not self.running:
            return
        self.train_one_epoch()
        self.draw()
        if self.epoch >= 60:
            self.running = False
            self.btn.config(text='▶  Train', bg=LIGHT['s1'])
            return
        self.root.after(1, self.step)

    # -- drawing ------------------------------------------------------------
    def _weight_tiles(self):
        """First-layer weights as square images: 8x8 per unit, or the conv kernels."""
        first = self.layers[0]
        if isinstance(first, nr.Conv2D):
            w = first.W.reshape(first.W.shape[0], -1)
            side, cols = 3, 4
        else:
            w = first.W.T                       # (units, inputs)
            side = int(round(np.sqrt(w.shape[1])))
            cols = 8 if w.shape[0] >= 16 else 5
        n = min(w.shape[0], 64)
        rows = int(np.ceil(n / cols))
        pad = 1
        canvas = np.full((rows * (side + pad) + pad, cols * (side + pad) + pad), 0.5)
        for i in range(n):
            cell = w[i].reshape(side, side)
            lo, hi = cell.min(), cell.max()
            rng = hi - lo if hi - lo > 1e-12 else 1.0
            r, c = divmod(i, cols)
            y0, x0 = pad + r * (side + pad), pad + c * (side + pad)
            canvas[y0:y0 + side, x0:x0 + side] = (cell - lo) / rng
        return canvas

    def draw(self):
        ep = np.arange(len(self.hist['loss']))

        self.ax_loss.clear()
        self.ax_loss.plot(ep, self.hist['loss'], color=LIGHT['s1'], lw=2,
                          solid_capstyle='round')
        self.ax_loss.set_yscale('log')
        self.ax_loss.set_title('Training loss', color=LIGHT['ink'], fontsize=11,
                               loc='left', pad=8)
        self.ax_loss.set_xlabel('epoch', color=LIGHT['muted'], fontsize=9)

        self.ax_acc.clear()
        self.ax_acc.plot(ep, np.array(self.hist['train']) * 100, color=LIGHT['s1'], lw=2,
                         label='train', solid_capstyle='round')
        self.ax_acc.plot(ep, np.array(self.hist['test']) * 100, color=LIGHT['s2'], lw=2,
                         label='test', solid_capstyle='round')
        self.ax_acc.set_ylim(0, 103)
        self.ax_acc.set_title('Accuracy (%)', color=LIGHT['ink'], fontsize=11,
                              loc='left', pad=8)
        self.ax_acc.set_xlabel('epoch', color=LIGHT['muted'], fontsize=9)
        leg = self.ax_acc.legend(loc='lower right', frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(LIGHT['ink2'])

        for ax in (self.ax_loss, self.ax_acc):
            ax.set_facecolor(LIGHT['surface'])
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(LIGHT['axis'])
            ax.tick_params(colors=LIGHT['muted'], labelsize=8)
            ax.grid(True, color=LIGHT['grid'], linewidth=1)
            ax.set_axisbelow(True)

        self.ax_w.clear()
        self.ax_w.imshow(self._weight_tiles(), cmap='gray', vmin=0, vmax=1,
                         interpolation='nearest')
        self.ax_w.set_title('First-layer weights, updating live', color=LIGHT['ink'],
                            fontsize=11, loc='left', pad=8)
        self.ax_w.set_xticks([])
        self.ax_w.set_yticks([])
        self.ax_w.set_anchor('N')

        # Ten held-out digits with the current prediction; wrong ones flagged.
        self.ax_p.clear()
        strip = np.hstack([self.x_te[i].reshape(8, 8) for i in self.sample])
        self.ax_p.imshow(strip, cmap='gray_r', interpolation='nearest')
        for k, i in enumerate(self.sample):
            ok = self.pred[i] == self.y_te[i]
            # A wrong prediction is red *and* bold and shows the true label, so it
            # never depends on colour alone.
            self.ax_p.text(k * 8 + 3.5, 9.6,
                           str(self.pred[i]) if ok else f'{self.pred[i]}→{self.y_te[i]}',
                           ha='center', fontsize=11,
                           color=LIGHT['ink'] if ok else '#d03b3b',
                           fontweight='normal' if ok else 'bold')
        self.ax_p.set_ylim(12, -1)
        self.ax_p.set_title('Held-out digits and what it predicts', color=LIGHT['ink'],
                            fontsize=11, loc='left', pad=8)
        self.ax_p.set_xticks([])
        self.ax_p.set_yticks([])
        self.ax_p.set_anchor('N')

        self.status.config(text=f"epoch {self.epoch:3d}   loss {self.hist['loss'][-1]:.4f}"
                                f"   test {self.hist['test'][-1] * 100:5.1f}%")
        self.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--csv', default=common.DEFAULT_CSV)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--autostart', action='store_true',
                   help='begin training as soon as the window opens')
    args = p.parse_args()

    root = tk.Tk()
    demo = Demo(root, args.csv, args.seed, args.batch)
    if args.autostart:
        root.after(300, demo.toggle)
    root.mainloop()


if __name__ == '__main__':
    main()
