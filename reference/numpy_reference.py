# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""numpy reference: the same network with the backward pass written out by hand.

This is the implementation nnscratch was ported *from*, reconstructed in numpy.
There is no autograd here either -- every ``backward`` below is the same
derivation the C++ code implements, so the two should agree to round-off.

    python reference/numpy_reference.py --model both

Reads the export produced by ``export_reference`` and writes
``output/reference/<model>_curve_numpy.csv``.
"""
from __future__ import annotations

import numpy as np

import common


# --------------------------------------------------------------------------
# layers -- mirrors include/nnscratch/layers.hpp and activations.hpp
# --------------------------------------------------------------------------

class Dense:
    """y = x . W + b, with W shaped (n_in, n_out) as in nnscratch."""

    def __init__(self, W, b):
        self.W, self.b = W, b.reshape(-1)          # nnscratch stores b as (1, n_out)
        self.dW = np.zeros_like(W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW = self.x.T @ g                     # X^T G
        self.db = g.sum(axis=0)                    # 1^T G -- a sum, not a mean:
        return g @ self.W.T                        # the 1/N already lives in the loss

    def params(self):
        return [(self.W, lambda: self.dW), (self.b, lambda: self.db)]


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0.0).astype(x.dtype)      # sub-gradient 0 at x == 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask

    def params(self):
        return []


class Flatten:
    """(N, C, H, W) -> (N, C*H*W), channel-major -- plain C-order reshape."""

    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, g):
        return g.reshape(self.shape)

    def params(self):
        return []


def im2col(x, k, stride, pad):
    """Rows are (n, i, j) batch-major; columns are (c, ky, kx) channel-major.

    Both orderings must match src/conv2d.cpp exactly or the gradients scatter
    into the wrong places. Unlike the C++ version this materialises a padded
    copy, which is equivalent but uses more memory.
    """
    n, c, h, w = x.shape
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    col = np.empty((n, c, k, k, oh, ow), dtype=x.dtype)
    for ky in range(k):
        for kx in range(k):
            col[:, :, ky, kx, :, :] = xp[:, :, ky:ky + stride * oh:stride,
                                         kx:kx + stride * ow:stride]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, c * k * k), oh, ow


def col2im(dcol, x_shape, k, stride, pad, oh, ow):
    """The transpose of im2col: += , never = , because the patches overlap."""
    n, c, h, w = x_shape
    d = dcol.reshape(n, oh, ow, c, k, k).transpose(0, 3, 4, 5, 1, 2)
    dxp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=dcol.dtype)
    for ky in range(k):
        for kx in range(k):
            dxp[:, :, ky:ky + stride * oh:stride,
                kx:kx + stride * ow:stride] += d[:, :, ky, kx, :, :]
    return dxp[:, :, pad:pad + h, pad:pad + w]


class Conv2D:
    """W is (out_c, in_c, k, k); cross-correlation, no kernel flip."""

    def __init__(self, W, b, stride, pad):
        self.W, self.b = W, b.reshape(-1)
        self.k = W.shape[2]
        self.stride, self.pad = stride, pad
        self.dW = np.zeros_like(W)
        self.db = np.zeros_like(self.b)
        self.col = self.x_shape = self.oh = self.ow = None

    def forward(self, x):
        self.x_shape = x.shape
        self.col, self.oh, self.ow = im2col(x, self.k, self.stride, self.pad)
        out_c = self.W.shape[0]
        wc = self.W.reshape(out_c, -1).T                 # (patch, out_c)
        out = self.col @ wc + self.b
        n = x.shape[0]
        return out.reshape(n, self.oh, self.ow, out_c).transpose(0, 3, 1, 2)

    def backward(self, g):
        out_c = self.W.shape[0]
        g_mat = g.transpose(0, 2, 3, 1).reshape(-1, out_c)
        self.db = g_mat.sum(axis=0)                      # shared across all positions
        self.dW = (g_mat.T @ self.col).reshape(self.W.shape)
        dcol = g_mat @ self.W.reshape(out_c, -1)
        return col2im(dcol, self.x_shape, self.k, self.stride, self.pad, self.oh, self.ow)

    def params(self):
        return [(self.W, lambda: self.dW), (self.b, lambda: self.db)]


# --------------------------------------------------------------------------
# loss and optimizers -- mirrors src/loss.cpp and src/optimizer.cpp
# --------------------------------------------------------------------------

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))         # stability; a no-op in exact math
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    def __init__(self):
        self.p = self.y = None

    def forward(self, logits, onehot):
        self.p, self.y = softmax(logits), onehot
        # the 1e-9 floor matches src/loss.cpp: it changes the reported loss only
        return float(-(onehot * np.log(self.p + 1e-9)).sum() / logits.shape[0])

    def backward(self):
        return (self.p - self.y) / self.p.shape[0]       # the whole fused gradient


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params):
        for p, grad in params:
            p -= self.lr * grad()


class Adam:
    """Adam, in either of the two placements of epsilon found in the wild.

    ``form='nnscratch'`` -- epsilon after bias correction, which is what
    nnscratch and PyTorch both do::

        p -= lr * m_hat / (sqrt(v_hat) + eps)

    ``form='keras'`` -- epsilon against the un-corrected second moment, the
    "epsilon hat" of the Kingma-Ba paper, which is what TensorFlow/Keras does::

        alpha = lr * sqrt(1 - b2**t) / (1 - b1**t)
        p -= alpha * m / (sqrt(v) + eps)

    The forms agree in exact arithmetic as eps -> 0; at eps = 1e-8 they differ
    by roughly 1e-8 per step, which is enough to be visible after a thousand
    steps. Running both is how ``reference/README.md`` attributes the
    TensorFlow CNN divergence.
    """

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, form='nnscratch'):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.form = form
        self.t = 0
        self.state = {}

    def step(self, params):
        self.t += 1                                      # once per step, not per tensor
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        alpha = self.lr * np.sqrt(bc2) / bc1
        for p, grad in params:
            st = self.state.setdefault(id(p), [np.zeros_like(p), np.zeros_like(p)])
            m, v = st
            g = grad()
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            if self.form == 'keras':
                p -= alpha * m / (np.sqrt(v) + self.eps)
            else:
                p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build(ref, dtype):
    w = {k: v.astype(dtype) for k, v in ref.weights.items()}
    if ref.cfg['model'] == 'mlp':
        return [Dense(w['dense0.W'], w['dense0.b']), ReLU(),
                Dense(w['dense2.W'], w['dense2.b']), ReLU(),
                Dense(w['dense4.W'], w['dense4.b'])]
    return [Conv2D(w['conv0.W'], w['conv0.b'], stride=1, pad=0), ReLU(), Flatten(),
            Dense(w['dense3.W'], w['dense3.b'])]


def forward(layers, x):
    for layer in layers:
        x = layer.forward(x)
    return x


def backward(layers, g):
    for layer in reversed(layers):
        g = layer.backward(g)


def run(model, args):
    dtype = np.dtype(args.dtype)
    ref = common.load_reference(model, args.export_dir, args.csv)
    cfg = ref.cfg
    layers = build(ref, dtype)
    params = [pg for layer in layers for pg in layer.params()]
    loss_fn = SoftmaxCrossEntropy()
    opt = (SGD(cfg['lr']) if cfg['optimizer'] == 'sgd'
           else Adam(cfg['lr'], cfg['beta1'], cfg['beta2'], cfg['eps'],
                     form=args.adam_form))

    k = int(cfg['num_classes'])
    x_tr, x_te = ref.x_train.astype(dtype), ref.x_test.astype(dtype)
    y_tr_1h = np.eye(k, dtype=dtype)[ref.y_train]

    def metrics():
        logits = forward(layers, x_tr)
        loss = loss_fn.forward(logits, y_tr_1h)
        tr = float((logits.argmax(1) == ref.y_train).mean())
        te = float((forward(layers, x_te).argmax(1) == ref.y_test).mean())
        return loss, tr, te

    rows = [(0, *metrics())]                             # epoch 0: before any update
    for ep in range(1, cfg['epochs'] + 1):
        for idx in common.batches(ref.batch_order[ep - 1], int(cfg['batch_size'])):
            logits = forward(layers, x_tr[idx])
            loss_fn.forward(logits, y_tr_1h[idx])
            backward(layers, loss_fn.backward())
            opt.step(params)
        rows.append((ep, *metrics()))

    suffix = '' if args.adam_form == 'nnscratch' else f'_adam-{args.adam_form}'
    common.write_curve(f'{args.out_dir}/{model}_curve_numpy{suffix}.csv', rows)
    common.report(f'numpy ({args.dtype}, adam={args.adam_form})', rows, ref)


if __name__ == '__main__':
    parser = common.common_args(__doc__.splitlines()[0])
    parser.add_argument('--adam-form', choices=['nnscratch', 'keras'], default='nnscratch',
                        help="where epsilon enters Adam; 'keras' reproduces "
                             'TensorFlow and is only meaningful for the CNN')
    parsed = parser.parse_args()
    for name in common.models_from(parsed):
        run(name, parsed)
