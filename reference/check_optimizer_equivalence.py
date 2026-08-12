# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""Which optimizer update does each framework actually implement?

``docs_*/experiments.md`` claims the three implementations differ in small ways
inside Momentum and Adam. This script settles it by measurement instead of by
reading release notes: it drives one scalar parameter through each framework's
optimizer in float64 and compares the trajectory against closed-form candidates.

    python reference/check_optimizer_equivalence.py

An exaggerated epsilon is used so that candidates which differ only in where
epsilon enters are far apart; the last section then reports the gap at the
realistic default.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import tensorflow as tf                                               # noqa: E402

GRADS = [3.0, -1.0, 2.0, 0.5, -2.5, 1.25]
TOL = 1e-12


def f32(x):
    """Round a Python float through float32.

    Keras stores scalar optimizer hyperparameters (momentum, beta_1, beta_2,
    epsilon) as float32 constants even when ``keras.backend.floatx()`` is
    float64 -- the learning rate is a Variable and does keep float64. Rounding
    the same way is what turns "close" into "exact" below.
    """
    return float(np.float32(x))


# --------------------------------------------------------------------------
# closed-form candidates
# --------------------------------------------------------------------------

def momentum_nnscratch(lr, mu):
    """v <- mu*v - lr*g ; p <- p + v   (nnscratch, Keras)"""
    p = v = 0.0
    out = []
    for g in GRADS:
        v = mu * v - lr * g
        p += v
        out.append(p)
    return out


def momentum_pytorch(lr, mu):
    """buf <- mu*buf + g ; p <- p - lr*buf   (PyTorch; buf differs by a factor lr)"""
    p = buf = 0.0
    out = []
    for i, g in enumerate(GRADS):
        buf = g if i == 0 else mu * buf + g
        p -= lr * buf
        out.append(p)
    return out


def adam_after_correction(lr, b1, b2, eps):
    """p -= lr * m_hat / (sqrt(v_hat) + eps)   -- epsilon after bias correction."""
    p = m = v = 0.0
    out = []
    for t, g in enumerate(GRADS, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p -= lr * (m / (1 - b1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + eps)
        out.append(p)
    return out


def adam_before_correction(lr, b1, b2, eps):
    """alpha = lr*sqrt(1-b2^t)/(1-b1^t) ; p -= alpha * m / (sqrt(v) + eps).

    Algebraically the same as `adam_after_correction` with epsilon replaced by
    eps/sqrt(1-b2^t), i.e. the paper's "epsilon hat". The moment recurrences are
    written the way Keras writes them -- as increments -- because the rounding
    of `m += (g - m)*(1 - b1)` differs from `m = b1*m + (1 - b1)*g`.
    """
    p = m = v = 0.0
    out = []
    for t, g in enumerate(GRADS, start=1):
        m += (g - m) * (1 - b1)
        v += (g * g - v) * (1 - b2)
        alpha = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        p -= alpha * m / (math.sqrt(v) + eps)
        out.append(p)
    return out


def adam_inside_sqrt(lr, b1, b2, eps):
    """p -= lr * m_hat / sqrt(v_hat + eps) -- the form the docs used to attribute
    to PyTorch. Kept as a control: nothing should match it."""
    p = m = v = 0.0
    out = []
    for t, g in enumerate(GRADS, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p -= lr * (m / (1 - b1 ** t)) / math.sqrt(v / (1 - b2 ** t) + eps)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# the frameworks
# --------------------------------------------------------------------------

def run_torch(make_opt):
    w = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = make_opt([w])
    out = []
    for g in GRADS:
        w.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        out.append(w.item())
    return out


def run_keras(opt):
    v = tf.Variable([0.0], dtype=tf.float64)
    out = []
    for g in GRADS:
        opt.apply_gradients([(tf.constant([g], dtype=tf.float64), v)])
        out.append(float(v.numpy()[0]))
    return out


def compare(label, got, candidates):
    print(f'\n{label}')
    best = None
    for name, ref in candidates.items():
        err = max(abs(a - b) for a, b in zip(got, ref))
        mark = 'MATCH  ' if err <= TOL else '       '
        print(f'  {mark}{name:<28s} max |diff| = {err:.3e}')
        if best is None or err < best[1]:
            best = (name, err)
    if best[1] > TOL:
        print(f'  -> closest is {best[0]} at {best[1]:.3e}, but no exact match')
    return best


def main():
    # Match how the reference runs are configured; Keras's casting path depends
    # on it, so leaving it at the float32 default changes the answers below.
    tf.keras.backend.set_floatx('float64')
    lr, mu = 0.05, 0.9
    b1, b2, eps = 0.9, 0.999, 0.5          # eps exaggerated on purpose

    print('=' * 70)
    print(f' Momentum  (lr={lr}, mu={mu})')
    print('=' * 70)
    cands = {'nnscratch / Keras form': momentum_nnscratch(lr, mu),
             'PyTorch form': momentum_pytorch(lr, mu),
             'Keras form, mu via float32': momentum_nnscratch(lr, f32(mu))}
    compare(f'PyTorch {torch.__version__}',
            run_torch(lambda ps: torch.optim.SGD(ps, lr=lr, momentum=mu)), cands)
    compare(f'Keras {tf.__version__}',
            run_keras(tf.keras.optimizers.SGD(learning_rate=lr, momentum=mu)), cands)
    print('\n  The first two forms produce the same trajectory -- they differ only in the\n'
          '  scale of the stored velocity (PyTorch keeps v larger by 1/lr). Keras matches\n'
          '  only once mu is rounded through float32; see f32() above.')

    print()
    print('=' * 70)
    print(f' Adam  (lr={lr}, beta1={b1}, beta2={b2}, eps={eps} -- exaggerated)')
    print('=' * 70)
    cands = {'eps after bias correction': adam_after_correction(lr, b1, b2, eps),
             'eps before bias correction': adam_before_correction(lr, b1, b2, eps),
             'eps inside the sqrt': adam_inside_sqrt(lr, b1, b2, eps),
             'eps before, betas via float32':
                 adam_before_correction(lr, f32(b1), f32(b2), f32(eps))}
    compare(f'PyTorch {torch.__version__}',
            run_torch(lambda ps: torch.optim.Adam(ps, lr=lr, betas=(b1, b2), eps=eps)),
            cands)
    compare(f'Keras {tf.__version__}',
            run_keras(tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2,
                                               epsilon=eps)), cands)
    print('\n  PyTorch reproduces nnscratch exactly. Keras differs in two independent\n'
          '  ways -- structurally (epsilon before bias correction) and numerically\n'
          '  (float32 hyperparameters). Accounting for both takes the gap from ~9e-2\n'
          '  to ~4e-8; the remaining residual is arithmetic ordering inside Keras\n'
          '  (it accumulates the moments as increments) and is not chased further.')

    print()
    print('=' * 70)
    print(' How much does the epsilon placement matter at the default 1e-8?')
    print('=' * 70)
    for e in (1e-8, 1e-4, 1e-2):
        a = adam_after_correction(0.01, 0.9, 0.999, e)
        b = adam_before_correction(0.01, 0.9, 0.999, e)
        gap = max(abs(x - y) for x, y in zip(a, b))
        print(f'  eps = {e:<8g} max |after - before| over {len(GRADS)} steps = {gap:.3e}')
    print('\n  Small per step -- but the CNN run takes ~1100 steps, which is why the\n'
          '  TensorFlow curve separates from the others. See reference/README.md.')


if __name__ == '__main__':
    main()
