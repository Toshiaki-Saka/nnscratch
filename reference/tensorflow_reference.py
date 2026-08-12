# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""TensorFlow/Keras reference: the same run, with GradientTape doing the backward pass.

    python reference/tensorflow_reference.py --model both

Reads the export produced by ``export_reference`` and writes
``output/reference/<model>_curve_tensorflow.csv``.

Two things make this port less mechanical than the PyTorch one, and both are
layout rather than mathematics -- see ``build()``.
"""
from __future__ import annotations

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np                                                    # noqa: E402
import tensorflow as tf                                               # noqa: E402

import common                                                         # noqa: E402

keras = tf.keras


def build(ref, floatx):
    """Rebuild the nnscratch network in Keras, weights and all.

    Dense transfers unchanged: Keras stores its kernel as (in, out) and
    computes ``x @ kernel``, exactly like nnscratch. The convolution does not:

    * Keras stores the kernel as (kH, kW, in_c, out_c) whereas nnscratch uses
      (out_c, in_c, k, k), so the axes are permuted on the way in. The
      convention is the same on both sides -- cross-correlation, no flip.
    * Keras convolves in NHWC, so the conv output is (N, oh, ow, out_c) and a
      plain ``Flatten`` would produce *spatial*-major order, while nnscratch's
      Flatten is channel-major. Feeding that to the next Dense would silently
      permute its inputs and train a different network. A ``Permute`` back to
      channels-first before flattening fixes it without touching the weights.
    """
    w = ref.weights
    if ref.cfg['model'] == 'mlp':
        model = keras.Sequential([
            keras.Input(shape=(64,), dtype=floatx),
            keras.layers.Dense(64, dtype=floatx), keras.layers.ReLU(dtype=floatx),
            keras.layers.Dense(32, dtype=floatx), keras.layers.ReLU(dtype=floatx),
            keras.layers.Dense(10, dtype=floatx)])
        for layer, name in ((model.layers[0], 'dense0'), (model.layers[2], 'dense2'),
                            (model.layers[4], 'dense4')):
            layer.set_weights([w[f'{name}.W'], w[f'{name}.b'].reshape(-1)])
    else:
        model = keras.Sequential([
            keras.Input(shape=(8, 8, 1), dtype=floatx),
            keras.layers.Conv2D(8, 3, strides=1, padding='valid', dtype=floatx),
            keras.layers.ReLU(dtype=floatx),
            keras.layers.Permute((3, 1, 2), dtype=floatx),   # NHWC -> NCHW ordering
            keras.layers.Flatten(dtype=floatx),              # ... so this is channel-major
            keras.layers.Dense(10, dtype=floatx)])
        # (out_c, in_c, kH, kW) -> (kH, kW, in_c, out_c)
        model.layers[0].set_weights([np.transpose(w['conv0.W'], (2, 3, 1, 0)),
                                     w['conv0.b'].reshape(-1)])
        model.layers[4].set_weights([w['dense3.W'], w['dense3.b'].reshape(-1)])
    return model


def run(model_name, args):
    keras.backend.set_floatx(args.dtype)
    floatx = args.dtype
    ref = common.load_reference(model_name, args.export_dir, args.csv)
    cfg = ref.cfg

    x_tr, x_te = ref.x_train, ref.x_test
    if cfg['input'] == 'img':                       # (N, 1, 8, 8) -> (N, 8, 8, 1)
        x_tr = np.transpose(x_tr, (0, 2, 3, 1))
        x_te = np.transpose(x_te, (0, 2, 3, 1))
    x_tr = tf.constant(x_tr, dtype=floatx)
    x_te = tf.constant(x_te, dtype=floatx)
    y_tr = tf.constant(ref.y_train, dtype=tf.int64)
    y_te = tf.constant(ref.y_test, dtype=tf.int64)

    net = build(ref, floatx)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = (keras.optimizers.SGD(learning_rate=cfg['lr'])
           if cfg['optimizer'] == 'sgd'
           else keras.optimizers.Adam(learning_rate=cfg['lr'], beta_1=cfg['beta1'],
                                      beta_2=cfg['beta2'], epsilon=cfg['eps']))

    def metrics():
        logits = net(x_tr, training=False)
        loss = float(loss_fn(y_tr, logits))
        tr = float(tf.reduce_mean(tf.cast(
            tf.argmax(logits, axis=1) == y_tr, floatx)))
        te = float(tf.reduce_mean(tf.cast(
            tf.argmax(net(x_te, training=False), axis=1) == y_te, floatx)))
        return loss, tr, te

    rows = [(0, *metrics())]                        # epoch 0: before any update
    for ep in range(1, cfg['epochs'] + 1):
        for idx in common.batches(ref.batch_order[ep - 1], int(cfg['batch_size'])):
            xb = tf.gather(x_tr, idx)
            yb = tf.gather(y_tr, idx)
            with tf.GradientTape() as tape:         # the tape *is* the graph here
                loss = loss_fn(yb, net(xb, training=True))
            opt.apply_gradients(zip(tape.gradient(loss, net.trainable_variables),
                                    net.trainable_variables))
        rows.append((ep, *metrics()))

    common.write_curve(f'{args.out_dir}/{model_name}_curve_tensorflow.csv', rows)
    common.report(f'tensorflow {tf.__version__} ({args.dtype})', rows, ref)


if __name__ == '__main__':
    parsed = common.common_args(__doc__.splitlines()[0]).parse_args()
    tf.random.set_seed(0)                           # nothing here is random, but be explicit
    for name in common.models_from(parsed):
        run(name, parsed)
