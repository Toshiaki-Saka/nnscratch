# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Toshiaki Saka
"""PyTorch reference: the same run, with autograd doing the backward pass.

Everything is held fixed except *who computes the gradients*. Same data, same
split, same initial weights, same mini-batch order -- so whatever gap remains
between this curve and nnscratch's is attributable to the framework, not to
luck.

    python reference/pytorch_reference.py --model both

Reads the export produced by ``export_reference`` and writes
``output/reference/<model>_curve_pytorch.csv``.
"""
from __future__ import annotations

import numpy as np
import torch

import common


def build(ref, dtype):
    """Rebuild the nnscratch network in PyTorch, weights and all.

    Two layout facts do the work here:

    * ``nn.Linear`` stores its weight as (out, in) and computes ``x @ W.T``,
      while nnscratch stores (in, out) and computes ``x @ W``. Hence the
      transpose.
    * ``nn.Conv2d`` stores (out_c, in_c, kH, kW) and cross-correlates without
      flipping the kernel -- exactly nnscratch's layout and convention, so the
      conv weight transfers unchanged. And because the tensor stays NCHW,
      ``nn.Flatten`` produces the same channel-major ordering the following
      Dense layer expects.
    """
    w = ref.weights
    if ref.cfg['model'] == 'mlp':
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 10))
        pairs = [(model[0], 'dense0'), (model[2], 'dense2'), (model[4], 'dense4')]
        with torch.no_grad():
            for layer, name in pairs:
                layer.weight.copy_(torch.as_tensor(w[f'{name}.W'].T))
                layer.bias.copy_(torch.as_tensor(w[f'{name}.b'].reshape(-1)))
    else:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0), torch.nn.ReLU(),
            torch.nn.Flatten(), torch.nn.Linear(8 * 6 * 6, 10))
        with torch.no_grad():
            model[0].weight.copy_(torch.as_tensor(w['conv0.W']))
            model[0].bias.copy_(torch.as_tensor(w['conv0.b'].reshape(-1)))
            model[3].weight.copy_(torch.as_tensor(w['dense3.W'].T))
            model[3].bias.copy_(torch.as_tensor(w['dense3.b'].reshape(-1)))
    return model.to(dtype)


def run(model_name, args):
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    ref = common.load_reference(model_name, args.export_dir, args.csv)
    cfg = ref.cfg

    net = build(ref, dtype)
    loss_fn = torch.nn.CrossEntropyLoss()          # takes logits, reduction='mean'
    opt = (torch.optim.SGD(net.parameters(), lr=cfg['lr'])
           if cfg['optimizer'] == 'sgd'
           else torch.optim.Adam(net.parameters(), lr=cfg['lr'],
                                 betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps']))

    x_tr = torch.as_tensor(ref.x_train, dtype=dtype)
    y_tr = torch.as_tensor(ref.y_train, dtype=torch.long)
    x_te = torch.as_tensor(ref.x_test, dtype=dtype)
    y_te = torch.as_tensor(ref.y_test, dtype=torch.long)

    def metrics():
        with torch.no_grad():
            logits = net(x_tr)
            loss = float(loss_fn(logits, y_tr))
            tr = float((logits.argmax(1) == y_tr).to(dtype).mean())
            te = float((net(x_te).argmax(1) == y_te).to(dtype).mean())
        return loss, tr, te

    rows = [(0, *metrics())]                       # epoch 0: before any update
    for ep in range(1, cfg['epochs'] + 1):
        for idx in common.batches(ref.batch_order[ep - 1], int(cfg['batch_size'])):
            batch = torch.as_tensor(idx, dtype=torch.long)
            # nnscratch assigns gradients rather than accumulating, so it needs
            # no zero_grad(); with autograd the accumulation is the default and
            # this call is mandatory.
            opt.zero_grad()
            loss_fn(net(x_tr[batch]), y_tr[batch]).backward()
            opt.step()
        rows.append((ep, *metrics()))

    common.write_curve(f'{args.out_dir}/{model_name}_curve_pytorch.csv', rows)
    common.report(f'pytorch {torch.__version__} ({args.dtype})', rows, ref)


if __name__ == '__main__':
    parsed = common.common_args(__doc__.splitlines()[0]).parse_args()
    torch.manual_seed(0)                           # nothing here is random, but be explicit
    for name in common.models_from(parsed):
        run(name, parsed)
