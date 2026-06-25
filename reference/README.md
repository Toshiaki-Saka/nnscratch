# Reference notes

nnscratch is a faithful C++ port of a numpy teaching implementation of an
MLP/CNN. **The C++ library in this repository *is* the reference** — every
`forward()` / `backward()` is written out by hand so you can read exactly what
a framework's autograd would otherwise hide.

How each piece maps to the mainstream frameworks (numpy, PyTorch, TensorFlow)
is documented as a side-by-side table in the top-level
[`README.md`](../README.md#numpy--c--framework-correspondence) — for example,
what `Dense::backward` corresponds to in `torch.nn.Linear`'s autograd.

> Standalone numpy / PyTorch / TensorFlow reference *scripts* are **not bundled**
> in this release; the correspondence table above is the canonical cross-framework
> reference.
