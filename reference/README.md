# Reference implementations

These are the original numpy / PyTorch / TensorFlow scripts that nnscratch was
ported from, kept here for provenance and side-by-side study. They are **not**
built or required by the C++ library; they only need a Python environment.

- `deeplearning_from_scratch.py` — numpy MLP, the basis for the `from_scratch` demo.
- `dl_compare.py` — numpy componentised framework, the basis for the `compare` demo.
- `pytorch_equivalent.py` / `tensorflow_equivalent.py` — the same experiments in
  each framework, showing what `backward()` and the optimizers map to.

```bash
pip install numpy matplotlib scikit-learn
python deeplearning_from_scratch.py
```
