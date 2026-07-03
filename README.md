# flare

An experiment in rebuilding a deep learning engine from scratch in Rust.

The heart of this repository is **ferro** (`rust_backend/`): a from-scratch,
PyTorch-style tensor + reverse-mode autograd runtime written in Rust, grown
from a small CPU MVP into a multi-crate engine with a kernel dispatcher,
device-resident storage, an optimized CPU backend, a staged CUDA backend, and
Python bindings - every operator gradient-checked and cross-validated against
PyTorch numerically.

## What works today

- **`ferro-core`** - pure-Rust, zero-dependency tensor + autograd runtime:
  ~35 operators with gradients, broadcasting, strided views, dtypes
  (f32 compute; f64/i64 storage and index tensors), torch-style gradient
  accumulation semantics, and an engine hardened for deep graphs (100k+ op
  chains). Ops are one-file-per-operator so coverage grows in parallel.
- **Dispatcher** - named kernels behind a per-device `Backend` trait and
  registry, with device-resident storage: whole training loops (forward with
  broadcast bias-adds, loss reductions, backward, SGD) run with tensors living
  on a device - proven by a counting test backend showing zero per-step
  uploads and exactly two scalar downloads per step.
- **`ferro-fastcpu`** - register-blocked AVX2+FMA matmul backend (~6x the
  reference kernel) installed through the dispatcher from outside core.
- **`ferro-cuda`** - cuBLAS + NVRTC backend implementing the full device
  surface (broadcasting, reductions, transposed matmul, autograd kernels).
  Compiles and unit-tests without CUDA installed; end-to-end GPU tests are
  staged behind runtime detection and awaiting real hardware.
- **`ferro-py`** - PyO3 bindings: train MLPs and CNNs from Python entirely on
  the Rust engine; leak-free DLPack interop with numpy and torch (which is
  also how ferro is validated against torch, values and gradients alike).
- **`nn` / `optim`** - Linear, LayerNorm, activations, Sequential,
  cross-entropy (one-hot and integer-target), embedding, SGD with momentum,
  Adam.

Start here:

- [`rust_backend/README.md`](rust_backend/README.md) - quickstart, layout,
  how to add an operator
- [`rust_backend/docs/ARCHITECTURE.md`](rust_backend/docs/ARCHITECTURE.md) -
  tensor/autograd design
- [`rust_backend/docs/DISPATCHER.md`](rust_backend/docs/DISPATCHER.md) -
  the dispatcher, phase by phase
- [`rust_backend/docs/FUTURE.md`](rust_backend/docs/FUTURE.md) - the master
  plan from here to a world-class engine

## Quickstart

```
cd rust_backend
cargo test -p ferro-core          # tensor + autograd + per-op gradient checks
cargo run -p ferro-fastcpu --bin bench --release   # kernel dispatch speedup
```

Python (builds the bindings and trains a CNN on the Rust engine):

```
cd rust_backend/crates/ferro-py
python3 -m venv .venv && . .venv/bin/activate
pip install -q maturin && maturin develop --release
python ../../examples/train_cnn_py.py
```

## Provenance and licensing

This repository began as a fork of [pytorch/pytorch](https://github.com/pytorch/pytorch)
and still contains the PyTorch source tree, which remains under its original
license (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)); the original
PyTorch README is preserved as [`PYTORCH_README.md`](PYTORCH_README.md).
This project is not affiliated with or endorsed by the PyTorch project.

The `ferro` crates under `rust_backend/` are new, independent work (MIT
licensed, per their manifests). PyTorch is used in this repository's test
suites purely as a numerical reference oracle.
