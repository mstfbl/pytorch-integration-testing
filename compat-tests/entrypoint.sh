#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -eou pipefail

(
    set -x
    # Skip certain unit tests that depend on the existance of a sister `build` folder to the `test` folder. As we're
    # not building PyTorch from source, the `build` folder doesn't exist.
    python -u run_test.py --continue-through-error -v -x test_binary_ufuncs test_jit test_linalg test_quantization test_spectral_ops test_jit_profiling test_jit_legacy test_openmp distributed/rpc/test_tensorpipe_agent test_fx
)

(
    set -x
    conda install -y tensorflow scipy
    python -u -c "import torch; import tensorflow"
    python -u -c "import tensorflow; import torch"
    python -u -c "import scipy; import torch"
    python -u -c "import torch; import scipy"
    python -u -c "import torch as th; x = th.autograd.Variable(th.rand(1, 3, 2, 2)); l = th.nn.Upsample(2); print(l(x))"
)
