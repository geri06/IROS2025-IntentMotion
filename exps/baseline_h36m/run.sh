# Baseline 48
# The value :4096:8 specifies that the workspace should use a size configuration of 4096 bytes and 8 streams.
# This configuration can help with deterministic results when using GPU computations with cuBLAS, which
# can otherwise introduce non-deterministic behavior in some operations.
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name baseline.txt --layer-norm-axis spatial --with-normalization --num 48

