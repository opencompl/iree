#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath $SCRIPT_DIR/../..)"
export PATH="$BASE_DIR/../iree-build/tools:$PATH"
cmake --build "$BASE_DIR/build" --target iree-compile
cd $SCRIPT_DIR
iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target=gfx942 \
    --mlir-print-ir-after-all \
    exp_reduce_base_attention_fused.mlir > exp_reduce_base_attention_fused.out 2>exp_reduce_base_attention_fused.err

# --mlir-print-ir-after-all \
# --mlir-print-ir-after-failure

    # --mlir-print-ir-before=iree-llvmgpu-vector-distribute \
    # --mlir-print-ir-after=iree-llvmgpu-vector-distribute \
