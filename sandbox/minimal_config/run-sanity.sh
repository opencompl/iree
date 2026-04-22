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
    --debug-only=iree-codegen-vector-layout-analysis \
    exp_reduce_base_attention_inst.mlir > exp_reduce_base_attention_inst.out 2>exp_reduce_base_attention_inst.err
