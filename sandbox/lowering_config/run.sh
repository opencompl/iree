#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath $SCRIPT_DIR/../..)"
export PATH="$BASE_DIR/../iree-build/tools:$PATH"
cmake --build "$BASE_DIR/build" --target iree-compile
cd $SCRIPT_DIR
iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target=gfx942 \
    --compile-to=executable-configurations \
    attention.mlir > attention.out 2>attention.err
