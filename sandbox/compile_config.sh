#!/bin/bash

INPUT_FILE="$(realpath "$1")"
BASE_NAME="${INPUT_FILE%.*}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath $SCRIPT_DIR/..)"
export PATH="$BASE_DIR/build/tools:$PATH"
cmake --build "$BASE_DIR/build" --target iree-compile
cd $SCRIPT_DIR

iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target=gfx942 \
    --mlir-print-ir-after-all \
    "$INPUT_FILE" > "${BASE_NAME}.out" 2>"${BASE_NAME}.err"
