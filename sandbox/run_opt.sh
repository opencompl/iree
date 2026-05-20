#!/bin/bash

INPUT_FILE="$(realpath "$1")"
BASE_NAME="${INPUT_FILE%.*}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath $SCRIPT_DIR/..)"
export PATH="$BASE_DIR/build/tools:$PATH"
cmake --build "$BASE_DIR/build" --target iree-opt
cd $SCRIPT_DIR

# python3 $BASE_DIR/third_party/llvm-project/llvm/utils/lit/lit.py -v \
#     "$INPUT_FILE" > "${BASE_NAME}.out" 2>"${BASE_NAME}.err"
iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.exp_reduction}), canonicalize, cse)" "$INPUT_FILE" > "${BASE_NAME}.out" 2>"${BASE_NAME}.err"
