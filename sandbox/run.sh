#!/bin/sh
export SCRIPT_PATH=$(realpath $(dirname "$0"))
export IREE_BUILD=$(realpath $SCRIPT_PATH/../build)
cd $SCRIPT_PATH
cmake -GNinja -B $IREE_BUILD .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_ENABLE_LLD=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DIREE_HAL_DRIVER_CUDA=ON
cmake --build $IREE_BUILD
$IREE_BUILD/tools/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=host  --compile-from=flow --mlir-print-ir-after-all e2e.mlir 2>e2e.dbg.mlir 1>e2e.out.vmfb
# uv run run-e2e.py
