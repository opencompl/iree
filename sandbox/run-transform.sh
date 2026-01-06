#!/bin/sh
export SCRIPT_PATH=$(realpath $(dirname "$0"))
export IREE_BUILD=$(realpath $SCRIPT_PATH/../build)
cd $SCRIPT_PATH
if [ ! -f .cmade ]; then
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
    touch .cmade
fi
cmake --build $IREE_BUILD
$IREE_BUILD/tools/iree-opt --iree-transform-dialect-interpreter --split-input-file --verify-diagnostics -canonicalize -cse tile-reduction.mlir 1> tile-reduction.out.mlir
