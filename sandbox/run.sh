#!/bin/sh
export SCRIPT_PATH=$(dirname "$0")
export IREE_BUILD=$SCRIPT_PATH/../build
cmake --build $IREE_BUILD
$IREE_BUILD/tools/iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile{tiling-level=distribution},iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.exp_reduction}))" $SCRIPT_PATH/dispatch.mlir
