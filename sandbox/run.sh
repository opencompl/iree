#!/bin/sh
export SCRIPT_PATH=$(realpath $(dirname "$0"))
export IREE_BUILD=$(realpath $SCRIPT_PATH/../build)
cd $SCRIPT_PATH
cmake --build $IREE_BUILD
# $IREE_BUILD/tools/iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile{tiling-level=distribution},iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.exp_reduction}))" $SCRIPT_PATH/dispatch.mlir > out.mlir
$IREE_BUILD/tools/iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=host  --compile-from=flow --mlir-print-ir-after-all e2e.mlir 2>e2e.dbg.mlir 1>e2e.out.vmfb
