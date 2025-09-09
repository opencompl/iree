#!/bin/sh
export IREE_BUILD=../../iree-build
export PATH=$PATH:$IREE_BUILD/tools

export VMFB=test.vmfb

cmake --build $IREE_BUILD --target iree-compile iree-run-module
iree-compile --split-input-file --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=host \
    test-codegen.mlir --compile-from=flow \
    -o $VMFB &&\
# iree-run-module --module=$VMFB --input="4096x64xf32=-2" --input="4096x64xf32=1"  > $VMFB.out
iree-run-module --module=$VMFB --input="20x4096x64xf32=2" --input="20x4096x64xf32=1"  --input="20x4096x64xf32=3"  > $VMFB.out
# iree-benchmark-module --module=$VMFB --function="attention" --input="20x4096x64xf32" --input="20x4096x64xf32" --input="20x4096x64xf32" > $VMFB.out
