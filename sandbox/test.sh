#!/bin/sh
export SCRIPT_PATH=$(dirname "$0")
export IREE_BUILD=$SCRIPT_PATH/../build
cd $IREE_BUILD
cmake --build .
ctest -j$(nproc)
