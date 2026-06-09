// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/ExpReducePartialReductionOptions.h"

#include "llvm/Support/CommandLine.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

static llvm::cl::opt<ExpReducePartialReductionMode>
    clExpReducePartialReduction(
        "iree-expreduce-partial-reduction",
        llvm::cl::desc("Select the ExpReduce partial reduction tactic."),
        llvm::cl::values(
            clEnumValN(ExpReducePartialReductionMode::On, "on",
                       "Use ExpReduce to combine partial reductions."),
            clEnumValN(ExpReducePartialReductionMode::Lse, "lse",
                       "Use add to combine partial reductions."),
            clEnumValN(ExpReducePartialReductionMode::Off, "off",
                       "Disable partial reductions.")),
        llvm::cl::init(ExpReducePartialReductionMode::On));

ExpReducePartialReductionMode getExpReducePartialReductionMode() {
  return clExpReducePartialReduction;
}

bool isExpReducePartialReductionEnabled() {
  return getExpReducePartialReductionMode() != ExpReducePartialReductionMode::Off;
}

bool useLseExpReducePartialReductionCombiner() {
  return getExpReducePartialReductionMode() == ExpReducePartialReductionMode::Lse;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
