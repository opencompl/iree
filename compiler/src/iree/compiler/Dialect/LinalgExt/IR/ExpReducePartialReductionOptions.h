// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_EXPREDUCEPARTIALREDUCTIONOPTIONS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_EXPREDUCEPARTIALREDUCTIONOPTIONS_H_

namespace mlir::iree_compiler::IREE::LinalgExt {

enum class ExpReducePartialReductionMode {
  On,
  Lse,
  Off,
};

ExpReducePartialReductionMode getExpReducePartialReductionMode();

bool isExpReducePartialReductionEnabled();

bool useLseExpReducePartialReductionCombiner();

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_EXPREDUCEPARTIALREDUCTIONOPTIONS_H_
