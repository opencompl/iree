// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_FORMATTINGUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_FORMATTINGUTILS_H_

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

inline std::string formatAffineMaps(ArrayRef<AffineMap> maps) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "[";
  llvm::interleaveComma(maps, os, [&](AffineMap map) { os << map; });
  os << "]";
  return storage;
}

inline StringRef getIteratorTypeName(utils::IteratorType iteratorType) {
  if (iteratorType == utils::IteratorType::parallel) {
    return "parallel";
  }
  if (iteratorType == utils::IteratorType::reduction) {
    return "reduction";
  }
  return "unknown";
}

inline std::string
formatIteratorTypes(ArrayRef<utils::IteratorType> iteratorTypes) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "[";
  llvm::interleaveComma(iteratorTypes, os, [&](utils::IteratorType type) {
    os << getIteratorTypeName(type);
  });
  os << "]";
  return storage;
}

inline std::string formatVectorTypes(ArrayRef<VectorType> types) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "[";
  llvm::interleaveComma(types, os, [&](VectorType type) { os << type; });
  os << "]";
  return storage;
}

inline std::string formatType(Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << type;
  return storage;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_FORMATTINGUTILS_H_
