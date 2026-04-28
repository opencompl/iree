// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUEXTRACTMMAELEMENTWISEINPUTSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct ScalingTruncPeel {
  arith::ScalingTruncFOp truncOp;
  BlockArgument sourceArg;
  Value scale;
  SmallPtrSet<Operation *, 4> scaleExpressionOps;
};

static bool
collectBlockLocalScalarExpressionOps(Value value, Block &body,
                                     SmallPtrSetImpl<Operation *> &ops) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != &body) {
    return true;
  }
  if (!def->hasTrait<OpTrait::OneResult>() || def->getNumRegions() != 0) {
    return false;
  }
  for (Value operand : def->getOperands()) {
    if (isa<BlockArgument>(operand)) {
      return false;
    }
    if (!collectBlockLocalScalarExpressionOps(operand, body, ops)) {
      return false;
    }
  }
  ops.insert(def);
  return true;
}

static FailureOr<Value> cloneScalarExpression(OpBuilder &builder, Value value,
                                              Block &body, IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(value)) {
    return mapped;
  }
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != &body) {
    return value;
  }
  if (!def->hasTrait<OpTrait::OneResult>() || def->getNumRegions() != 0) {
    return failure();
  }

  IRMapping operandsMapping;
  for (Value operand : def->getOperands()) {
    FailureOr<Value> clonedOperand =
        cloneScalarExpression(builder, operand, body, mapping);
    if (failed(clonedOperand)) {
      return failure();
    }
    operandsMapping.map(operand, *clonedOperand);
  }
  Operation *cloned = builder.clone(*def, operandsMapping);
  mapping.map(value, cloned->getResult(0));
  return cloned->getResult(0);
}

static llvm::SmallDenseSet<int64_t>
getIndexingMapDimsWithIteratorType(AffineMap map,
                                   ArrayRef<utils::IteratorType> iterators,
                                   utils::IteratorType iteratorType) {
  llvm::SmallDenseSet<int64_t> dims;
  if (!map.isProjectedPermutation()) {
    return dims;
  }
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      continue;
    }
    int64_t position = dimExpr.getPosition();
    if (iterators[position] == iteratorType) {
      dims.insert(position);
    }
  }
  return dims;
}

static llvm::SmallDenseSet<int64_t>
inferKBDimsForLhsScale(linalg::GenericOp genericOp) {
  constexpr unsigned rhsScaleOperandIndex = 2;
  if (genericOp.getNumDpsInputs() <= rhsScaleOperandIndex) {
    return {};
  }

  SmallVector<AffineMap> maps = genericOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iterators =
      genericOp.getIteratorTypesArray();
  llvm::SmallDenseSet<int64_t> lhsRed = getIndexingMapDimsWithIteratorType(
      maps[IREE::GPU::kScaledMMAOperandLhs], iterators,
      utils::IteratorType::reduction);
  llvm::SmallDenseSet<int64_t> rhsRed = getIndexingMapDimsWithIteratorType(
      maps[IREE::GPU::kScaledMMAOperandRhs], iterators,
      utils::IteratorType::reduction);
  llvm::SmallDenseSet<int64_t> rhsScaleRed = getIndexingMapDimsWithIteratorType(
      maps[rhsScaleOperandIndex], iterators, utils::IteratorType::reduction);

  llvm::SmallDenseSet<int64_t> commonDataRed = lhsRed;
  llvm::set_intersect(commonDataRed, rhsRed);
  llvm::SmallDenseSet<int64_t> kBDims = commonDataRed;
  llvm::set_subtract(kBDims, rhsScaleRed);
  if (!kBDims.empty()) {
    return kBDims;
  }

  // Some producers keep the RHS MX scale in the same high-level layout as the
  // data tensor. In that form the RHS scale still mentions the block dimension,
  // but the lhs scale synthesized from `scaling_truncf` should not.
  if (commonDataRed.size() > 1) {
    kBDims.insert(*llvm::max_element(commonDataRed));
  }
  return kBDims;
}

static AffineMap dropIteratorDims(AffineMap map,
                                  const llvm::SmallDenseSet<int64_t> &dims) {
  SmallVector<AffineExpr> results;
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dims.contains(dimExpr.getPosition())) {
      continue;
    }
    results.push_back(expr);
  }
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), results,
                        map.getContext());
}

static AffineMap getScaleAccessMapForInput(
    MLIRContext *context, AffineMap inputMap,
    const llvm::SmallDenseSet<int64_t> &droppedIteratorDims) {
  SmallVector<AffineExpr> results;
  for (auto [inputDim, expr] : llvm::enumerate(inputMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || droppedIteratorDims.contains(dimExpr.getPosition())) {
      continue;
    }
    results.push_back(getAffineDimExpr(inputDim, context));
  }
  return AffineMap::get(inputMap.getNumResults(), /*symbolCount=*/0, results,
                        context);
}

static bool mapContainsIteratorDim(AffineMap map, int64_t iteratorDim) {
  return llvm::any_of(map.getResults(), [&](AffineExpr expr) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    return dimExpr && dimExpr.getPosition() == iteratorDim;
  });
}

static FailureOr<unsigned> getOperandDimForIteratorDim(AffineMap map,
                                                       int64_t iteratorDim) {
  std::optional<unsigned> operandDim;
  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != iteratorDim) {
      continue;
    }
    if (operandDim) {
      return failure();
    }
    operandDim = i;
  }
  if (!operandDim) {
    return failure();
  }
  return *operandDim;
}

static FailureOr<AffineMap> splitMapIteratorDim(AffineMap map,
                                                int64_t splitIteratorDim) {
  SmallVector<AffineExpr> results;
  MLIRContext *context = map.getContext();
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      return failure();
    }
    int64_t position = dimExpr.getPosition();
    if (position == splitIteratorDim) {
      results.push_back(getAffineDimExpr(position, context));
      results.push_back(getAffineDimExpr(position + 1, context));
      continue;
    }
    results.push_back(
        getAffineDimExpr(position > splitIteratorDim ? position + 1 : position,
                         context));
  }
  return AffineMap::get(map.getNumDims() + 1, map.getNumSymbols(), results,
                        context);
}

static SmallVector<utils::IteratorType>
splitIteratorTypes(ArrayRef<utils::IteratorType> iteratorTypes,
                   int64_t splitIteratorDim) {
  SmallVector<utils::IteratorType> newIteratorTypes;
  newIteratorTypes.reserve(iteratorTypes.size() + 1);
  for (auto [i, iteratorType] : llvm::enumerate(iteratorTypes)) {
    newIteratorTypes.push_back(iteratorType);
    if (i == splitIteratorDim) {
      newIteratorTypes.push_back(iteratorType);
    }
  }
  return newIteratorTypes;
}

static FailureOr<Value>
expandTensorDim(RewriterBase &rewriter, Location loc, Value value,
                unsigned operandDim, int64_t firstSize, int64_t secondSize) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || ShapedType::isDynamic(type.getDimSize(operandDim)) ||
      type.getDimSize(operandDim) != firstSize * secondSize) {
    return failure();
  }

  SmallVector<int64_t> newShape;
  SmallVector<ReassociationIndices> reassociation;
  newShape.reserve(type.getRank() + 1);
  reassociation.reserve(type.getRank());
  int64_t newDim = 0;
  for (int64_t dim = 0, rank = type.getRank(); dim < rank; ++dim) {
    if (dim == operandDim) {
      newShape.push_back(firstSize);
      newShape.push_back(secondSize);
      reassociation.push_back({newDim++, newDim++});
      continue;
    }
    newShape.push_back(type.getDimSize(dim));
    reassociation.push_back({newDim++});
  }

  auto expandedType =
      RankedTensorType::get(newShape, type.getElementType(), type.getEncoding());
  return tensor::ExpandShapeOp::create(rewriter, loc, expandedType, value,
                                       reassociation)
      .getResult();
}

static std::optional<SmallVector<int64_t>> getI64Array(ArrayAttr arrayAttr) {
  if (!arrayAttr ||
      !llvm::all_of(arrayAttr.getValue(), llvm::IsaPred<IntegerAttr>)) {
    return std::nullopt;
  }
  return llvm::map_to_vector(arrayAttr.getValue(), [](Attribute attr) {
    return cast<IntegerAttr>(attr).getInt();
  });
}

static ArrayAttr splitI64Array(Builder &builder, ArrayAttr arrayAttr,
                               int64_t splitDim, int64_t firstSize,
                               int64_t secondSize) {
  std::optional<SmallVector<int64_t>> values = getI64Array(arrayAttr);
  if (!values || splitDim >= static_cast<int64_t>(values->size())) {
    return arrayAttr;
  }

  SmallVector<int64_t> newValues;
  newValues.reserve(values->size() + 1);
  for (auto [i, value] : llvm::enumerate(*values)) {
    if (i != splitDim) {
      newValues.push_back(value);
      continue;
    }
    if (value == firstSize * secondSize) {
      newValues.push_back(firstSize);
      newValues.push_back(secondSize);
    } else if (value == 0) {
      newValues.push_back(0);
      newValues.push_back(0);
    } else {
      newValues.push_back(value);
      newValues.push_back(1);
    }
  }
  return builder.getI64ArrayAttr(newValues);
}

static ArrayAttr splitSubgroupBasis(Builder &builder, ArrayAttr basisAttr,
                                    int64_t splitDim) {
  if (!basisAttr || basisAttr.size() != 2) {
    return basisAttr;
  }
  std::optional<SmallVector<int64_t>> counts =
      getI64Array(dyn_cast<ArrayAttr>(basisAttr[0]));
  std::optional<SmallVector<int64_t>> mapping =
      getI64Array(dyn_cast<ArrayAttr>(basisAttr[1]));
  if (!counts || !mapping || splitDim >= static_cast<int64_t>(counts->size())) {
    return basisAttr;
  }

  SmallVector<int64_t> newCounts;
  newCounts.reserve(counts->size() + 1);
  for (auto [i, count] : llvm::enumerate(*counts)) {
    if (i == splitDim) {
      newCounts.push_back(count);
      newCounts.push_back(1);
    } else {
      newCounts.push_back(count);
    }
  }

  SmallVector<int64_t> newMapping;
  newMapping.reserve(mapping->size() + 1);
  for (int64_t dim : *mapping) {
    if (dim == splitDim) {
      newMapping.push_back(dim);
      newMapping.push_back(dim + 1);
    } else {
      newMapping.push_back(dim > splitDim ? dim + 1 : dim);
    }
  }

  return builder.getArrayAttr(
      {builder.getI64ArrayAttr(newCounts), builder.getI64ArrayAttr(newMapping)});
}

static IREE::GPU::LoweringConfigAttr
splitReductionConfig(IREE::GPU::LoweringConfigAttr config,
                     int64_t splitIteratorDim, int64_t firstSize,
                     int64_t secondSize) {
  Builder builder(config.getContext());
  NamedAttrList attrs(config.getAttributes());
  for (StringRef attrName : {"workgroup", "partial_reduction", "reduction",
                             "serial", "thread", "subgroup", "lane"}) {
    if (auto arrayAttr = dyn_cast_if_present<ArrayAttr>(attrs.get(attrName))) {
      attrs.set(attrName, splitI64Array(builder, arrayAttr, splitIteratorDim,
                                        firstSize, secondSize));
    }
  }
  if (auto basisAttr =
          dyn_cast_if_present<ArrayAttr>(attrs.get("subgroup_basis"))) {
    attrs.set("subgroup_basis",
              splitSubgroupBasis(builder, basisAttr, splitIteratorDim));
  }
  return IREE::GPU::LoweringConfigAttr::get(
      config.getContext(), attrs.getDictionary(config.getContext()));
}

static std::optional<ScalingTruncPeel>
matchScalingTruncPeel(linalg::GenericOp genericOp) {
  if (!genericOp.hasPureTensorSemantics() || genericOp.getNumDpsInputs() < 3 ||
      genericOp.getNumDpsInits() != 1) {
    return std::nullopt;
  }

  Block &body = genericOp.getRegion().front();
  for (Operation &op : body.without_terminator()) {
    auto truncOp = dyn_cast<arith::ScalingTruncFOp>(&op);
    if (!truncOp) {
      continue;
    }

    auto sourceArg = dyn_cast<BlockArgument>(truncOp.getIn());
    if (!sourceArg || sourceArg.getOwner() != &body ||
        sourceArg.getArgNumber() >= genericOp.getNumDpsInputs()) {
      continue;
    }
    // This pass is intentionally conservative. The decomposed exp-reduction
    // PV generic uses the lhs value only for the truncation being peeled.
    if (!sourceArg.hasOneUse() || !truncOp->hasOneUse()) {
      continue;
    }

    auto extOp = dyn_cast<arith::ScalingExtFOp>(*truncOp->user_begin());
    if (!extOp || extOp.getScale() != truncOp.getScale()) {
      continue;
    }

    ScalingTruncPeel peel;
    peel.truncOp = truncOp;
    peel.sourceArg = sourceArg;
    peel.scale = truncOp.getScale();
    if (!collectBlockLocalScalarExpressionOps(peel.scale, body,
                                              peel.scaleExpressionOps)) {
      continue;
    }
    return peel;
  }

  return std::nullopt;
}

static FailureOr<linalg::GenericOp>
splitFlatScaledContractionReduction(RewriterBase &rewriter,
                                    linalg::GenericOp genericOp,
                                    ScalingTruncPeel peel) {
  constexpr unsigned lhsOperandIndex = IREE::GPU::kScaledMMAOperandLhs;
  constexpr unsigned rhsOperandIndex = IREE::GPU::kScaledMMAOperandRhs;
  constexpr unsigned rhsScaleOperandIndex = 2;
  if (peel.sourceArg.getArgNumber() != lhsOperandIndex ||
      genericOp.getNumDpsInputs() != 3 || genericOp.getNumDpsInits() != 1) {
    return failure();
  }

  auto loweringConfig =
      getLoweringConfig<IREE::GPU::LoweringConfigAttr>(genericOp);
  if (!loweringConfig) {
    return failure();
  }
  auto mmaKind = dyn_cast_if_present<IREE::GPU::ScaledMMAAttr>(
      IREE::GPU::getMmaKind(loweringConfig));
  if (!mmaKind) {
    return failure();
  }
  auto [intrinsicM, intrinsicN, intrinsicK, intrinsicKB] =
      mmaKind.getScaledMNKShape();
  (void)intrinsicM;
  (void)intrinsicN;

  SmallVector<AffineMap> oldMaps = genericOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> oldIteratorTypes =
      genericOp.getIteratorTypesArray();
  llvm::SmallDenseSet<int64_t> lhsRed = getIndexingMapDimsWithIteratorType(
      oldMaps[lhsOperandIndex], oldIteratorTypes,
      utils::IteratorType::reduction);
  llvm::SmallDenseSet<int64_t> rhsRed = getIndexingMapDimsWithIteratorType(
      oldMaps[rhsOperandIndex], oldIteratorTypes,
      utils::IteratorType::reduction);
  llvm::SmallDenseSet<int64_t> rhsScaleRed = getIndexingMapDimsWithIteratorType(
      oldMaps[rhsScaleOperandIndex], oldIteratorTypes,
      utils::IteratorType::reduction);

  llvm::set_intersect(lhsRed, rhsRed);
  llvm::set_intersect(lhsRed, rhsScaleRed);
  if (lhsRed.size() != 1) {
    return failure();
  }
  int64_t splitIteratorDim = *lhsRed.begin();

  SmallVector<Value> newInputs = genericOp.getDpsInputs();
  Location loc = genericOp.getLoc();
  rewriter.setInsertionPoint(genericOp);
  for (unsigned inputIndex = 0, e = genericOp.getNumDpsInputs();
       inputIndex < e; ++inputIndex) {
    if (!mapContainsIteratorDim(oldMaps[inputIndex], splitIteratorDim)) {
      continue;
    }
    FailureOr<unsigned> operandDim =
        getOperandDimForIteratorDim(oldMaps[inputIndex], splitIteratorDim);
    if (failed(operandDim)) {
      return failure();
    }
    FailureOr<Value> expandedInput =
        expandTensorDim(rewriter, loc, newInputs[inputIndex], *operandDim,
                        intrinsicK, intrinsicKB);
    if (failed(expandedInput)) {
      return failure();
    }
    newInputs[inputIndex] = *expandedInput;
  }

  SmallVector<AffineMap> newMaps;
  newMaps.reserve(oldMaps.size());
  for (AffineMap map : oldMaps) {
    FailureOr<AffineMap> newMap = splitMapIteratorDim(map, splitIteratorDim);
    if (failed(newMap)) {
      return failure();
    }
    newMaps.push_back(*newMap);
  }
  SmallVector<utils::IteratorType> newIteratorTypes =
      splitIteratorTypes(oldIteratorTypes, splitIteratorDim);

  auto newGenericOp = linalg::GenericOp::create(
      rewriter, loc, genericOp->getResultTypes(), newInputs,
      genericOp.getDpsInits(), newMaps, newIteratorTypes);
  setLoweringConfig(newGenericOp,
                    splitReductionConfig(loweringConfig, splitIteratorDim,
                                         intrinsicK, intrinsicKB));

  Block &oldBody = genericOp.getRegion().front();
  SmallVector<Type> argTypes =
      llvm::map_to_vector(oldBody.getArguments(), [](BlockArgument arg) {
        return arg.getType();
      });
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *newBody =
      rewriter.createBlock(&newGenericOp.getRegion(), {}, argTypes, argLocs);

  IRMapping mapping;
  for (auto [oldArg, newArg] :
       llvm::zip_equal(oldBody.getArguments(), newBody->getArguments())) {
    mapping.map(oldArg, newArg);
  }
  rewriter.setInsertionPointToStart(newBody);
  for (Operation &op : oldBody.without_terminator()) {
    rewriter.clone(op, mapping);
  }
  rewriter.clone(*oldBody.getTerminator(), mapping);

  rewriter.replaceOp(genericOp, newGenericOp.getResults());
  return newGenericOp;
}

static Value createScaleTensor(
    RewriterBase &rewriter, Location loc, Value input, AffineMap inputMap,
    const llvm::SmallDenseSet<int64_t> &droppedIteratorDims, Value scale,
    Block &sourceBody) {
  auto inputType = cast<RankedTensorType>(input.getType());
  Type scaleElementType = getElementTypeOrSelf(scale.getType());
  SmallVector<int64_t> scaleShape;
  SmallVector<OpFoldResult> scaleMixedSizes;
  auto inputMixedSizes = tensor::getMixedSizes(rewriter, loc, input);
  for (auto [inputDim, expr] : llvm::enumerate(inputMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || droppedIteratorDims.contains(dimExpr.getPosition())) {
      continue;
    }
    scaleShape.push_back(inputType.getDimSize(inputDim));
    scaleMixedSizes.push_back(inputMixedSizes[inputDim]);
  }
  auto scaleType = RankedTensorType::get(scaleShape, scaleElementType,
                                         inputType.getEncoding());
  Value empty =
      inputType.getEncoding()
          ? tensor::EmptyOp::create(rewriter, loc, scaleMixedSizes,
                                    scaleElementType,
                                    inputType.getEncoding())
          : tensor::EmptyOp::create(rewriter, loc, scaleMixedSizes,
                                    scaleElementType);
  AffineMap map = rewriter.getMultiDimIdentityMap(scaleType.getRank());
  SmallVector<utils::IteratorType> iteratorTypes(scaleType.getRank(),
                                                 utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{scaleType}, ValueRange{},
             ValueRange{empty}, ArrayRef<AffineMap>{map}, iteratorTypes,
             [&](OpBuilder &builder, Location nestedLoc, ValueRange args) {
               IRMapping mapping;
               FailureOr<Value> clonedScale =
                   cloneScalarExpression(builder, scale, sourceBody, mapping);
               if (failed(clonedScale)) {
                 return;
               }
               linalg::YieldOp::create(builder, nestedLoc, *clonedScale);
             })
      .getResult(0);
}

static Value createTruncatedInputTensor(RewriterBase &rewriter, Location loc,
                                        Value input, Value scaleTensor,
                                        AffineMap scaleAccessMap,
                                        arith::ScalingTruncFOp truncOp) {
  auto inputType = cast<RankedTensorType>(input.getType());
  Type resultElementType = getElementTypeOrSelf(truncOp.getOut().getType());
  auto resultType = RankedTensorType::get(
      inputType.getShape(), resultElementType, inputType.getEncoding());
  auto mixedSizes = tensor::getMixedSizes(rewriter, loc, input);
  Value empty =
      inputType.getEncoding()
          ? tensor::EmptyOp::create(rewriter, loc, mixedSizes,
                                    resultElementType, inputType.getEncoding())
          : tensor::EmptyOp::create(rewriter, loc, mixedSizes,
                                    resultElementType);
  AffineMap map = rewriter.getMultiDimIdentityMap(inputType.getRank());
  SmallVector<AffineMap> maps{map, scaleAccessMap, map};
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{resultType},
             ValueRange{input, scaleTensor}, ValueRange{empty}, maps,
             iteratorTypes,
             [&](OpBuilder &builder, Location nestedLoc, ValueRange args) {
               auto trunc = arith::ScalingTruncFOp::create(
                   builder, nestedLoc, resultElementType, args[0], args[1],
                   truncOp.getRoundingmodeAttr(), truncOp.getFastmathAttr());
               linalg::YieldOp::create(builder, nestedLoc, trunc.getResult());
             })
      .getResult(0);
}

static FailureOr<linalg::GenericOp>
peelScalingTrunc(RewriterBase &rewriter, linalg::GenericOp genericOp,
                 ScalingTruncPeel peel) {
  Location loc = genericOp.getLoc();
  unsigned sourceOperandIndex = peel.sourceArg.getArgNumber();
  if (sourceOperandIndex != IREE::GPU::kScaledMMAOperandLhs) {
    return failure();
  }

  SmallVector<Value> oldInputs = genericOp.getDpsInputs();
  Value sourceInput = oldInputs[sourceOperandIndex];
  SmallVector<AffineMap> oldMaps = genericOp.getIndexingMapsArray();
  llvm::SmallDenseSet<int64_t> droppedIteratorDims =
      inferKBDimsForLhsScale(genericOp);
  if (droppedIteratorDims.empty()) {
    return failure();
  }
  AffineMap lhsScaleMap = dropIteratorDims(
      oldMaps[IREE::GPU::kScaledMMAOperandLhs], droppedIteratorDims);
  AffineMap scaleAccessMap = getScaleAccessMapForInput(
      rewriter.getContext(), oldMaps[IREE::GPU::kScaledMMAOperandLhs],
      droppedIteratorDims);
  rewriter.setInsertionPoint(genericOp);
  Value scaleTensor =
      createScaleTensor(rewriter, loc, sourceInput,
                        oldMaps[IREE::GPU::kScaledMMAOperandLhs],
                        droppedIteratorDims, peel.scale,
                        genericOp.getRegion().front());
  Value truncatedInput = createTruncatedInputTensor(rewriter, loc, sourceInput,
                                                    scaleTensor, scaleAccessMap,
                                                    peel.truncOp);

  SmallVector<Value> newInputs;
  newInputs.reserve(oldInputs.size() + 1);
  newInputs.push_back(truncatedInput);
  newInputs.push_back(oldInputs[IREE::GPU::kScaledMMAOperandRhs]);
  newInputs.push_back(scaleTensor);
  constexpr unsigned oldFirstInputAfterRhs = 2;
  for (unsigned i = oldFirstInputAfterRhs; i < oldInputs.size(); ++i) {
    newInputs.push_back(oldInputs[i]);
  }

  SmallVector<Value> newOperands(newInputs);
  llvm::append_range(newOperands, genericOp.getDpsInits());

  SmallVector<AffineMap> newMaps;
  newMaps.reserve(oldMaps.size() + 1);
  newMaps.push_back(oldMaps[IREE::GPU::kScaledMMAOperandLhs]);
  newMaps.push_back(oldMaps[IREE::GPU::kScaledMMAOperandRhs]);
  newMaps.push_back(lhsScaleMap);
  for (unsigned i = oldFirstInputAfterRhs; i < genericOp.getNumDpsInputs();
       ++i) {
    newMaps.push_back(oldMaps[i]);
  }
  for (unsigned i = genericOp.getNumDpsInputs(); i < oldMaps.size(); ++i) {
    newMaps.push_back(oldMaps[i]);
  }

  rewriter.setInsertionPoint(genericOp);
  auto newGenericOp = linalg::GenericOp::create(
      rewriter, loc, genericOp->getResultTypes(), newInputs,
      genericOp.getDpsInits(), newMaps, genericOp.getIteratorTypesArray());
  if (auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(genericOp)) {
    setLoweringConfig(newGenericOp, loweringConfig);
  }

  Block &oldBody = genericOp.getRegion().front();
  SmallVector<Type> newArgTypes;
  newArgTypes.reserve(oldBody.getNumArguments() + 1);
  newArgTypes.push_back(getElementTypeOrSelf(truncatedInput.getType()));
  newArgTypes.push_back(
      oldBody.getArgument(IREE::GPU::kScaledMMAOperandRhs).getType());
  newArgTypes.push_back(getElementTypeOrSelf(scaleTensor.getType()));
  for (unsigned i = oldFirstInputAfterRhs; i < genericOp.getNumDpsInputs();
       ++i) {
    newArgTypes.push_back(oldBody.getArgument(i).getType());
  }
  for (unsigned i = genericOp.getNumDpsInputs(); i < oldBody.getNumArguments();
       ++i) {
    newArgTypes.push_back(oldBody.getArgument(i).getType());
  }
  SmallVector<Location> argLocs(newArgTypes.size(), loc);
  Block *newBody =
      rewriter.createBlock(&newGenericOp.getRegion(), {}, newArgTypes, argLocs);

  IRMapping mapping;
  mapping.map(oldBody.getArgument(IREE::GPU::kScaledMMAOperandLhs),
              newBody->getArgument(IREE::GPU::kScaledMMAOperandLhs));
  mapping.map(oldBody.getArgument(IREE::GPU::kScaledMMAOperandRhs),
              newBody->getArgument(IREE::GPU::kScaledMMAOperandRhs));
  mapping.map(peel.scale,
              newBody->getArgument(IREE::GPU::kScaledMMAOperandLhsScale));
  for (unsigned i = oldFirstInputAfterRhs; i < genericOp.getNumDpsInputs();
       ++i) {
    mapping.map(oldBody.getArgument(i), newBody->getArgument(i + 1));
  }
  for (unsigned i = genericOp.getNumDpsInputs(); i < oldBody.getNumArguments();
       ++i) {
    mapping.map(oldBody.getArgument(i), newBody->getArgument(i + 1));
  }
  mapping.map(peel.truncOp.getResult(),
              newBody->getArgument(IREE::GPU::kScaledMMAOperandLhs));

  rewriter.setInsertionPointToStart(newBody);
  Operation *truncOperation = peel.truncOp.getOperation();
  for (Operation &op : oldBody.without_terminator()) {
    if (&op == truncOperation || peel.scaleExpressionOps.contains(&op)) {
      continue;
    }
    rewriter.clone(op, mapping);
  }
  rewriter.clone(*oldBody.getTerminator(), mapping);

  rewriter.replaceOp(genericOp, newGenericOp.getResults());
  return newGenericOp;
}

struct PeelScalingTruncFromMMA final : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    std::optional<ScalingTruncPeel> peel = matchScalingTruncPeel(genericOp);
    if (!peel) {
      return failure();
    }
    return peelScalingTrunc(rewriter, genericOp, *peel);
  }
};

struct SplitFlatScaledContractionReduction final
    : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    std::optional<ScalingTruncPeel> peel = matchScalingTruncPeel(genericOp);
    if (!peel) {
      return failure();
    }
    return splitFlatScaledContractionReduction(rewriter, genericOp, *peel);
  }
};

struct LLVMGPUExtractMMAElementwiseInputsPass final
    : impl::LLVMGPUExtractMMAElementwiseInputsPassBase<
          LLVMGPUExtractMMAElementwiseInputsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SplitFlatScaledContractionReduction, PeelScalingTruncFromMMA>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
