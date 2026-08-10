// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches, cse))" --split-input-file --mlir-print-local-scope %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches, iree-dispatch-creation-convert-dispatch-regions-to-workgroups, iree-dispatch-creation-materialize-default-workgroup-count-region, canonicalize))" --split-input-file --mlir-print-local-scope %s | FileCheck %s --check-prefix=WORKGROUP
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches{enable-fuse-pad}, cse))" --split-input-file --mlir-print-local-scope %s | FileCheck %s --check-prefix=FUSE-PAD

util.func public @split_reduction_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"],
      iree_linalg_ext.split_reduction = [128]}
      ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}
// CHECK-LABEL:  @split_reduction_dynamic
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:    %[[EMPTY0:.+]] = tensor.empty(%{{.+}}) : tensor<?xf32>
//       CHECK:    %[[FILL0:.+]] = linalg.fill
//  CHECK-SAME:        outs(%[[EMPTY0]] :
//   CHECK-DAG:    %[[EMPTY1:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32>
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[FORALL:.+]] = scf.forall
//  CHECK-SAME:          shared_outs(%[[INIT:.+]] = %[[EMPTY1]]) -> (tensor<?x?xf32>) {
//       CHECK:        %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]]
//       CHECK:        %[[FILL1:.+]] = linalg.fill
//  CHECK-SAME:            outs(%[[INIT_SLICE]] :
//       CHECK:        %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:            outs(%[[FILL1]] :
//       CHECK:        tensor.parallel_insert_slice %[[GENERIC]] into %[[INIT]]
//       CHECK:      } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
//       CHECK:      flow.return %[[FORALL]] : tensor<?x?xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg.reduce
//  CHECK-SAME:        ins(%[[DISPATCH]] :
//  CHECK-SAME:        outs(%[[FILL0]] :
//  CHECK-SAME:        dimensions = [1]
//       CHECK:    util.return %[[REDUCE]]

// Check that count region contains splitk modifier.
// WORKGROUP-LABEL:   @split_reduction_dynamic(
//       WORKGROUP:     count(
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_from_slice
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier

// -----

util.func public @split_reduction_static(%arg0: tensor<64x4096xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"],
      iree_linalg_ext.split_reduction = [128]}
      ins(%arg0 : tensor<64x4096xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<64xf32>
  util.return %2 : tensor<64xf32>
}
// CHECK-LABEL:  @split_reduction_static(
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<64x4096xf32>
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[FORALL:.+]] = scf.forall
//       CHECK:        linalg.generic
//       CHECK:      flow.return %[[FORALL]] : tensor<64x32xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg.reduce
//  CHECK-SAME:        ins(%[[DISPATCH]] :
//  CHECK-SAME:        dimensions = [1]
//       CHECK:    util.return %[[REDUCE]]

// Check that count region contains splitk modifier.
// WORKGROUP-LABEL:   @split_reduction_static
//       WORKGROUP:     count(
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_from_slice()
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier

// -----

// Test multiple reduction dimensions.

util.func public @split_reduction_multiple_dims(%arg0: tensor<?x?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction"],
      iree_linalg_ext.split_reduction = [128, 16]}
      ins(%arg0 : tensor<?x?x?xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}
// CHECK-LABEL:  @split_reduction_multiple_dims(
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:    %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:    %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:    %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//   CHECK-DAG:    %[[PARTIAL_D1:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%[[D1]]]
//   CHECK-DAG:    %[[PARTIAL_D2:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%[[D2]]]
//       CHECK:    %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[PARTIAL_D1]], %[[PARTIAL_D2]]) : tensor<?x?x?xf32>
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[FORALL:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
//  CHECK-SAME:          (0, 0) to (%[[D1]], %[[D2]]) step (128, 16)
//  CHECK-SAME:          shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<?x?x?xf32>) {
//   CHECK-DAG:        %[[T1:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%[[IV0]])[%[[D1]]]
//   CHECK-DAG:        %[[T2:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 16)>(%[[IV1]])[%[[D2]]]
//       CHECK:        %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IV0]], %[[IV1]]] [%[[D0]], %[[T1]], %[[T2]]]
//   CHECK-DAG:        %[[OUT_OFFSET0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 128)>()[%[[IV0]]]
//   CHECK-DAG:        %[[OUT_OFFSET1:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%[[IV1]]]
//       CHECK:        %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][0, %[[OUT_OFFSET0]], %[[OUT_OFFSET1]]] [%[[D0]], 1, 1]
//       CHECK:        %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:            outs(%[[INIT_SLICE]] :
//       CHECK:        %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:            iterator_types = ["parallel", "reduction", "reduction"]
//  CHECK-SAME:            ins(%[[INPUT_SLICE]] :
//  CHECK-SAME:            outs(%[[FILL]] :
//       CHECK:        tensor.parallel_insert_slice %[[GENERIC]] into %[[INIT]]
//       CHECK:      } {mapping = [#iree_linalg_ext.split_reduction_mapping<1>, #iree_linalg_ext.split_reduction_mapping<0>]}
//       CHECK:      flow.return %[[FORALL]] : tensor<?x?x?xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg.reduce
//  CHECK-SAME:        ins(%[[DISPATCH]] :
//  CHECK-SAME:        dimensions = [1, 2]
//       CHECK:    util.return %[[REDUCE]]

// -----

util.func public @fuse_pad_with_conv(%arg0 : tensor<16x225x225x16xbf16>, %arg1 : tensor<16x225x225x64xbf16>) -> (tensor<64x3x3x16xf32>) {
  %cst = arith.constant 0.0 : bf16
  %cst_0 = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
    tensor.yield %cst : bf16
  } : tensor<16x225x225x16xbf16> to tensor<16x227x227x16xbf16>
  %2 = tensor.empty() : tensor<64x3x3x16xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64x3x3x16xf32>) -> tensor<64x3x3x16xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg1 : tensor<16x227x227x16xbf16>, tensor<16x225x225x64xbf16>) outs(%3 : tensor<64x3x3x16xf32>) attrs =  {iree_linalg_ext.split_reduction = [1, 255, 255]} {
  ^bb0(%in: bf16, %in_1: bf16, %out: f32):
    %9 = arith.extf %in : bf16 to f32
    %10 = arith.extf %in_1 : bf16 to f32
    %11 = arith.mulf %9, %10 : f32
    %12 = arith.addf %out, %11 : f32
    linalg.yield %12 : f32
  } -> tensor<64x3x3x16xf32>
  util.return %4 : tensor<64x3x3x16xf32>
}
// FUSE-PAD-LABEL:  @fuse_pad_with_conv(
//       FUSE-PAD:    %[[DISPATCH:.+]] = flow.dispatch.region
//       FUSE-PAD:      scf.forall
//   FUSE-PAD-NOT:        scf.if
//       FUSE-PAD:        tensor.extract_slice
//       FUSE-PAD:        tensor.pad
//       FUSE-PAD:        linalg.generic
//       FUSE-PAD:    linalg.reduce
//       FUSE-PAD:      ins(%[[DISPATCH]]

// Not enabled: pad should be outside of dispatch.
// CHECK-LABEL:  @fuse_pad_with_conv(
//       CHECK:    tensor.pad
//       CHECK:    flow.dispatch.region
//       CHECK:      linalg.generic

// -----

util.func public @split_reduction_arg_compare(%arg0: tensor<4x1x128256xf16>) -> (tensor<4x1xf16>, tensor<4x1xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0xFC00 : f16
  %0 = tensor.empty() : tensor<4x1xf16>
  %1 = tensor.empty() : tensor<4x1xi32>
  %2 = linalg.fill ins(%cst : f16) outs(%0 : tensor<4x1xf16>) -> tensor<4x1xf16>
  %3 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<4x1xi32>) -> tensor<4x1xi32>
  %4:2 = iree_linalg_ext.arg_compare {iree_linalg_ext.split_reduction = [1336 : index]} dimension(2) ins(%arg0 : tensor<4x1x128256xf16>) outs(%2, %3 : tensor<4x1xf16>, tensor<4x1xi32>) {
  ^bb0(%arg1: f16, %arg2: f16):
    %5 = arith.cmpf ogt, %arg1, %arg2 : f16
    iree_linalg_ext.yield %5 : i1
  } -> tensor<4x1xf16>, tensor<4x1xi32>
  util.return %4#0, %4#1 : tensor<4x1xf16>, tensor<4x1xi32>
}
// CHECK-LABEL:  @split_reduction_arg_compare(
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x1x128256xf16>
//       CHECK:    %[[FILL_F16:.+]] = linalg.fill {{.+}} -> tensor<4x1xf16>
//       CHECK:    %[[FILL_I32:.+]] = linalg.fill {{.+}} -> tensor<4x1xi32>
//       CHECK:    %[[EMPTY_F16:.+]] = tensor.empty() : tensor<4x1x96xf16>
//       CHECK:    %[[EMPTY_I32:.+]] = tensor.empty() : tensor<4x1x96xi32>
//       CHECK:    %[[DISPATCH:.+]]:2 = flow.dispatch.region -> (tensor<4x1x96xf16>, tensor<4x1x96xi32>) {
//       CHECK:      %[[FORALL:.+]]:2 = scf.forall (%{{.+}}) = (0) to (128256) step (1336)
//  CHECK-SAME:          shared_outs(%[[OUT0:.+]] = %[[EMPTY_F16]], %[[OUT1:.+]] = %[[EMPTY_I32]])
//  CHECK-SAME:          -> (tensor<4x1x96xf16>, tensor<4x1x96xi32>) {
//       CHECK:        %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:            [0, 0, %{{.+}}] [4, 1, 1336] [1, 1, 1]
//  CHECK-SAME:            : tensor<4x1x128256xf16> to tensor<4x1x1336xf16>
//   CHECK-NOT:        tensor.extract_slice %[[OUT0]]
//   CHECK-NOT:        linalg.broadcast
//       CHECK:        %[[ARG_COMPARE:.+]]:2 = iree_linalg_ext.arg_compare
//  CHECK-SAME:            dimension(2)
//  CHECK-SAME:            ins(%[[INPUT_SLICE]] : tensor<4x1x1336xf16>)
//  CHECK-SAME:            outs(%[[FILL_F16]], %[[FILL_I32]] : tensor<4x1xf16>, tensor<4x1xi32>)
//       CHECK:        scf.forall.in_parallel {
//       CHECK:          tensor.parallel_insert_slice %[[ARG_COMPARE]]#0 into %[[OUT0]]
//       CHECK:          tensor.parallel_insert_slice %[[ARG_COMPARE]]#1 into %[[OUT1]]
//       CHECK:        }
//       CHECK:      } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
//       CHECK:      flow.return %[[FORALL]]#0, %[[FORALL]]#1 : tensor<4x1x96xf16>, tensor<4x1x96xi32>
//       CHECK:    }
//       CHECK:    %[[REDUCE:.+]]:2 = iree_linalg_ext.arg_compare dimension(2)
//  CHECK-SAME:        ins(%[[DISPATCH]]#0, %[[DISPATCH]]#1 : tensor<4x1x96xf16>, tensor<4x1x96xi32>)
//  CHECK-SAME:        outs(%[[FILL_F16]], %[[FILL_I32]] : tensor<4x1xf16>, tensor<4x1xi32>)
//       CHECK:    util.return %[[REDUCE]]#0, %[[REDUCE]]#1

// -----

util.func public @split_exp_reduction(
    %arg0: tensor<4x16xf32>, %arg1: tensor<4096x16xf32>,
    %arg2: tensor<4096x16xf32>) -> tensor<4x16xf32> {
  %zero = arith.constant 0.0 : f32
  %neg_inf = arith.constant -3.40282347E+38 : f32
  %valid_length = arith.constant 4095 : index
  %score_empty = tensor.empty() : tensor<4x4096xf32>
  %score_init = linalg.fill ins(%zero : f32) outs(%score_empty : tensor<4x4096xf32>) -> tensor<4x4096xf32>
  %score = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4096x16xf32>)
    outs(%score_init : tensor<4x4096xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
    %product = arith.mulf %lhs, %rhs : f32
    %sum = arith.addf %product, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<4x4096xf32>
  %masked_empty = tensor.empty() : tensor<4x4096xf32>
  %masked_score = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%score : tensor<4x4096xf32>)
    outs(%masked_empty : tensor<4x4096xf32>) {
  ^bb0(%input: f32, %unused: f32):
    %index = linalg.index 1 : index
    %is_valid = arith.cmpi ult, %index, %valid_length : index
    %masked = arith.select %is_valid, %input, %neg_inf : f32
    linalg.yield %masked : f32
  } -> tensor<4x4096xf32>
  %stat_empty = tensor.empty() : tensor<4xf32>
  %max_init = linalg.fill ins(%neg_inf : f32) outs(%stat_empty : tensor<4xf32>) -> tensor<4xf32>
  %sum_init = linalg.fill ins(%zero : f32) outs(%stat_empty : tensor<4xf32>) -> tensor<4xf32>
  %output_empty = tensor.empty() : tensor<4x16xf32>
  %output_init = linalg.fill ins(%zero : f32) outs(%output_empty : tensor<4x16xf32>) -> tensor<4x16xf32>
  %result:3 = iree_linalg_ext.exp_reduction {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0)>,
      affine_map<(d0, d1, d2) -> (d0)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [
      #iree_linalg_ext.iterator_type<parallel>,
      #iree_linalg_ext.iterator_type<parallel>,
      #iree_linalg_ext.iterator_type<reduction>
    ],
    exp_reduced_operands = [1, 2]
  } attributes {
    iree_linalg_ext.split_reduction = [128 : index]
  } ins(%masked_score, %arg2 : tensor<4x4096xf32>, tensor<4096x16xf32>)
    outs(%max_init, %sum_init, %output_init : tensor<4xf32>, tensor<4xf32>, tensor<4x16xf32>) {
  ^bb0(%input: f32, %value: f32, %max: f32, %sum: f32, %output: f32):
    %index = iree_linalg_ext.index 2 : index
    %is_valid = arith.cmpi ult, %index, %valid_length : index
    %safe_input = arith.select %is_valid, %input, %zero : f32
    %new_sum = arith.addf %safe_input, %sum : f32
    %safe_value = arith.select %is_valid, %value, %zero : f32
    %weighted = arith.mulf %safe_input, %safe_value : f32
    %new_output = arith.addf %weighted, %output : f32
    iree_linalg_ext.yield %max, %new_sum, %new_output : f32, f32, f32
  } -> tensor<4xf32>, tensor<4xf32>, tensor<4x16xf32>
  util.return %result#2 : tensor<4x16xf32>
}
// CHECK-LABEL:  @split_exp_reduction(
//       CHECK:    %[[DISPATCH:.+]]:3 = flow.dispatch.region -> (tensor<4x32xf32>, tensor<4x32xf32>, tensor<4x16x32xf32>) {
//       CHECK:      %[[FORALL:.+]]:3 = scf.forall
//       CHECK:        %[[SCORE:.+]] = linalg.generic
//       CHECK:        %[[MASKED_SCORE:.+]] = linalg.generic
//       CHECK:          linalg.index 1
//       CHECK:        %[[MAX_SLICE:.+]] = tensor.extract_slice {{.+}} : tensor<4x32xf32> to tensor<4xf32>
//       CHECK:        %[[MAX_FILL:.+]] = linalg.fill {{.+}} outs(%[[MAX_SLICE]]
//       CHECK:        %[[SUM_SLICE:.+]] = tensor.extract_slice {{.+}} : tensor<4x32xf32> to tensor<4xf32>
//       CHECK:        %[[SUM_FILL:.+]] = linalg.fill {{.+}} outs(%[[SUM_SLICE]]
//       CHECK:        %[[OUTPUT_SLICE:.+]] = tensor.extract_slice {{.+}} : tensor<4x16x32xf32> to tensor<4x16xf32>
//       CHECK:        %[[OUTPUT_FILL:.+]] = linalg.fill {{.+}} outs(%[[OUTPUT_SLICE]]
//       CHECK:        %[[PARTIAL:.+]]:3 = iree_linalg_ext.exp_reduction
//   CHECK-NOT:            iree_linalg_ext.split_reduction
//  CHECK-SAME:            iterator_types = [
//  CHECK-SAME:              #iree_linalg_ext.iterator_type<parallel>,
//  CHECK-SAME:              #iree_linalg_ext.iterator_type<parallel>,
//  CHECK-SAME:              #iree_linalg_ext.iterator_type<reduction>
//  CHECK-SAME:            ins(%[[MASKED_SCORE]]{{.*}}tensor<4x128xf32>, tensor<128x16xf32>)
//  CHECK-SAME:            outs(%[[MAX_FILL]], %[[SUM_FILL]], %[[OUTPUT_FILL]] : tensor<4xf32>, tensor<4xf32>, tensor<4x16xf32>)
//       CHECK:        tensor.parallel_insert_slice %[[PARTIAL]]#0
//       CHECK:        tensor.parallel_insert_slice %[[PARTIAL]]#1
//       CHECK:        tensor.parallel_insert_slice %[[PARTIAL]]#2
//       CHECK:      flow.return %[[FORALL]]#0, %[[FORALL]]#1, %[[FORALL]]#2 : tensor<4x32xf32>, tensor<4x32xf32>, tensor<4x16x32xf32>
//       CHECK:    linalg.reduce
//       CHECK:    linalg.reduce

// -----

util.func public @split_reduction_bitext_producer(%arg0: tensor<?x?xf32>) -> tensor<?xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>

  %empty0 = tensor.empty(%dim0, %dim1) : tensor<?x?xf64>
  %extf = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>) outs(%empty0 : tensor<?x?xf64>) {
  ^bb0(%in: f32, %out: f64):
    %e = arith.extf %in : f32 to f64
    linalg.yield %e : f64
  } -> tensor<?x?xf64>

  %cst = arith.constant 0.000000e+00 : f64
  %empty1 = tensor.empty(%dim0) : tensor<?xf64>
  %init = linalg.fill ins(%cst : f64) outs(%empty1 : tensor<?xf64>) -> tensor<?xf64>
  %reduce = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"],
      iree_linalg_ext.split_reduction = [128]}
      ins(%extf : tensor<?x?xf64>) outs(%init : tensor<?xf64>) {
  ^bb0(%in: f64, %out: f64):
    %add = arith.addf %in, %out : f64
    linalg.yield %add : f64
  } -> tensor<?xf64>
  util.return %reduce : tensor<?xf64>
}
// Bit extends should be fused into the 'forall' loop.
// CHECK-LABEL:  @split_reduction_bitext_producer
// CHECK: scf.forall {{.+}} {
// CHECK:   linalg.generic {
// CHECK:     arith.extf
// CHECK:   }
// CHECK:   linalg.generic {
// CHECK:     arith.addf
// CHECK:   }
// CHECK: }
