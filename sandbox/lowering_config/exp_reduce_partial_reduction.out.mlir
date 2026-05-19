#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @exp_reduction_partial_reduction(%arg0: tensor<64x4096xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %3:2 = scf.for %arg3 = %c0 to %c4096 step %c128 iter_args(%arg4 = %1, %arg5 = %2) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[0, %arg3] [64, 128] [1, 1] : tensor<64x4096xf32> to tensor<64x128xf32>
      %6:2 = iree_linalg_ext.exp_reduction{indexing_maps = [#map, #map, #map], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>], exp_reduced_operands = [1]} ins(%extracted_slice : tensor<64x128xf32>) outs(%arg4, %arg5 : tensor<64x128xf32>, tensor<64x128xf32>) {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
        %7 = arith.addf %arg6, %arg8 : f32
        iree_linalg_ext.yield %arg7, %7 : f32, f32
      } -> tensor<64x128xf32>, tensor<64x128xf32>
      scf.yield %6#0, %6#1 : tensor<64x128xf32>, tensor<64x128xf32>
    }
    %reduced = linalg.reduce ins(%3#0 : tensor<64x128xf32>) outs(%arg1 : tensor<64xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %6 = arith.maximumf %in, %init : f32
        linalg.yield %6 : f32
      }
    %4 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%reduced : tensor<64xf32>) outs(%3#0 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.subf %out, %in : f32
      %7 = math.exp2 %6 : f32
      linalg.yield %7 : f32
    } -> tensor<64x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<64x128xf32>) outs(%3#1 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.mulf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<64x128xf32>
    %reduced_1 = linalg.reduce ins(%5 : tensor<64x128xf32>) outs(%arg2 : tensor<64xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %6 = arith.addf %in, %init : f32
        linalg.yield %6 : f32
      }
    return %reduced, %reduced_1 : tensor<64xf32>, tensor<64xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["iree_linalg_ext.exp_reduction"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %fill_op:3, %split_op, %combining_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [0, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

