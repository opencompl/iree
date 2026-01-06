#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @exp_reduction_tile_tensor_static_uniform(%arg0: tensor<100xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c10 = arith.constant 10 : index
    %0:2 = scf.for %arg3 = %c0 to %c100 step %c10 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<f32>, tensor<f32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3] [10] [1] : tensor<100xf32> to tensor<10xf32>
      %1:2 = iree_linalg_ext.exp_reduction{indexing_maps = [#map, #map1, #map1], iterator_types = [#iree_linalg_ext.iterator_type<reduction>], exp_reduced_operands = [1]} ins(%extracted_slice : tensor<10xf32>) outs(%arg4, %arg5 : tensor<f32>, tensor<f32>) {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
        %2 = arith.addf %arg6, %arg8 : f32
        iree_linalg_ext.yield %arg7, %2 : f32, f32
      } -> tensor<f32>, tensor<f32>
      scf.yield %1#0, %1#1 : tensor<f32>, tensor<f32>
    }
    return %0#0, %0#1 : tensor<f32>, tensor<f32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["iree_linalg_ext.exp_reduction"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
