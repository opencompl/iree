// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --verify-diagnostics -canonicalize -cse %s

func.func @exp_reduction_split_k_partial_reduction(%S: tensor<64x4096xf32>,
                                                   %M: tensor<64xf32>,
                                                   %out: tensor<64xf32>)
    -> (tensor<64xf32>, tensor<64xf32>) {
  %max, %sum = iree_linalg_ext.exp_reduction {
    indexing_maps = [
      affine_map<(M,N)->(M,N)>,
      affine_map<(M,N)->(M)>,
      affine_map<(M,N)->(M)>
    ],
    iterator_types = [
      #iree_linalg_ext.iterator_type<parallel>,
      #iree_linalg_ext.iterator_type<reduction>
    ],
    exp_reduced_operands = [1]
  } ins(%S: tensor<64x4096xf32>)
    outs(%M, %out: tensor<64xf32>, tensor<64xf32>)
  {
  ^bb0(%s: f32, %m: f32, %o: f32):
    %add = arith.addf %s, %o: f32
    iree_linalg_ext.yield %m, %add: f32, f32
  } -> tensor<64xf32>, tensor<64xf32>
  return %max, %sum : tensor<64xf32>, tensor<64xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.exp_reduction"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %fill_op:2, %split_op, %combining_op:2, %forall_op = transform.structured.tile_reduction_using_forall %0 by tile_sizes = [0, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
