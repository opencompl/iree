
func.func @exp_reduction_tile_tensor_static_uniform(
    %S: tensor<100xf32>,
    %M: tensor<f32>,
    %out: tensor<f32>
) -> (tensor<f32>, tensor<f32>) {
    %max, %sum = iree_linalg_ext.exp_reduction {
    indexing_maps = [
      affine_map<(N)->(N)>,
      affine_map<(N)->()>,
      affine_map<(N)->()>
    ],
    iterator_types = [
      #iree_linalg_ext.iterator_type<reduction>
    ],
    exp_reduced_operands = [1]
  } ins(%S: tensor<100xf32>)
    outs(%M, %out: tensor<f32>, tensor<f32>)
  {
  ^bb0(%s: f32, %m: f32, %o: f32):
    %add = arith.addf %s, %o: f32
    iree_linalg_ext.yield %m, %add: f32, f32
  } -> tensor<f32>, tensor<f32>
  return %max, %sum : tensor<f32>, tensor<f32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.exp_reduction"]} in %module_op
         : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [10]
         : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // transform.structured.tile_reduction_using_for %0 by tile_sizes = [10, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
