#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @e2e {
  func.func @attention(
      %arg0: tensor<4x32x4096x64xf8E4M3FN>,
      %arg1: tensor<4x32x4096x64xf8E4M3FN>,
      %arg2: tensor<4x32x4096x64xf8E4M3FN>)
      -> tensor<4x32x4096x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x32x4096x4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x32x4096x4096xf32>) -> tensor<4x32x4096x4096xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x32x4096x64xf8E4M3FN>, tensor<4x32x4096x64xf8E4M3FN>) outs(%1 : tensor<4x32x4096x4096xf32>) {
    ^bb0(%in: f8E4M3FN, %in_0: f8E4M3FN, %out: f32):
      %8 = arith.extf %in : f8E4M3FN to f32
      %9 = arith.extf %in_0 : f8E4M3FN to f32
      %10 = arith.mulf %8, %9 : f32
      %11 = arith.addf %10, %out : f32
      linalg.yield %11 : f32
    } -> tensor<4x32x4096x4096xf32>
    %3 = tensor.empty() : tensor<4x32x4096x64xf32>
    %4 = tensor.empty() : tensor<4x32x4096xf32>
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<4x32x4096xf32>) -> tensor<4x32x4096xf32>
    %6 = tensor.empty() : tensor<4x32x4096xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<4x32x4096xf32>) -> tensor<4x32x4096xf32>
    %12 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<4x32x4096x64xf32>) -> tensor<4x32x4096x64xf32>
    %13:3 = iree_linalg_ext.exp_reduction{indexing_maps = [#map, #map3, #map4, #map4, #map2], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>], exp_reduced_operands = [1, 2]} ins(%2, %arg2 : tensor<4x32x4096x4096xf32>, tensor<4x32x4096x64xf8E4M3FN>) outs(%5, %7, %12 : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>) {
    ^bb0(%arg3: f32, %arg4: f8E4M3FN, %arg5: f32, %arg6: f32, %arg7: f32):
      %cst_2 = arith.constant 4.480000e+02 : f32
      %15 = arith.mulf %arg3, %cst_2 : f32
      %16 = arith.truncf %15 : f32 to f8E4M3FN
      %17 = arith.extf %16 : f8E4M3FN to f32
      %18 = arith.extf %arg4 : f8E4M3FN to f32
      %19 = arith.addf %arg3, %arg6 : f32
      %20 = arith.mulf %17, %18 : f32
      %21 = arith.addf %20, %arg7 : f32
      iree_linalg_ext.yield %arg5, %19, %21 : f32, f32, f32
    } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>
    %14 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13#1 : tensor<4x32x4096xf32>) outs(%13#2 : tensor<4x32x4096x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.divf %out, %in : f32
      linalg.yield %15 : f32
    } -> tensor<4x32x4096x64xf32>
    return %14 : tensor<4x32x4096x64xf32>
  }
}
