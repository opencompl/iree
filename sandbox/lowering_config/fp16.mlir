#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @e2e {
  func.func @attention(
      %arg0: tensor<4x32x4096x64xf16>,
      %arg1: tensor<4x32x4096x64xf16>,
      %arg2: tensor<4x32x4096x64xf16>)
      -> tensor<4x32x4096x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x32x4096x4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x32x4096x4096xf32>) -> tensor<4x32x4096x4096xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>) outs(%1 : tensor<4x32x4096x4096xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>, promote_operands = [0, 1], subgroup_basis = [[1, 1, 4, 1, 1], [0, 1, 2, 3, 4]]}>} {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %8 = arith.extf %in : f16 to f32
      %9 = arith.extf %in_0 : f16 to f32
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
    %13:3 = iree_linalg_ext.exp_reduction{indexing_maps = [#map, #map3, #map4, #map4, #map2], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>], exp_reduced_operands = [1, 2]} attributes {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>, reduction = [0, 0, 0, 0, 32], subgroup_basis = [[1, 1, 4, 1, 1], [0, 1, 2, 3, 4]], workgroup = [1, 1, 64, 0, 0]}>} ins(%2, %arg2 : tensor<4x32x4096x4096xf32>, tensor<4x32x4096x64xf16>) outs(%5, %7, %12 : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>) {
    ^bb0(%arg3: f32, %arg4: f16, %arg5: f32, %arg6: f32, %arg7: f32):
      %15 = arith.addf %arg3, %arg6 : f32
      %16 = arith.truncf %arg3 : f32 to f16
      %17 = arith.extf %16 : f16 to f32
      %18 = arith.extf %arg4 : f16 to f32
      %19 = arith.mulf %17, %18 : f32
      %20 = arith.addf %19, %arg7 : f32
      iree_linalg_ext.yield %arg5, %15, %20 : f32, f32, f32
    } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>
    %14 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13#1 : tensor<4x32x4096xf32>) outs(%13#2 : tensor<4x32x4096x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.divf %out, %in : f32
      linalg.yield %15 : f32
    } -> tensor<4x32x4096x64xf32>
    return %14 : tensor<4x32x4096x64xf32>
  }
}
