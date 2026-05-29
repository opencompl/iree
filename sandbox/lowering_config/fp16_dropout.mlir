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
    ^bb0(%ex: f32, %v: f16, %max: f32, %sum: f32, %pv: f32):
      %idx0 = iree_linalg_ext.index 0 : index
      %idx1 = iree_linalg_ext.index 1 : index
      %idx2 = iree_linalg_ext.index 2 : index
      %idx4 = iree_linalg_ext.index 4 : index
      %idx0_i32 = arith.index_castui %idx0 : index to i32
      %idx1_i32 = arith.index_castui %idx1 : index to i32
      %idx2_i32 = arith.index_castui %idx2 : index to i32
      %idx4_i32 = arith.index_castui %idx4 : index to i32
      %p = arith.constant 5.000000e-02 : f32
      %cst_10000_f32 = arith.constant 1.000000e+04 : f32
      %cutoff_f32 = arith.mulf %cst_10000_f32, %p : f32
      %cutoff = arith.fptoui %cutoff_f32 : f32 to i32
      %cst_10000_i32 = arith.constant 10000 : i32
      %cst_hash_bound = arith.constant 99990000 : i32
      %cst_hash0 = arith.constant 73856093 : i32
      %cst_hash1 = arith.constant 19349663 : i32
      %cst_hash2 = arith.constant 83492791 : i32
      %cst_hash3 = arith.constant 265443576 : i32
      %hash0 = arith.muli %idx0_i32, %cst_hash0 : i32
      %hash1 = arith.muli %idx1_i32, %cst_hash1 : i32
      %hash2 = arith.muli %idx2_i32, %cst_hash2 : i32
      %hash3 = arith.muli %idx4_i32, %cst_hash3 : i32
      %hash01 = arith.xori %hash0, %hash1 : i32
      %hash23 = arith.xori %hash2, %hash3 : i32
      %hash = arith.xori %hash01, %hash23 : i32
      %bounded_hash = arith.remui %hash, %cst_hash_bound : i32
      %random = arith.ceildivui %bounded_hash, %cst_10000_i32 : i32
      %drop = arith.cmpi ult, %random, %cutoff : i32
      %zero = arith.constant 0.000000e+00 : f32
      %masked_ex = arith.select %drop, %zero, %ex : f32
      %14 = arith.addf %masked_ex, %sum : f32
      %15 = arith.truncf %masked_ex : f32 to f16
      %16 = arith.extf %15 : f16 to f32
      %17 = arith.extf %v : f16 to f32
      %18 = arith.mulf %16, %17 : f32
      %19 = arith.addf %18, %pv : f32
      iree_linalg_ext.yield %max, %14, %19 : f32, f32, f32
    } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>
    %14 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13#1 : tensor<4x32x4096xf32>) outs(%13#2 : tensor<4x32x4096x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.divf %out, %in : f32
      linalg.yield %15 : f32
    } -> tensor<4x32x4096x64xf32>
    return %14 : tensor<4x32x4096x64xf32>
  }
}
