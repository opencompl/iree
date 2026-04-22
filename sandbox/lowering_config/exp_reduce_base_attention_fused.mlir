#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}>
module @e2e {
  flow.executable private @executable_0 {
    flow.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 32, 4096, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>> -> tensor<4x32x4096x64xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 32, 4096, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>> -> tensor<4x32x4096x64xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [4, 32, 4096, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>> -> tensor<4x32x4096x64xf16>
        %7 = tensor.empty() : tensor<4x32x4096x4096xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x32x4096x4096xf32>) -> tensor<4x32x4096x4096xf32>
        %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%4, %5 : tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>) outs(%8 : tensor<4x32x4096x4096xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>, promote_operands = [0, 1], subgroup_basis = [[1, 1, 4, 1, 1], [0, 1, 2, 3, 4]]}>} {
        ^bb0(%in: f16, %in_2: f16, %out: f32):
          %18 = arith.extf %in : f16 to f32
          %19 = arith.extf %in_2 : f16 to f32
          %20 = arith.mulf %18, %19 : f32
          %21 = arith.addf %20, %out : f32
          linalg.yield %21 : f32
        } -> tensor<4x32x4096x4096xf32>
        %10 = tensor.empty() : tensor<4x32x4096x64xf32>
        %11 = tensor.empty() : tensor<4x32x4096xf32>
        %cst_0 = arith.constant -3.40282347E+38 : f32
        %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<4x32x4096xf32>) -> tensor<4x32x4096xf32>
        %13 = tensor.empty() : tensor<4x32x4096xf32>
        %cst_1 = arith.constant 0.000000e+00 : f32
        %14 = linalg.fill ins(%cst_1 : f32) outs(%13 : tensor<4x32x4096xf32>) -> tensor<4x32x4096xf32>
        %15 = linalg.fill ins(%cst_1 : f32) outs(%10 : tensor<4x32x4096x64xf32>) -> tensor<4x32x4096x64xf32>
        %16:3 = iree_linalg_ext.exp_reduction{indexing_maps = [#map, #map3, #map4, #map4, #map2], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>], exp_reduced_operands = [1, 2]} attributes {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>, reduction = [0, 0, 0, 0, 32], subgroup_basis = [[1, 1, 4, 1, 1], [0, 1, 2, 3, 4]], workgroup = [1, 1, 64, 0, 0]}>} ins(%9, %6 : tensor<4x32x4096x4096xf32>, tensor<4x32x4096x64xf16>) outs(%12, %14, %15 : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>) {
        ^bb0(%arg4: f32, %arg5: f16, %arg6: f32, %arg7: f32, %arg8: f32):
          %21 = arith.addf %arg4, %arg7 : f32
          %18 = arith.truncf %arg4 : f32 to f16
          %19 = arith.extf %18 : f16 to f32
          %20 = arith.extf %arg5 : f16 to f32
          %22 = arith.mulf %19, %20 : f32
          %23 = arith.addf %22, %arg8 : f32
          iree_linalg_ext.yield %arg6, %21, %23 : f32, f32, f32
        } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>
        %17 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16#1 : tensor<4x32x4096xf32>) outs(%16#2 : tensor<4x32x4096x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %18 = arith.divf %out, %in : f32
          linalg.yield %18 : f32
        } -> tensor<4x32x4096x64xf32>
        iree_tensor_ext.dispatch.tensor.store %17, %3, offsets = [0, 0, 0, 0], sizes = [4, 32, 4096, 64], strides = [1, 1, 1, 1] : tensor<4x32x4096x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>
        return
      }
    }
  }
  func.func @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view {
    %c1 = arith.constant 1 : index
    %0 = hal.tensor.import %arg0 "q" : !hal.buffer_view -> tensor<4x32x4096x64xf16>
    %1 = hal.tensor.import %arg1 "k" : !hal.buffer_view -> tensor<4x32x4096x64xf16>
    %2 = hal.tensor.import %arg2 "v" : !hal.buffer_view -> tensor<4x32x4096x64xf16>
    %3 = flow.dispatch @executable_0::@dispatch[%c1](%0, %1, %2) : (tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>) -> tensor<4x32x4096x64xf32>
    %4 = hal.tensor.export %3 "out" : tensor<4x32x4096x64xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
