
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [256, 1, 1]
    subgroup_size = 64,
    {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}
  >

module @e2e {

flow.executable private @executable_0 {
  flow.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @dispatch(
      %argS: !stream.binding,
      %argV: !stream.binding,
      %ret: !stream.binding
    ) attributes {translation_info = #translation} {
      %c0 = arith.constant 0 : index

      %dispS = stream.binding.subspan %argS[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x4096xf32>>
      %dispV = stream.binding.subspan %argV[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>>
      %dispR = stream.binding.subspan %ret[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>

      %S = iree_tensor_ext.dispatch.tensor.load %dispS, offsets = [0,0,0,0], sizes = [4,32,4096,4096], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x4096xf32>> -> tensor<4x32x4096x4096xf32>
      %V = iree_tensor_ext.dispatch.tensor.load %dispV, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>> -> tensor<4x32x4096x64xf16>

      %red_empty = tensor.empty() : tensor<4x32x4096x64xf32>
      %cst_neg_inf = arith.constant -3.40282347E+38 : f32
      %cst_zero = arith.constant 0.000000e+00 : f32

      %max_empty = tensor.empty() : tensor<4x32x4096xf32>
      %max_init = linalg.fill ins(%cst_neg_inf : f32)
                              outs(%max_empty : tensor<4x32x4096xf32>)
                              -> tensor<4x32x4096xf32>

      %sum_empty = tensor.empty() : tensor<4x32x4096xf32>
      %sum_init = linalg.fill ins(%cst_zero : f32)
                              outs(%sum_empty : tensor<4x32x4096xf32>)
                              -> tensor<4x32x4096xf32>
      %acc_empty = tensor.empty() : tensor<4x32x4096x64xf32>
      %acc_init = linalg.fill ins(%cst_zero : f32)
                              outs(%acc_empty : tensor<4x32x4096x64xf32>)
                              -> tensor<4x32x4096x64xf32>

      %MAX, %SUM, %PV = iree_linalg_ext.exp_reduction {
        indexing_maps = [
          affine_map<(B, H, M, N, K2) -> (B, H, M, K2)>,
          affine_map<(B, H, M, N, K2) -> (B, H, K2, N)>,
          affine_map<(B, H, M, N, K2) -> (B, H, M)>,
          affine_map<(B, H, M, N, K2) -> (B, H, M)>,
          affine_map<(B, H, M, N, K2) -> (B, H, M, N)>
        ],
        iterator_types = [
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<reduction>
        ],
        exp_reduced_operands = [1, 2]
      }
        attributes {lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 64, 0, 0],
          reduction = [0, 0, 0,  0, 32],
          subgroup_basis = [[1, 1, 4, 1, 1], [0, 1, 2, 3, 4]],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major=true>
        }>}
        ins(%S, %V : tensor<4x32x4096x4096xf32>, tensor<4x32x4096x64xf16>)
        outs(%max_init, %sum_init, %acc_init : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>)
      {
      ^bb0(%ex : f32, %v : f16, %m : f32, %sum : f32, %acc : f32):
        %trunc = arith.truncf %ex : f32 to f16
        %ex_ext = arith.extf %trunc : f16 to f32
        %v_ext = arith.extf %v : f16 to f32
        %nsum = arith.addf %ex_ext, %sum : f32
        %mul  = arith.mulf %ex_ext, %v_ext : f32
        %nacc = arith.addf %mul, %acc : f32
        iree_linalg_ext.yield %m, %nsum, %nacc : f32, f32, f32
      } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>

      %result = linalg.generic {
                  indexing_maps = [
                    affine_map<(B, H, M, N) -> (B, H, M)>,
                    affine_map<(B, H, M, N) -> (B, H, M, N)>
                  ],
                  iterator_types = ["parallel",  "parallel", "parallel", "parallel"]
                }
                ins(%SUM : tensor<4x32x4096xf32>)
                outs(%PV : tensor<4x32x4096x64xf32>) {
      ^bb0(%sum : f32, %pv : f32):
        %out = arith.divf %pv, %sum : f32
        linalg.yield %out : f32
      } -> tensor<4x32x4096x64xf32>

      iree_tensor_ext.dispatch.tensor.store %result, %dispR, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : tensor<4x32x4096x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>
      return
    }
  }
}

func.func @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view {
  %c = arith.constant 1 : index
  %0 = hal.tensor.import %arg0 "s" : !hal.buffer_view -> tensor<4x32x4096x4096xf32>
  %1 = hal.tensor.import %arg1 "v" : !hal.buffer_view -> tensor<4x32x4096x64xf16>

  %ret0 = flow.dispatch @executable_0::@dispatch[%c](%0, %1) : (tensor<4x32x4096x4096xf32>, tensor<4x32x4096x64xf16>) ->  tensor<4x32x4096x64xf32>

  %3 = hal.tensor.export %ret0 "out" : tensor<4x32x4096x64xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

} // module
