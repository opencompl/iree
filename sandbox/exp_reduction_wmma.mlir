
#translation = #iree_codegen.translation_info<
    pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 1, 1]
    subgroup_size = 32,
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
      %argQ: !stream.binding,
      %argK: !stream.binding,
      %argV: !stream.binding,
      %ret: !stream.binding
    ) attributes {translation_info = #translation} {
      %cst0 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index

      %dispQ = stream.binding.subspan %argQ[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>>
      %dispK = stream.binding.subspan %argK[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>>
      %dispV = stream.binding.subspan %argV[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>>
      %dispR = stream.binding.subspan %ret[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>

      %Q = iree_tensor_ext.dispatch.tensor.load %dispQ, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>> -> tensor<4x32x4096x64xf16>
      %K = iree_tensor_ext.dispatch.tensor.load %dispK, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>> -> tensor<4x32x4096x64xf16>
      %V = iree_tensor_ext.dispatch.tensor.load %dispV, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf16>> -> tensor<4x32x4096x64xf16>

      %S_empty = tensor.empty() : tensor<4x32x4096x4096xf32>
      %S_fill  = linalg.fill ins(%cst0 : f32)
                              outs(%S_empty : tensor<4x32x4096x4096xf32>)
                              -> tensor<4x32x4096x4096xf32>

      %S = linalg.generic  {
          indexing_maps = [
            affine_map<(Z, H, N1, N2, D) -> (Z, H, N1, D)>,
            affine_map<(Z, H, N1, N2, D) -> (Z, H, N2, D)>,
            affine_map<(Z, H, N1, N2, D) -> (Z, H, N1, N2)>
          ],
          iterator_types = ["parallel", "parallel",  "parallel", "parallel", "reduction"],
          lowering_config = #iree_gpu.lowering_config<{
            subgroup_basis = [[1, 1, 4, 1, 1], [0, 1, 2, 3, 4]],
            mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>,
            promote_operands = [0, 1]
          }>
        }
        ins(%Q, %K : tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>)
        outs(%S_fill : tensor<4x32x4096x4096xf32>)
      {
      ^bb0(%q : f16, %k : f16, %s : f32):
        %q_ext = arith.extf %q : f16 to f32
        %k_ext = arith.extf %k : f16 to f32
        %mul  = arith.mulf %q_ext, %k_ext : f32
        %sum  = arith.addf %mul, %s : f32
        linalg.yield %sum : f32
      } -> tensor<4x32x4096x4096xf32>

      // Truncate S to f16 for the MMA in exp_reduction.
      %S_f16_empty = tensor.empty() : tensor<4x32x4096x4096xf16>
      %S_f16 = linalg.generic {
          indexing_maps = [
            affine_map<(a, b, c, d) -> (a, b, c, d)>,
            affine_map<(a, b, c, d) -> (a, b, c, d)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%S : tensor<4x32x4096x4096xf32>)
        outs(%S_f16_empty : tensor<4x32x4096x4096xf16>) {
      ^bb0(%s_in : f32, %s_out : f16):
        %trunc = arith.truncf %s_in : f32 to f16
        linalg.yield %trunc : f16
      } -> tensor<4x32x4096x4096xf16>

      %red_empty = tensor.empty() : tensor<4x32x4096x64xf32>
      %max_empty = tensor.empty() : tensor<4x32x4096xf32>

      %max_el = arith.constant -3.40282347E+38 : f32
      %max_init = linalg.fill ins(%max_el : f32)
                              outs(%max_empty : tensor<4x32x4096xf32>)
                              -> tensor<4x32x4096xf32>

      %sum_empty = tensor.empty() : tensor<4x32x4096xf32>
      %sum_el = arith.constant 0.000000e+00 : f32
      %sum_init = linalg.fill ins(%sum_el : f32)
                              outs(%sum_empty : tensor<4x32x4096xf32>)
                              -> tensor<4x32x4096xf32>
      %acc_init = linalg.fill ins(%sum_el : f32)
                              outs(%red_empty : tensor<4x32x4096x64xf32>)
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
          reduction = [0, 0, 0,  0, 64],
          subgroup_basis = [[1, 1, 4, 1, 1], [0, 1, 2, 3, 4]],
          mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16, col_major=true>,
          promote_operands = [1]
        }>}
        ins(%S_f16, %V : tensor<4x32x4096x4096xf16>, tensor<4x32x4096x64xf16>)
        outs(%max_init, %sum_init, %acc_init : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>)
      {
      ^bb0(%ex : f16, %v : f16, %m : f32, %sum : f32, %acc : f32):
        %ex_ext = arith.extf %ex : f16 to f32
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

func.func @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view {
  %c = arith.constant 1 : index
  %0 = hal.tensor.import %arg0 "q" : !hal.buffer_view -> tensor<4x32x4096x64xf16>
  %1 = hal.tensor.import %arg1 "k" : !hal.buffer_view -> tensor<4x32x4096x64xf16>
  %2 = hal.tensor.import %arg2 "v" : !hal.buffer_view -> tensor<4x32x4096x64xf16>

  %ret0 = flow.dispatch @executable_0::@dispatch[%c](%0, %1, %2) : (tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>) ->  tensor<4x32x4096x64xf32>

  %3 = hal.tensor.export %ret0 "out" : tensor<4x32x4096x64xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

} // module
