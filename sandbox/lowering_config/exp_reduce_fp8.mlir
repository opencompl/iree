
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
      %argQ: !stream.binding,
      %argK: !stream.binding,
      %argV: !stream.binding,
      %ret: !stream.binding
    ) attributes {translation_info = #translation} {
      %cst0 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index

      %dispQ = stream.binding.subspan %argQ[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf8E4M3FNUZ>>
      %dispK = stream.binding.subspan %argK[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf8E4M3FNUZ>>
      %dispV = stream.binding.subspan %argV[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf8E4M3FNUZ>>
      %dispR = stream.binding.subspan %ret[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>

      %Q = iree_tensor_ext.dispatch.tensor.load %dispQ, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf8E4M3FNUZ>> -> tensor<4x32x4096x64xf8E4M3FNUZ>
      %K = iree_tensor_ext.dispatch.tensor.load %dispK, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf8E4M3FNUZ>> -> tensor<4x32x4096x64xf8E4M3FNUZ>
      %V = iree_tensor_ext.dispatch.tensor.load %dispV, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x64xf8E4M3FNUZ>> -> tensor<4x32x4096x64xf8E4M3FNUZ>

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
            subgroup_basis = [[1, 1, 2, 1, 1], [0, 1, 2, 3, 4]],
            mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
            promote_operands = [0, 1]
          }>
        }
        ins(%Q, %K : tensor<4x32x4096x64xf8E4M3FNUZ>, tensor<4x32x4096x64xf8E4M3FNUZ>)
        outs(%S_fill : tensor<4x32x4096x4096xf32>)
      {
      ^bb0(%q : f8E4M3FNUZ, %k : f8E4M3FNUZ, %s : f32):
        %q_ext = arith.extf %q : f8E4M3FNUZ to f32
        %k_ext = arith.extf %k : f8E4M3FNUZ to f32
        %mul  = arith.mulf %q_ext, %k_ext : f32
        %sum  = arith.addf %mul, %s : f32
        linalg.yield %sum : f32
      } -> tensor<4x32x4096x4096xf32>

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
          reduction = [0, 0, 0,  0, 32],
          subgroup_basis = [[1, 1, 2, 1, 1], [0, 1, 2, 3, 4]],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ, col_major=true>
        }>}
        ins(%S, %V : tensor<4x32x4096x4096xf8E4M3FNUZ>, tensor<4x32x4096x64xf8E4M3FNUZ>)
        outs(%max_init, %sum_init, %acc_init : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>)
      {
      ^bb0(%ex : f8E4M3FNUZ, %v : f8E4M3FNUZ, %m : f32, %sum : f32, %acc : f32):
        %trunc = arith.truncf %ex : f32 to f8E4M3FNUZ
        %ex_ext = arith.extf %trunc : f8E4M3FNUZ to f32
        %v_ext = arith.extf %v : f8E4M3FNUZ to f32
        %nsum = arith.addf %ex, %sum : f32
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
  %0 = hal.tensor.import %arg0 "q" : !hal.buffer_view -> tensor<4x32x4096x64xf8E4M3FNUZ>
  %1 = hal.tensor.import %arg1 "k" : !hal.buffer_view -> tensor<4x32x4096x64xf8E4M3FNUZ>
  %2 = hal.tensor.import %arg2 "v" : !hal.buffer_view -> tensor<4x32x4096x64xf8E4M3FNUZ>

  %ret0 = flow.dispatch @executable_0::@dispatch[%c](%0, %1, %2) : (tensor<4x32x4096x64xf8E4M3FNUZ>, tensor<4x32x4096x64xf8E4M3FNUZ>, tensor<4x32x4096x64xf8E4M3FNUZ>) ->  tensor<4x32x4096x64xf32>

  %3 = hal.tensor.export %ret0 "out" : tensor<4x32x4096x64xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

} // module
