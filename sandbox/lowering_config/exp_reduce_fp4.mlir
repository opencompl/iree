
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
      %argQS: !stream.binding,
      %argKS: !stream.binding,
      %argVS: !stream.binding,
      %ret: !stream.binding
    ) attributes {translation_info = #translation} {
      %cst0 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index

      %dispQ = stream.binding.subspan %argQ[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2x32xf4E2M1FN>>
      %dispK = stream.binding.subspan %argK[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2x32xf4E2M1FN>>
      %dispV = stream.binding.subspan %argV[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2x32xf4E2M1FN>>
      %dispQS = stream.binding.subspan %argQS[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2xf8E8M0FNU>>
      %dispKS = stream.binding.subspan %argKS[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2xf8E8M0FNU>>
      %dispVS = stream.binding.subspan %argVS[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2xf8E8M0FNU>>
      %dispR = stream.binding.subspan %ret[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>

      %Q = iree_tensor_ext.dispatch.tensor.load %dispQ, offsets = [0,0,0,0,0], sizes = [4,32,4096,2,32], strides = [1,1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2x32xf4E2M1FN>> -> tensor<4x32x4096x2x32xf4E2M1FN>
      %K = iree_tensor_ext.dispatch.tensor.load %dispK, offsets = [0,0,0,0,0], sizes = [4,32,4096,2,32], strides = [1,1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2x32xf4E2M1FN>> -> tensor<4x32x4096x2x32xf4E2M1FN>
      %V = iree_tensor_ext.dispatch.tensor.load %dispV, offsets = [0,0,0,0,0], sizes = [4,32,4096,2,32], strides = [1,1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2x32xf4E2M1FN>> -> tensor<4x32x4096x2x32xf4E2M1FN>
      %QS = iree_tensor_ext.dispatch.tensor.load %dispQS, offsets = [0,0,0,0], sizes = [4,32,4096,2], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2xf8E8M0FNU>> -> tensor<4x32x4096x2xf8E8M0FNU>
      %KS = iree_tensor_ext.dispatch.tensor.load %dispKS, offsets = [0,0,0,0], sizes = [4,32,4096,2], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2xf8E8M0FNU>> -> tensor<4x32x4096x2xf8E8M0FNU>
      %VS = iree_tensor_ext.dispatch.tensor.load %dispVS, offsets = [0,0,0,0], sizes = [4,32,4096,2], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x2xf8E8M0FNU>> -> tensor<4x32x4096x2xf8E8M0FNU>

      %S_empty = tensor.empty() : tensor<4x32x4096x4096xf32>
      %S_fill  = linalg.fill ins(%cst0 : f32)
                              outs(%S_empty : tensor<4x32x4096x4096xf32>)
                              -> tensor<4x32x4096x4096xf32>

      %S = linalg.generic  {
          indexing_maps = [
            affine_map<(Z, H, N1, N2, DG, DB) -> (Z, H, N1, DG, DB)>,
            affine_map<(Z, H, N1, N2, DG, DB) -> (Z, H, N2, DG, DB)>,
            affine_map<(Z, H, N1, N2, DG, DB) -> (Z, H, N1, DG)>,
            affine_map<(Z, H, N1, N2, DG, DB) -> (Z, H, N2, DG)>,
            affine_map<(Z, H, N1, N2, DG, DB) -> (Z, H, N1, N2)>
          ],
          iterator_types = ["parallel", "parallel",  "parallel", "parallel", "reduction", "reduction"],
          lowering_config = #iree_gpu.lowering_config<{
            subgroup_basis = [[1, 1, 2, 1, 1, 1], [0, 1, 2, 3, 4, 5]],
            mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>,
            promote_operands = [0, 1, 2, 3]
          }>
        }
        ins(%Q, %K, %QS, %KS : tensor<4x32x4096x2x32xf4E2M1FN>, tensor<4x32x4096x2x32xf4E2M1FN>, tensor<4x32x4096x2xf8E8M0FNU>, tensor<4x32x4096x2xf8E8M0FNU>)
        outs(%S_fill : tensor<4x32x4096x4096xf32>)
      {
      ^bb0(%q : f4E2M1FN, %k : f4E2M1FN, %q_scale : f8E8M0FNU, %k_scale : f8E8M0FNU, %s : f32):
        %q_ext = arith.scaling_extf %q, %q_scale : f4E2M1FN, f8E8M0FNU to f32
        %k_ext = arith.scaling_extf %k, %k_scale : f4E2M1FN, f8E8M0FNU to f32
        %mul  = arith.mulf %q_ext, %k_ext : f32
        %sum  = arith.addf %mul, %s : f32
        linalg.yield %sum : f32
      } -> tensor<4x32x4096x4096xf32>

      %red_empty = tensor.empty() : tensor<4x32x4096x2x32xf32>
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
                              outs(%red_empty : tensor<4x32x4096x2x32xf32>)
                              -> tensor<4x32x4096x2x32xf32>

      %MAX, %SUM, %PV = iree_linalg_ext.exp_reduction {
        indexing_maps = [
          affine_map<(B, H, M, DG, DB, K2) -> (B, H, M, K2)>,
          affine_map<(B, H, M, DG, DB, K2) -> (B, H, K2, DG, DB)>,
          affine_map<(B, H, M, DG, DB, K2) -> (B, H, K2, DG)>,
          affine_map<(B, H, M, DG, DB, K2) -> (B, H, M)>,
          affine_map<(B, H, M, DG, DB, K2) -> (B, H, M)>,
          affine_map<(B, H, M, DG, DB, K2) -> (B, H, M, DG, DB)>
        ],
        iterator_types = [
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<reduction>
        ],
        exp_reduced_operands = [1, 2]
      }
        attributes {lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 0, 128],
          subgroup_basis = [[1, 1, 2, 1, 1, 1], [0, 1, 2, 3, 4, 5]],
          mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
        }>}
        ins(%S, %V, %VS : tensor<4x32x4096x4096xf32>, tensor<4x32x4096x2x32xf4E2M1FN>, tensor<4x32x4096x2xf8E8M0FNU>)
        outs(%max_init, %sum_init, %acc_init : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x2x32xf32>)
      {
      ^bb0(%ex : f32, %v : f4E2M1FN, %v_scale : f8E8M0FNU, %m : f32, %sum : f32, %acc : f32):
        %s_scale_f32 = arith.constant 2.500000e-01 : f32
        %s_scale = arith.truncf %s_scale_f32 : f32 to f8E8M0FNU
        %ex_trunc = arith.scaling_truncf %ex, %s_scale : f32, f8E8M0FNU to f4E2M1FN
        %ex_ext = arith.scaling_extf %ex_trunc, %s_scale : f4E2M1FN, f8E8M0FNU to f32
        %v_ext = arith.scaling_extf %v, %v_scale : f4E2M1FN, f8E8M0FNU to f32
        %nsum = arith.addf %ex, %sum : f32
        %mul  = arith.mulf %ex_ext, %v_ext : f32
        %nacc = arith.addf %mul, %acc : f32
        iree_linalg_ext.yield %m, %nsum, %nacc : f32, f32, f32
      } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x2x32xf32>

      %result = linalg.generic {
                  indexing_maps = [
                    affine_map<(B, H, M, DG, DB) -> (B, H, M)>,
                    affine_map<(B, H, M, DG, DB) -> (B, H, M, DG, DB)>
                  ],
                  iterator_types = ["parallel",  "parallel", "parallel", "parallel", "parallel"]
                }
                ins(%SUM : tensor<4x32x4096xf32>)
                outs(%PV : tensor<4x32x4096x2x32xf32>) {
      ^bb0(%sum : f32, %pv : f32):
        %out = arith.divf %pv, %sum : f32
        linalg.yield %out : f32
      } -> tensor<4x32x4096x2x32xf32>

      %result_flat = tensor.collapse_shape %result [[0], [1], [2], [3, 4]] : tensor<4x32x4096x2x32xf32> into tensor<4x32x4096x64xf32>
      iree_tensor_ext.dispatch.tensor.store %result_flat, %dispR, offsets = [0,0,0,0], sizes = [4,32,4096,64], strides = [1,1,1,1] : tensor<4x32x4096x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x64xf32>>
      return
    }
  }
}

func.func @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view) -> !hal.buffer_view {
  %c = arith.constant 1 : index
  %q_bytes = hal.tensor.import %arg0 "q" : !hal.buffer_view -> tensor<4x32x4096x32xi8>
  %k_bytes = hal.tensor.import %arg1 "k" : !hal.buffer_view -> tensor<4x32x4096x32xi8>
  %v_bytes = hal.tensor.import %arg2 "v" : !hal.buffer_view -> tensor<4x32x4096x32xi8>
  %q_scale_bytes = hal.tensor.import %arg3 "q_scale" : !hal.buffer_view -> tensor<4x32x4096x2xi8>
  %k_scale_bytes = hal.tensor.import %arg4 "k_scale" : !hal.buffer_view -> tensor<4x32x4096x2xi8>
  %v_scale_bytes = hal.tensor.import %arg5 "v_scale" : !hal.buffer_view -> tensor<4x32x4096x2xi8>
  %q = iree_tensor_ext.bitcast %q_bytes : tensor<4x32x4096x32xi8> -> tensor<4x32x4096x2x32xf4E2M1FN>
  %k = iree_tensor_ext.bitcast %k_bytes : tensor<4x32x4096x32xi8> -> tensor<4x32x4096x2x32xf4E2M1FN>
  %v = iree_tensor_ext.bitcast %v_bytes : tensor<4x32x4096x32xi8> -> tensor<4x32x4096x2x32xf4E2M1FN>
  %q_scale = iree_tensor_ext.bitcast %q_scale_bytes : tensor<4x32x4096x2xi8> -> tensor<4x32x4096x2xf8E8M0FNU>
  %k_scale = iree_tensor_ext.bitcast %k_scale_bytes : tensor<4x32x4096x2xi8> -> tensor<4x32x4096x2xf8E8M0FNU>
  %v_scale = iree_tensor_ext.bitcast %v_scale_bytes : tensor<4x32x4096x2xi8> -> tensor<4x32x4096x2xf8E8M0FNU>

  %ret0 = flow.dispatch @executable_0::@dispatch[%c](%q, %k, %v, %q_scale, %k_scale, %v_scale) : (tensor<4x32x4096x2x32xf4E2M1FN>, tensor<4x32x4096x2x32xf4E2M1FN>, tensor<4x32x4096x2x32xf4E2M1FN>, tensor<4x32x4096x2xf8E8M0FNU>, tensor<4x32x4096x2xf8E8M0FNU>, tensor<4x32x4096x2xf8E8M0FNU>) ->  tensor<4x32x4096x64xf32>

  %ret = hal.tensor.export %ret0 "out" : tensor<4x32x4096x64xf32> -> !hal.buffer_view
  return %ret : !hal.buffer_view
}

} // module
