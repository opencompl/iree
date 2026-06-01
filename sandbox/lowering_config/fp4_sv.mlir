#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [256, 1, 1]
    subgroup_size = 64,
    {
      gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true>,
      iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">
    }
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
      %argVS: !stream.binding,
      %argPS: !stream.binding,
      %ret: !stream.binding
    ) attributes {translation_info = #translation} {
      %c0 = arith.constant 0 : index

      %dispS = stream.binding.subspan %argS[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x128x32xf32>>
      %dispV = stream.binding.subspan %argV[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x128x32x4x32xf4E2M1FN>>
      %dispVS = stream.binding.subspan %argVS[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x128x4x32xf8E8M0FNU>>
      %dispPS = stream.binding.subspan %argPS[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x128xf8E8M0FNU>>
      %dispR = stream.binding.subspan %ret[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x128xf32>>

      %S = iree_tensor_ext.dispatch.tensor.load %dispS, offsets = [0,0,0,0,0], sizes = [4,32,4096,128,32], strides = [1,1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x128x32xf32>> -> tensor<4x32x4096x128x32xf32>
      %V = iree_tensor_ext.dispatch.tensor.load %dispV, offsets = [0,0,0,0,0,0], sizes = [4,32,128,32,4,32], strides = [1,1,1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x128x32x4x32xf4E2M1FN>> -> tensor<4x32x128x32x4x32xf4E2M1FN>
      %VS = iree_tensor_ext.dispatch.tensor.load %dispVS, offsets = [0,0,0,0,0], sizes = [4,32,128,4,32], strides = [1,1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x128x4x32xf8E8M0FNU>> -> tensor<4x32x128x4x32xf8E8M0FNU>
      %PS = iree_tensor_ext.dispatch.tensor.load %dispPS, offsets = [0,0,0,0], sizes = [4,32,4096,128], strides = [1,1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x4096x128xf8E8M0FNU>> -> tensor<4x32x4096x128xf8E8M0FNU>

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

      %red_empty = tensor.empty() : tensor<4x32x4096x4x32xf32>
      %acc_init = linalg.fill ins(%sum_el : f32)
                              outs(%red_empty : tensor<4x32x4096x4x32xf32>)
                              -> tensor<4x32x4096x4x32xf32>

      %MAX, %SUM, %PV = iree_linalg_ext.exp_reduction {
        indexing_maps = [
          affine_map<(B, H, M, DG, DB, KG, KB) -> (B, H, M, KG, KB)>,
          affine_map<(B, H, M, DG, DB, KG, KB) -> (B, H, KG, KB, DG, DB)>,
          affine_map<(B, H, M, DG, DB, KG, KB) -> (B, H, M, KG)>,
          affine_map<(B, H, M, DG, DB, KG, KB) -> (B, H, KG, DG, DB)>,
          affine_map<(B, H, M, DG, DB, KG, KB) -> (B, H, M)>,
          affine_map<(B, H, M, DG, DB, KG, KB) -> (B, H, M)>,
          affine_map<(B, H, M, DG, DB, KG, KB) -> (B, H, M, DG, DB)>
        ],
        iterator_types = [
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<reduction>,
          #iree_linalg_ext.iterator_type<reduction>
        ],
        exp_reduced_operands = [1, 2]
      }
        attributes {lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 64, 1, 0, 0, 0],
          reduction = [0, 0, 0, 0, 0, 32, 32],
          subgroup_basis = [[1, 1, 2, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6]],
          mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
        }>}
        ins(%S, %V, %PS, %VS : tensor<4x32x4096x128x32xf32>, tensor<4x32x128x32x4x32xf4E2M1FN>, tensor<4x32x4096x128xf8E8M0FNU>, tensor<4x32x128x4x32xf8E8M0FNU>)
        outs(%max_init, %sum_init, %acc_init : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x4x32xf32>)
      {
      ^bb0(%ex : f32, %v : f4E2M1FN, %ex_scale : f8E8M0FNU, %v_scale : f8E8M0FNU, %m : f32, %sum : f32, %acc : f32):
        %fp4max = arith.constant 6.0 : f32
        %ex4m = arith.mulf %ex, %fp4max : f32
        %ex_trunc = arith.scaling_truncf %ex4m, %ex_scale : f32, f8E8M0FNU to f4E2M1FN
        %ex_ext = arith.scaling_extf %ex_trunc, %ex_scale : f4E2M1FN, f8E8M0FNU to f32
        %v_ext = arith.scaling_extf %v, %v_scale : f4E2M1FN, f8E8M0FNU to f32
        %nsum = arith.addf %ex, %sum : f32
        %mul  = arith.mulf %ex_ext, %v_ext : f32
        %nacc = arith.addf %mul, %acc : f32
        iree_linalg_ext.yield %m, %nsum, %nacc : f32, f32, f32
      } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x4x32xf32>

      %result_flat = tensor.collapse_shape %PV [[0], [1], [2], [3, 4]] : tensor<4x32x4096x4x32xf32> into tensor<4x32x4096x128xf32>
      iree_tensor_ext.dispatch.tensor.store %result_flat, %dispR, offsets = [0,0,0,0], sizes = [4,32,4096,128], strides = [1,1,1,1] : tensor<4x32x4096x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x4096x128xf32>>
      return
    }
  }
}

func.func @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view) -> !hal.buffer_view {
  %c = arith.constant 1 : index
  %s = hal.tensor.import %arg0 "s" : !hal.buffer_view -> tensor<4x32x4096x128x32xf32>
  %v_bytes = hal.tensor.import %arg1 "v" : !hal.buffer_view -> tensor<4x32x4096x64xi8>
  %v_scale_bytes = hal.tensor.import %arg2 "v_scale" : !hal.buffer_view -> tensor<4x32x128x4x32xi8>
  %p_scale_bytes = hal.tensor.import %arg3 "p_scale" : !hal.buffer_view -> tensor<4x32x4096x128xi8>
  %v = iree_tensor_ext.bitcast %v_bytes : tensor<4x32x4096x64xi8> -> tensor<4x32x128x32x4x32xf4E2M1FN>
  %v_scale = iree_tensor_ext.bitcast %v_scale_bytes : tensor<4x32x128x4x32xi8> -> tensor<4x32x128x4x32xf8E8M0FNU>
  %p_scale = iree_tensor_ext.bitcast %p_scale_bytes : tensor<4x32x4096x128xi8> -> tensor<4x32x4096x128xf8E8M0FNU>

  %ret0 = flow.dispatch @executable_0::@dispatch[%c](%s, %v, %v_scale, %p_scale) : (tensor<4x32x4096x128x32xf32>, tensor<4x32x128x32x4x32xf4E2M1FN>, tensor<4x32x128x4x32xf8E8M0FNU>, tensor<4x32x4096x128xf8E8M0FNU>) -> tensor<4x32x4096x128xf32>

  %ret = hal.tensor.export %ret0 "out" : tensor<4x32x4096x128xf32> -> !hal.buffer_view
  return %ret : !hal.buffer_view
}

} // module
