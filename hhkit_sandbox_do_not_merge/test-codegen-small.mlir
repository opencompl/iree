// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmcpu-lower-executable-target))' --mlir-print-after-all %s > out

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
    ) attributes {translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
      %cst0 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index

      %dispQ = stream.binding.subspan %argQ[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4xf32>>
      %dispK = stream.binding.subspan %argK[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4xf32>>
      %dispV = stream.binding.subspan %argV[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4xf32>>
      %dispR = stream.binding.subspan %ret[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x4xf32>>

      %Q = iree_tensor_ext.dispatch.tensor.load %dispQ, offsets = [0,0,0], sizes = [1,4,4], strides = [1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4xf32>> -> tensor<1x4x4xf32>
      %K = iree_tensor_ext.dispatch.tensor.load %dispK, offsets = [0,0,0], sizes = [1,4,4], strides = [1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4xf32>> -> tensor<1x4x4xf32>
      %V = iree_tensor_ext.dispatch.tensor.load %dispV, offsets = [0,0,0], sizes = [1,4,4], strides = [1,1,1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4xf32>> -> tensor<1x4x4xf32>

      %S_empty = tensor.empty() : tensor<1x4x4xf32>
      %S_fill  = linalg.fill ins(%cst0 : f32)
                              outs(%S_empty : tensor<1x4x4xf32>)
                              -> tensor<1x4x4xf32>

      %S = linalg.generic  {
          indexing_maps = [
            affine_map<(B, M, K2, K1) -> (B, M, K1)>,
            affine_map<(B, M, K2, K1) -> (B, K2, K1)>,
            affine_map<(B, M, K2, K1) -> (B, M, K2)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]
        }
        ins(%Q, %K : tensor<1x4x4xf32>, tensor<1x4x4xf32>)
        outs(%S_fill : tensor<1x4x4xf32>)
      {
      ^bb0(%q : f32, %k : f32, %s : f32):
        %mul  = arith.mulf %q, %k : f32
        %sum  = arith.addf %mul, %s : f32
        linalg.yield %sum : f32
      } -> tensor<1x4x4xf32>

      %red_empty = tensor.empty() : tensor<1x4x4xf32>
      %max_empty = tensor.empty() : tensor<1x4xf32>

      %max_el = arith.constant -3.40282347E+38 : f32
      %max_init = linalg.fill ins(%max_el : f32)
                              outs(%max_empty : tensor<1x4xf32>)
                              -> tensor<1x4xf32>

      %sum_empty = tensor.empty() : tensor<1x4xf32>
      %sum_el = arith.constant 0.000000e+00 : f32
      %sum_init = linalg.fill ins(%sum_el : f32)
                              outs(%sum_empty : tensor<1x4xf32>)
                              -> tensor<1x4xf32>
      %acc_init = linalg.fill ins(%sum_el : f32)
                              outs(%red_empty : tensor<1x4x4xf32>)
                              -> tensor<1x4x4xf32>

      %MAX, %SUM, %PV = iree_linalg_ext.exp_reduction {
        indexing_maps = [
          affine_map<(B, M, N, K2) -> (B, M, K2)>,
          affine_map<(B, M, N, K2) -> (B, K2, N)>,
          affine_map<(B, M, N, K2) -> (B, M)>,
          affine_map<(B, M, N, K2) -> (B, M)>,
          affine_map<(B, M, N, K2) -> (B, M, N)>
        ],
        iterator_types = [
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<parallel>,
          #iree_linalg_ext.iterator_type<reduction>
        ],
        exp_reduced_operands = [1, 2]
      }
        {lowering_config = #iree_cpu.lowering_config<
          distribution = [1, 32, 32, 0],
          vector_common_parallel = [0, 8, 8, 8],
          vector_reduction = [0, 0, 0, 16]
        >}
        ins(%S, %V : tensor<1x4x4xf32>, tensor<1x4x4xf32>)
        outs(%max_init, %sum_init, %acc_init : tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x4xf32>)
      {
      ^bb0(%ex : f32, %v : f32, %m : f32, %sum : f32, %acc : f32):
        %nsum = arith.addf %ex, %sum : f32
        %mul  = arith.mulf %ex, %v : f32
        %nacc = arith.addf %mul, %acc : f32
        linalg.yield %m, %nsum, %nacc : f32, f32, f32
      } -> tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x4xf32>

      %result_empty = tensor.empty() : tensor<1x4x4xf32>

      // %result = linalg.generic{
      //   indexing_maps = [
      //     affine_map<(B,M,N) -> (B,M)>,
      //     affine_map<(B,M,N) -> (B,M,N)>
      //   ],
      //   iterator_types= ["parallel", "parallel", "parallel"]
      // } ins(%MAX: tensor<1x4xf32>)
      //   outs(%result_empty : tensor<1x4x4xf32>) {
      //     ^bb0(%m: f32, %o: f32):
      //       linalg.yield %m : f32
      //   } -> tensor<1x4x4xf32>

      %result = linalg.generic {
                  indexing_maps = [
                    affine_map<(B, M, N) -> (B, M, N)>,
                    affine_map<(B, M, N) -> (B, M)>,
                    affine_map<(B, M, N) -> (B, M, N)>
                  ],
                  iterator_types = ["parallel", "parallel", "parallel"]
                }
                ins(%PV, %SUM : tensor<1x4x4xf32>, tensor<1x4xf32>)
                outs(%result_empty : tensor<1x4x4xf32>) {
      ^bb0(%pv : f32, %sum : f32, %res : f32):
        %out = arith.divf %pv, %sum : f32
        linalg.yield %out : f32
      } -> tensor<1x4x4xf32>

      iree_tensor_ext.dispatch.tensor.store %result, %dispR, offsets = [0,0,0], sizes = [1,4,4], strides = [1,1,1] : tensor<1x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x4xf32>>
      return
    }
  }
}

func.func @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view {
  %c = arith.constant 1 : index
  %0 = hal.tensor.import %arg0 "q" : !hal.buffer_view -> tensor<1x4x4xf32>
  %1 = hal.tensor.import %arg1 "k" : !hal.buffer_view -> tensor<1x4x4xf32>
  %2 = hal.tensor.import %arg2 "v" : !hal.buffer_view -> tensor<1x4x4xf32>

  %ret0 = flow.dispatch @executable_0::@dispatch[%c](%0, %1, %2) : (tensor<1x4x4xf32>, tensor<1x4x4xf32>, tensor<1x4x4xf32>) ->  tensor<1x4x4xf32>

  %3 = hal.tensor.export %ret0 "out" : tensor<1x4x4xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

} // module
