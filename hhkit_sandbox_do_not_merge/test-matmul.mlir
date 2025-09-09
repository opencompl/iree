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
          %ret: !stream.binding)
          attributes { translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
      %cst0 = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index
      %dispQ = stream.binding.subspan %argQ[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
      %dispK = stream.binding.subspan %argK[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
      %dispR = stream.binding.subspan %ret[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>

      %4 = iree_tensor_ext.dispatch.tensor.load %dispQ, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>> -> tensor<4096x64xf32>
      %5 = iree_tensor_ext.dispatch.tensor.load %dispK, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>> -> tensor<4096x64xf32>

      %S_empty = tensor.empty() : tensor<4096x4096xf32>

      %result = linalg.generic  {
          indexing_maps = [
            affine_map<(M, K2, K1) -> (M, K1)>,
            affine_map<(M, K2, K1) -> (K2, K1)>,
            affine_map<(M, K2, K1) -> (M, K2)>
          ],
          iterator_types = ["parallel", "parallel", "reduction"]
        }
        {lowering_config = #iree_cpu.lowering_config<
          distribution = [1, 64, 64],
          vector_common_parallel = [0, 8, 8],
          vector_reduction = [0, 0, 16]
        >}
        ins(%4, %5 : tensor<4096x64xf32>, tensor<4096x64xf32>)
        outs(%S_empty : tensor<4096x4096xf32>)
      {
      ^bb0(%q : f32, %k : f32, %s : f32):
        %mul  = arith.mulf %q, %k : f32
        %sum  = arith.addf %mul, %s : f32
        linalg.yield %sum : f32
      } -> tensor<4096x4096xf32>

      iree_tensor_ext.dispatch.tensor.store  %result, %dispR, offsets = [0, 0], sizes = [ 4096, 4096], strides = [1, 1]
        : tensor<4096x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
      return
    }
  }
}


func.func @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view {
  %c = arith.constant 1 : index
  %0 = hal.tensor.import %arg0 "q" : !hal.buffer_view -> tensor<4096x64xf32>
  %1 = hal.tensor.import %arg1 "k" : !hal.buffer_view -> tensor<4096x64xf32>

  %ret0 = flow.dispatch @executable_0::@dispatch[%c](%0, %1) : (tensor<4096x64xf32>, tensor<4096x64xf32>) ->  tensor<4096x4096xf32>

  %3 = hal.tensor.export %ret0 "input1" : tensor<4096x4096xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}
}
