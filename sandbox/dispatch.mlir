module {
  util.func public @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @attention(%input0: tensor<20x4096x64xf16>, %input1: tensor<20x4096x64xf16>, %input2: tensor<20x4096x64xf16>) -> (%output0: tensor<20x4096x64xf32>)"}} {
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<20x4096x64xf16>
    %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<20x4096x64xf16>
    %2 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<20x4096x64xf16>
    %3 = flow.dispatch.workgroups(%0, %1, %2) : (tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>) -> tensor<20x4096x64xf32> =
        (%arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x4096x4096xf32>>, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x4096x64xf32>>) {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 1.250000e-01 : f32
      %10 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
      %11 = iree_tensor_ext.dispatch.tensor.load %arg4, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
      %a11 = iree_tensor_ext.dispatch.tensor.load %arg4, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
      %12 = tensor.empty() : tensor<20x4096x4096xf32>
      %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<20x4096x4096xf32>) -> tensor<20x4096x4096xf32>
      %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%10, %11 : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>) outs(%13 : tensor<20x4096x4096xf32>) attrs =  {attention_qk_matmul, lowering_config = #iree_cpu.lowering_config<distribution = [32, 16, 0, 0], vector_common_parallel = [0, 0, 0, 32]>} {
      ^bb0(%in: f16, %in_1: f16, %out: f32):
        %16 = arith.extf %in : f16 to f32
        %17 = arith.extf %in_1 : f16 to f32
        %18 = arith.mulf %16, %17 : f32
        %19 = arith.addf %18, %out : f32
        linalg.yield %19 : f32
      } -> tensor<20x4096x4096xf32>
      %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<20x4096x4096xf32>) outs(%12 : tensor<20x4096x4096xf32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = arith.mulf %in, %cst_0 : f32
        linalg.yield %16 : f32
      } -> tensor<20x4096x4096xf32>
      %22 = tensor.empty() : tensor<20x4096x64xf32>
      %23 = tensor.empty() : tensor<20x4096xf32>
      %24 = linalg.fill ins(%cst_0 : f32) outs(%22 : tensor<20x4096x64xf32>) -> tensor<20x4096x64xf32>
      %25 = linalg.fill ins(%cst : f32) outs(%23 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
      %26 = linalg.fill ins(%cst_0 : f32) outs(%23 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
      %27:3 = iree_linalg_ext.exp_reduction{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>], exp_reduced_operands = [1, 2]} attributes {attention_pv_matmul, lowering_config = #iree_cpu.lowering_config<distribution = [32, 16, 0, 0]>}
        ins(%15, %a11 : tensor<20x4096x4096xf32>, tensor<20x4096x64xf16>)
        outs(%25, %26, %24 : tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>) {
      ^bb0(%arg7: f32, %arg8: f16, %arg9: f32, %arg10: f32, %arg11: f32):
        %28 = arith.addf %arg7, %arg10 : f32
        %29 = arith.truncf %arg7 : f32 to f16
        %30 = arith.extf %29 : f16 to f32
        %31 = arith.extf %arg8 : f16 to f32
        %32 = arith.mulf %30, %31 : f32
        %33 = arith.addf %32, %arg11 : f32
        iree_linalg_ext.yield %arg9, %28, %33 : f32, f32, f32
      } -> tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>


       iree_tensor_ext.dispatch.tensor.store %27#2, %arg6, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x4096x64xf32>>
      flow.return
    } count() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    %9 = hal.tensor.export %3 "output0" : tensor<20x4096x64xf32> -> !hal.buffer_view
    util.return %9 : !hal.buffer_view
  }
}
