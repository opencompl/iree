module {
  util.func public @attention(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @attention(%input0: tensor<20x4096x64xf16>, %input1: tensor<20x4096x64xf16>, %input2: tensor<20x4096x64xf16>) -> (%output0: tensor<20x4096x64xf16>)"}} {
    %cst = arith.constant 1.250000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<20x4096x64xf16>
    %1 = iree_tensor_ext.compute_barrier.start %0 : tensor<20x4096x64xf16> -> tensor<20x4096x64xf16>
    %2 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<20x4096x64xf16>
    %3 = iree_tensor_ext.compute_barrier.start %2 : tensor<20x4096x64xf16> -> tensor<20x4096x64xf16>
    %4 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<20x4096x64xf16>
    %5 = iree_tensor_ext.compute_barrier.start %4 : tensor<20x4096x64xf16> -> tensor<20x4096x64xf16>
    %6 = tensor.empty() : tensor<20x4096x4096xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<20x4096x4096xf32>) -> tensor<20x4096x4096xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %3 : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>) outs(%7 : tensor<20x4096x4096xf32>) attrs =  {attention_qk_matmul, lowering_config = #iree_cpu.lowering_config<distribution = [32, 16, 0, 0], vector_common_parallel = [0, 0, 0, 32]>} {
    ^bb0(%in: f16, %in_2: f16, %out: f32):
      %20 = arith.extf %in : f16 to f32
      %21 = arith.extf %in_2 : f16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %22, %out : f32
      linalg.yield %23 : f32
    } -> tensor<20x4096x4096xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<20x4096x4096xf32>) outs(%6 : tensor<20x4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %20 = arith.mulf %in, %cst : f32
      linalg.yield %20 : f32
    } -> tensor<20x4096x4096xf32>
    %10 = tensor.empty() : tensor<20x4096xf32>
    %11 = tensor.empty() : tensor<20x4096x64xf32>
    %12 = linalg.fill ins(%cst_1 : f32) outs(%10 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
    %14 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<20x4096x64xf32>) -> tensor<20x4096x64xf32>
    %15:3 = iree_linalg_ext.exp_reduction{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>], exp_reduced_operands = [1, 2]} attributes {attention_pv_matmul, lowering_config = #iree_cpu.lowering_config<distribution = [32, 16, 0, 0]>} ins(%9, %5 : tensor<20x4096x4096xf32>, tensor<20x4096x64xf16>) outs(%12, %13, %14 : tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>) {
    ^bb0(%arg3: f32, %arg4: f16, %arg5: f32, %arg6: f32, %arg7: f32):
      %20 = arith.addf %arg3, %arg6 : f32
      %21 = arith.truncf %arg3 : f32 to f16
      %22 = arith.extf %21 : f16 to f32
      %23 = arith.extf %arg4 : f16 to f32
      %24 = arith.mulf %22, %23 : f32
      %25 = arith.addf %24, %arg7 : f32
      iree_linalg_ext.yield %arg5, %20, %25 : f32, f32, f32
    } -> tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>
    %16 = tensor.empty() : tensor<20x4096x64xf16>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15#2, %15#1 : tensor<20x4096x64xf32>, tensor<20x4096xf32>) outs(%16 : tensor<20x4096x64xf16>) {
    ^bb0(%in: f32, %in_2: f32, %out: f16):
      %20 = arith.divf %in, %in_2 : f32
      %21 = arith.truncf %20 : f32 to f16
      linalg.yield %21 : f16
    } -> tensor<20x4096x64xf16>
    %18 = iree_tensor_ext.compute_barrier.end %17 : tensor<20x4096x64xf16> -> tensor<20x4096x64xf16>
    %19 = hal.tensor.export %18 "output0" : tensor<20x4096x64xf16> -> !hal.buffer_view
    util.return %19 : !hal.buffer_view
  }
}
