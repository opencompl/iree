#config = #iree_cpu.lowering_config<distribution = [32, 16, 0, 0], vector_common_parallel = [0, 0, 0, 32]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @attention(%arg0: tensor<20x4096x64xf16>, %arg1: tensor<20x4096x64xf16>, %arg2: tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16> {
    %c16 = arith.constant 16 : index
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<20x4096x4096xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<20x4096x4096xf32>) -> tensor<20x4096x4096xf32>
    %2 = scf.for %arg3 = %c0 to %c4096 step %c16 iter_args(%arg4 = %1) -> (tensor<20x4096x4096xf32>) {
      %extracted_slice = tensor.extract_slice %arg4[0, %arg3, 0] [20, 16, 4096] [1, 1, 1] : tensor<20x4096x4096xf32> to tensor<20x16x4096xf32>
      %extracted_slice_2 = tensor.extract_slice %arg0[0, %arg3, 0] [20, 16, 64] [1, 1, 1] : tensor<20x4096x64xf16> to tensor<20x16x64xf16>
      %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice_2, %arg1 : tensor<20x16x64xf16>, tensor<20x4096x64xf16>) outs(%extracted_slice : tensor<20x16x4096xf32>) attrs =  {attention_qk_matmul, lowering_config = #config} {
      ^bb0(%in: f16, %in_3: f16, %out: f32):
        %13 = arith.extf %in : f16 to f32
        %14 = arith.extf %in_3 : f16 to f32
        %15 = arith.mulf %13, %14 : f32
        %16 = arith.addf %15, %out : f32
        linalg.yield %16 : f32
      } -> tensor<20x16x4096xf32>
      %inserted_slice = tensor.insert_slice %12 into %arg4[0, %arg3, 0] [20, 16, 4096] [1, 1, 1] : tensor<20x16x4096xf32> into tensor<20x4096x4096xf32>
      scf.yield %inserted_slice : tensor<20x4096x4096xf32>
    }
    %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<20x4096x4096xf32>) outs(%0 : tensor<20x4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.mulf %in, %cst_1 : f32
      linalg.yield %12 : f32
    } -> tensor<20x4096x4096xf32>
    %4 = tensor.empty() : tensor<20x4096xf32>
    %5 = tensor.empty() : tensor<20x4096x64xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%4 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<20x4096x64xf32>) -> tensor<20x4096x64xf32>
    %9:3 = scf.for %arg3 = %c0 to %c4096 step %c16 iter_args(%arg4 = %6, %arg5 = %7, %arg6 = %8) -> (tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %3[0, %arg3, 0] [20, 16, 4096] [1, 1, 1] : tensor<20x4096x4096xf32> to tensor<20x16x4096xf32>
      %extracted_slice_2 = tensor.extract_slice %arg4[0, %arg3] [20, 16] [1, 1] : tensor<20x4096xf32> to tensor<20x16xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[0, %arg3] [20, 16] [1, 1] : tensor<20x4096xf32> to tensor<20x16xf32>
      %extracted_slice_4 = tensor.extract_slice %arg6[0, %arg3, 0] [20, 16, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<20x16x64xf32>
      %12 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<20x16x4096xf32>) outs(%extracted_slice_2 : tensor<20x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.maximumf %in, %out : f32
        linalg.yield %21 : f32
      } -> tensor<20x16xf32>
      %13 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<20x16xf32>) outs(%extracted_slice : tensor<20x16x4096xf32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.subf %out, %in : f32
        %22 = math.exp2 %21 : f32
        linalg.yield %22 : f32
      } -> tensor<20x16x4096xf32>
      %14 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<20x16xf32>) outs(%extracted_slice_2 : tensor<20x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.subf %out, %in : f32
        %22 = math.exp2 %21 : f32
        linalg.yield %22 : f32
      } -> tensor<20x16xf32>
      %15 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<20x16xf32>) outs(%extracted_slice_3 : tensor<20x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.mulf %in, %out : f32
        linalg.yield %21 : f32
      } -> tensor<20x16xf32>
      %16 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<20x16xf32>) outs(%extracted_slice_4 : tensor<20x16x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.mulf %in, %out : f32
        linalg.yield %21 : f32
      } -> tensor<20x16x64xf32>
      %17 = linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel"]} outs(%12 : tensor<20x16xf32>) {
      ^bb0(%out: f32):
        linalg.yield %out : f32
      } -> tensor<20x16xf32>
      %18 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13 : tensor<20x16x4096xf32>) outs(%15 : tensor<20x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %21 = arith.addf %in, %out : f32
        linalg.yield %21 : f32
      } -> tensor<20x16xf32>
      %19 = linalg.generic {indexing_maps = [#map, #map6, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%13, %arg2 : tensor<20x16x4096xf32>, tensor<20x4096x64xf16>) outs(%16 : tensor<20x16x64xf32>) {
      ^bb0(%in: f32, %in_7: f16, %out: f32):
        %21 = arith.truncf %in : f32 to f16
        %22 = arith.extf %21 : f16 to f32
        %23 = arith.extf %in_7 : f16 to f32
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %24, %out : f32
        linalg.yield %25 : f32
      } -> tensor<20x16x64xf32>
      %20:3 = linalg.generic {indexing_maps = [#map, #map6, #map7, #map7, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%13, %arg2 : tensor<20x16x4096xf32>, tensor<20x4096x64xf16>) outs(%12, %15, %16 : tensor<20x16xf32>, tensor<20x16xf32>, tensor<20x16x64xf32>) {
      ^bb0(%in: f32, %in_7: f16, %out: f32, %out_8: f32, %out_9: f32):
        %21 = arith.addf %in, %out_8 : f32
        %22 = arith.truncf %in : f32 to f16
        %23 = arith.extf %22 : f16 to f32
        %24 = arith.extf %in_7 : f16 to f32
        %25 = arith.mulf %23, %24 : f32
        %26 = arith.addf %25, %out_9 : f32
        linalg.yield %out, %21, %26 : f32, f32, f32
      } -> (tensor<20x16xf32>, tensor<20x16xf32>, tensor<20x16x64xf32>)
      %inserted_slice = tensor.insert_slice %17 into %arg4[0, %arg3] [20, 16] [1, 1] : tensor<20x16xf32> into tensor<20x4096xf32>
      %inserted_slice_5 = tensor.insert_slice %18 into %arg5[0, %arg3] [20, 16] [1, 1] : tensor<20x16xf32> into tensor<20x4096xf32>
      %inserted_slice_6 = tensor.insert_slice %19 into %arg6[0, %arg3, 0] [20, 16, 64] [1, 1, 1] : tensor<20x16x64xf32> into tensor<20x4096x64xf32>
      scf.yield %inserted_slice, %inserted_slice_5, %inserted_slice_6 : tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>
    }
    %10 = tensor.empty() : tensor<20x4096x64xf16>
    %11 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9#2, %9#1 : tensor<20x4096x64xf32>, tensor<20x4096xf32>) outs(%10 : tensor<20x4096x64xf16>) {
    ^bb0(%in: f32, %in_2: f32, %out: f16):
      %12 = arith.divf %in, %in_2 : f32
      %13 = arith.truncf %12 : f32 to f16
      linalg.yield %13 : f16
    } -> tensor<20x4096x64xf16>
    return %11 : tensor<20x4096x64xf16>
  }
}
