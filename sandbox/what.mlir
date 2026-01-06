func.func @dispatch() attributes {translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<20x4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<20x4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<20x4096x64xf32>
  %7 = tensor.empty() : tensor<20x4096x64xf32>
  %8 = tensor.empty() : tensor<20x4096xf32>
  %9:2 = scf.forall (%arg0, %arg1) = (0, 0) to (20, 4096) step (1, 32) shared_outs(%arg2 = %8, %arg3 = %7) -> (tensor<20x4096xf32>, tensor<20x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1] [1, 32] [1, 1] : tensor<20x4096xf32> to tensor<1x32xf32>
    %extracted_slice_1 = tensor.extract_slice %arg3[%arg0, %arg1, 0] [1, 32, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x32x64xf32>
    %11:2 = scf.forall (%arg4) = (0) to (32) step (4) shared_outs(%arg5 = %extracted_slice, %arg6 = %extracted_slice_1) -> (tensor<1x32xf32>, tensor<1x32x64xf32>) {
      %12 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg4, %arg1]
      %extracted_slice_2 = tensor.extract_slice %4[%arg0, %12, 0] [1, 4, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x4x64xf32>
      %13 = tensor.empty() : tensor<1x4xf32>
      %14 = linalg.fill ins(%cst_0 : f32) outs(%13 : tensor<1x4xf32>) -> tensor<1x4xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[0, %arg4] [1, 4] [1, 1] : tensor<1x32xf32> to tensor<1x4xf32>
      %15 = linalg.fill ins(%cst : f32) outs(%extracted_slice_3 : tensor<1x4xf32>) -> tensor<1x4xf32>
      %extracted_slice_4 = tensor.extract_slice %arg6[0, %arg4, 0] [1, 4, 64] [1, 1, 1] : tensor<1x32x64xf32> to tensor<1x4x64xf32>
      %16 = linalg.fill ins(%cst : f32) outs(%extracted_slice_4 : tensor<1x4x64xf32>) -> tensor<1x4x64xf32>
      %17:3 = scf.for %arg7 = %c0 to %c4096 step %c8 iter_args(%arg8 = %14, %arg9 = %15, %arg10 = %16) -> (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>) {
        %extracted_slice_6 = tensor.extract_slice %5[%arg0, %arg7, 0] [1, 8, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x8x64xf32>
        %18 = tensor.empty() : tensor<1x4x8xf32>
        %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
        %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice_2, %extracted_slice_6 : tensor<1x4x64xf32>, tensor<1x8x64xf32>) outs(%19 : tensor<1x4x8xf32>) {
        ^bb0(%in: f32, %in_9: f32, %out: f32):
          %33 = arith.mulf %in, %in_9 : f32
          %34 = arith.addf %33, %out : f32
          linalg.yield %34 : f32
        } -> tensor<1x4x8xf32>
        %extracted_slice_7 = tensor.extract_slice %6[%arg0, %arg7, 0] [1, 8, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x8x64xf32>
        %cst_8 = arith.constant 1.44269502 : f32
        %21 = tensor.empty() : tensor<1x4x8xf32>
        %22 = linalg.fill ins(%cst_8 : f32) outs(%21 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
        %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<1x4x8xf32>) outs(%22 : tensor<1x4x8xf32>) {
        ^bb0(%in: f32, %out: f32):
          %33 = arith.mulf %in, %out : f32
          linalg.yield %33 : f32
        } -> tensor<1x4x8xf32>
        %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%23 : tensor<1x4x8xf32>) outs(%arg8 : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %33 = arith.maximumf %in, %out : f32
          linalg.yield %33 : f32
        } -> tensor<1x4xf32>
        %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<1x4xf32>) outs(%23 : tensor<1x4x8xf32>) {
        ^bb0(%in: f32, %out: f32):
          %33 = arith.subf %out, %in : f32
          %34 = math.exp2 %33 : f32
          linalg.yield %34 : f32
        } -> tensor<1x4x8xf32>
        %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%24 : tensor<1x4xf32>) outs(%arg8 : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %33 = arith.subf %out, %in : f32
          %34 = math.exp2 %33 : f32
          linalg.yield %34 : f32
        } -> tensor<1x4xf32>
        %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<1x4xf32>) outs(%arg9 : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %33 = arith.mulf %in, %out : f32
          linalg.yield %33 : f32
        } -> tensor<1x4xf32>
        %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26 : tensor<1x4xf32>) outs(%arg10 : tensor<1x4x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %33 = arith.mulf %in, %out : f32
          linalg.yield %33 : f32
        } -> tensor<1x4x64xf32>
        %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%24 : tensor<1x4xf32>) {
        ^bb0(%out: f32):
          linalg.yield %out : f32
        } -> tensor<1x4xf32>
        %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25 : tensor<1x4x8xf32>) outs(%27 : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %33 = arith.addf %in, %out : f32
          linalg.yield %33 : f32
        } -> tensor<1x4xf32>
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%25, %extracted_slice_7 : tensor<1x4x8xf32>, tensor<1x8x64xf32>) outs(%28 : tensor<1x4x64xf32>) {
        ^bb0(%in: f32, %in_9: f32, %out: f32):
          %33 = arith.mulf %in, %in_9 : f32
          %34 = arith.addf %33, %out : f32
          linalg.yield %34 : f32
        } -> tensor<1x4x64xf32>
        %32:3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%25, %extracted_slice_7 : tensor<1x4x8xf32>, tensor<1x8x64xf32>) outs(%24, %27, %28 : tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>) {
        ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32, %out_11: f32):
          %33 = arith.addf %in, %out_10 : f32
          %34 = arith.mulf %in, %in_9 : f32
          %35 = arith.addf %34, %out_11 : f32
          linalg.yield %out, %33, %35 : f32, f32, f32
        } -> (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>)
        scf.yield %29, %30, %31 : tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>
      }
      %cast = tensor.cast %17#1 : tensor<1x4xf32> to tensor<?x4xf32>
      %cast_5 = tensor.cast %17#2 : tensor<1x4x64xf32> to tensor<?x4x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cast into %arg5[%c0, %arg4] [%c1, 4] [1, 1] : tensor<?x4xf32> into tensor<1x32xf32>
        tensor.parallel_insert_slice %cast_5 into %arg6[%c0, %arg4, 0] [%c1, 4, 64] [1, 1, 1] : tensor<?x4x64xf32> into tensor<1x32x64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11#0 into %arg2[%arg0, %arg1] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<20x4096xf32>
      tensor.parallel_insert_slice %11#1 into %arg3[%arg0, %arg1, 0] [1, 32, 64] [1, 1, 1] : tensor<1x32x64xf32> into tensor<20x4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9#0 : tensor<20x4096xf32>) outs(%9#1 : tensor<20x4096x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %11 = arith.divf %out, %in : f32
    linalg.yield %11 : f32
  } -> tensor<20x4096x64xf32>
  iree_codegen.store_to_buffer %10, %3 : tensor<20x4096x64xf32> into memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}
