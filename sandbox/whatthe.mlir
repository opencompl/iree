
func.func @dispatch() attributes {translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c4 = arith.constant 4 : index
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
  %8 = scf.forall (%arg0, %arg1) = (0, 0) to (20, 4096) step (1, 32) shared_outs(%arg2 = %7) -> (tensor<20x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1, 0] [1, 32, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x32x64xf32>
    %9 = scf.forall (%arg3) = (0) to (32) step (4) shared_outs(%arg4 = %extracted_slice) -> (tensor<1x32x64xf32>) {
      %10 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg3, %arg1]
      %extracted_slice_1 = tensor.extract_slice %4[%arg0, %10, 0] [1, 4, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x4x64xf32>
      %11 = tensor.empty() : tensor<1x4xf32>
      %neginf = linalg.fill
        ins(%cst_0 : f32) outs(%11 : tensor<1x4xf32>) -> tensor<1x4xf32>
      %13 = tensor.empty() : tensor<1x4xf32>
      %zero_acc = linalg.fill
        ins(%cst : f32) outs(%13 : tensor<1x4xf32>) -> tensor<1x4xf32>
      %15 = tensor.empty() : tensor<1x4x64xf32>
      %zero_sum = linalg.fill
        ins(%cst : f32) outs(%15 : tensor<1x4x64xf32>) -> tensor<1x4x64xf32>
      %17:3 = scf.for %arg5 = %c0 to %c4096 step %c4 iter_args(%prev_max = %neginf, %prev_acc = %zero_acc, %prev_pv_sum = %zero_sum) -> (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>) {
        %extracted_slice_3 = tensor.extract_slice %5[%arg0, %arg5, 0] [1, 4, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x4x64xf32>
        %19 = tensor.empty() : tensor<1x4x4xf32>
        %zero_sum_qk = linalg.fill
          ins(%cst : f32) outs(%19 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
        %Smat = linalg.generic {indexing_maps = [affine_map<(B, d1, d2, d3) -> (B, d1, d3)>, affine_map<(B, d1, d2, d3) -> (B, d2, d3)>, affine_map<(B, d1, d2, d3) -> (B, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%extracted_slice_1, %extracted_slice_3 : tensor<1x4x64xf32>, tensor<1x4x64xf32>) outs(%zero_sum_qk : tensor<1x4x4xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %31 = arith.mulf %in, %in_5 : f32
          %32 = arith.addf %31, %out : f32
          linalg.yield %32 : f32
        } -> tensor<1x4x4xf32>
        %extracted_slice_4 = tensor.extract_slice %6[%arg0, %arg5, 0] [1, 4, 64] [1, 1, 1] : tensor<20x4096x64xf32> to tensor<1x4x64xf32>
        %curr_max = linalg.generic {indexing_maps = [affine_map<(B, d1, d2) -> (B, d1, d2)>, affine_map<(B, d1, d2) -> (B, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%Smat : tensor<1x4x4xf32>) outs(%prev_max : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %31 = arith.maximumf %in, %out : f32
          linalg.yield %31 : f32
        } -> tensor<1x4xf32>
        %s_minus_max = linalg.generic {indexing_maps = [affine_map<(B, d1, d2) -> (B, d1)>, affine_map<(B, d1, d2) -> (B, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%curr_max : tensor<1x4xf32>) outs(%Smat : tensor<1x4x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %31 = arith.subf %out, %in : f32
          %32 = math.exp2 %31 : f32
          linalg.yield %32 : f32
        } -> tensor<1x4x4xf32>
        %norm = linalg.generic {indexing_maps = [affine_map<(B, d1) -> (B, d1)>, affine_map<(B, d1) -> (B, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%curr_max : tensor<1x4xf32>) outs(%prev_max : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %31 = arith.subf %out, %in : f32
          %32 = math.exp2 %31 : f32
          linalg.yield %32 : f32
        } -> tensor<1x4xf32>
        %pacc_norm = linalg.generic {indexing_maps = [affine_map<(B, d1) -> (B, d1)>, affine_map<(B, d1) -> (B, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%norm : tensor<1x4xf32>) outs(%prev_acc : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %31 = arith.mulf %in, %out : f32
          linalg.yield %31 : f32
        } -> tensor<1x4xf32>
        %ppv_sum_norm = linalg.generic {indexing_maps = [affine_map<(B, d1, d2) -> (B, d1)>, affine_map<(B, d1, d2) -> (B, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%norm : tensor<1x4xf32>) outs(%prev_pv_sum : tensor<1x4x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %31 = arith.mulf %in, %out : f32
          linalg.yield %31 : f32
        } -> tensor<1x4x64xf32>
        %27 = linalg.generic {indexing_maps = [affine_map<(B, d1) -> (B, d1)>], iterator_types = ["parallel", "parallel"]} outs(%curr_max : tensor<1x4xf32>) {
        ^bb0(%out: f32):
          linalg.yield %out : f32
        } -> tensor<1x4xf32>
        %curr_acc = linalg.generic {indexing_maps = [affine_map<(B, d1, d2) -> (B, d1, d2)>, affine_map<(B, d1, d2) -> (B, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%s_minus_max : tensor<1x4x4xf32>) outs(%pacc_norm : tensor<1x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %31 = arith.addf %in, %out : f32
          linalg.yield %31 : f32
        } -> tensor<1x4xf32>
        %curr_pv_sum = linalg.generic {indexing_maps = [affine_map<(B, d1, d2, d3) -> (B, d1, d3)>, affine_map<(B, d1, d2, d3) -> (B, d3, d2)>, affine_map<(B, d1, d2, d3) -> (B, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%s_minus_max, %extracted_slice_4 : tensor<1x4x4xf32>, tensor<1x4x64xf32>) outs(%ppv_sum_norm : tensor<1x4x64xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %31 = arith.mulf %in, %in_5 : f32
          %32 = arith.addf %31, %out : f32
          linalg.yield %32 : f32
        } -> tensor<1x4x64xf32>
        %30:3 = linalg.generic {indexing_maps = [affine_map<(B, d1, d2, d3) -> (B, d1, d3)>, affine_map<(B, d1, d2, d3) -> (B, d3, d2)>, affine_map<(B, d1, d2, d3) -> (B, d1)>, affine_map<(B, d1, d2, d3) -> (B, d1)>, affine_map<(B, d1, d2, d3) -> (B, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%s_minus_max, %extracted_slice_4 : tensor<1x4x4xf32>, tensor<1x4x64xf32>) outs(%curr_max, %pacc_norm, %ppv_sum_norm : tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32, %out_6: f32, %out_7: f32):
          %31 = arith.addf %in, %out_6 : f32
          %32 = arith.mulf %in, %in_5 : f32
          %33 = arith.addf %32, %out_7 : f32
          linalg.yield %out, %31, %33 : f32, f32, f32
        } -> (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>)
        scf.yield %27, %curr_acc, %curr_pv_sum : tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4x64xf32>
      }
      %extracted_slice_2 = tensor.extract_slice %arg4[0, %arg3, 0] [1, 4, 64] [1, 1, 1] : tensor<1x32x64xf32> to tensor<1x4x64xf32>
      %18 = linalg.generic {indexing_maps = [affine_map<(B, d1, d2) -> (B, d1, d2)>, affine_map<(B, d1, d2) -> (B, d1)>, affine_map<(B, d1, d2) -> (B, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%17#2, %17#1 : tensor<1x4x64xf32>, tensor<1x4xf32>) outs(%extracted_slice_2 : tensor<1x4x64xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %19 = arith.divf %in, %in_3 : f32
        linalg.yield %19 : f32
      } -> tensor<1x4x64xf32>
      %cast = tensor.cast %18 : tensor<1x4x64xf32> to tensor<?x4x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cast into %arg4[%c0, %arg3, 0] [%c1, 4, 64] [1, 1, 1] : tensor<?x4x64xf32> into tensor<1x32x64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg2[%arg0, %arg1, 0] [1, 32, 64] [1, 1, 1] : tensor<1x32x64xf32> into tensor<20x4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<20x4096x64xf32> into memref<20x4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}
