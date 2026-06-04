// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.exp_reduction}), canonicalize, cse)" --split-input-file %s
func.func @dispatch_scaled_matmul_like_4x32x4096x4096x2x32_f4E2M1FNxf4E2M1FNxf8E8M0FNUxf8E8M0FNUxf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}>} {
  %c64 = arith.constant 64 : index
  %c4096 = arith.constant 4096 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %cst_1 = arith.constant 6.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4x32x4096x2x32xf4E2M1FN, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<4x32x4096x2x32xf4E2M1FN, #hal.descriptor_type<storage_buffer>> to memref<4x32x4096x2x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4x32x4096x2x32xf4E2M1FN, #hal.descriptor_type<storage_buffer>>
  %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<4x32x4096x2x32xf4E2M1FN, #hal.descriptor_type<storage_buffer>> to memref<4x32x4096x2x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4x32x4096x2x32xf4E2M1FN, #hal.descriptor_type<storage_buffer>>
  %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<4x32x4096x2x32xf4E2M1FN, #hal.descriptor_type<storage_buffer>> to memref<4x32x4096x2x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>
  %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4x32x4096x2xf8E8M0FNU, #hal.descriptor_type<storage_buffer>>
  %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<4x32x4096x2xf8E8M0FNU, #hal.descriptor_type<storage_buffer>> to memref<4x32x4096x2xf8E8M0FNU, #amdgpu.address_space<fat_raw_buffer>>
  %8 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4x32x4096x2xf8E8M0FNU, #hal.descriptor_type<storage_buffer>>
  %9 = amdgpu.fat_raw_buffer_cast %8 resetOffset : memref<4x32x4096x2xf8E8M0FNU, #hal.descriptor_type<storage_buffer>> to memref<4x32x4096x2xf8E8M0FNU, #amdgpu.address_space<fat_raw_buffer>>
  %10 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(5) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4x32x4096x2xf8E8M0FNU, #hal.descriptor_type<storage_buffer>>
  %11 = amdgpu.fat_raw_buffer_cast %10 resetOffset : memref<4x32x4096x2xf8E8M0FNU, #hal.descriptor_type<storage_buffer>> to memref<4x32x4096x2xf8E8M0FNU, #amdgpu.address_space<fat_raw_buffer>>
  %12 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(6) alignment(64) offset(%c0) flags(Indirect) : memref<4x32x4096x2x32xf32, #hal.descriptor_type<storage_buffer>>
  %13 = amdgpu.fat_raw_buffer_cast %12 resetOffset : memref<4x32x4096x2x32xf32, #hal.descriptor_type<storage_buffer>> to memref<4x32x4096x2x32xf32, #amdgpu.address_space<fat_raw_buffer>>
  %14 = iree_codegen.load_from_buffer %1 : memref<4x32x4096x2x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x32x4096x2x32xf4E2M1FN>
  %15 = iree_codegen.load_from_buffer %3 : memref<4x32x4096x2x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x32x4096x2x32xf4E2M1FN>
  %16 = iree_codegen.load_from_buffer %5 : memref<4x32x4096x2x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x32x4096x2x32xf4E2M1FN>
  %17 = iree_codegen.load_from_buffer %7 : memref<4x32x4096x2xf8E8M0FNU, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x32x4096x2xf8E8M0FNU>
  %18 = iree_codegen.load_from_buffer %9 : memref<4x32x4096x2xf8E8M0FNU, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x32x4096x2xf8E8M0FNU>
  %19 = iree_codegen.load_from_buffer %11 : memref<4x32x4096x2xf8E8M0FNU, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4x32x4096x2xf8E8M0FNU>
  %20 = tensor.empty() : tensor<4x32x4096x2x32xf32>
  %21 = scf.forall (%arg0, %arg1, %arg2) = (0, 0, 0) to (4, 32, 4096) step (1, 1, 64) shared_outs(%arg3 = %20) -> (tensor<4x32x4096x2x32xf32>) {
    %extracted_slice = tensor.extract_slice %14[%arg0, %arg1, %arg2, 0, 0] [1, 1, 64, 2, 32] [1, 1, 1, 1, 1] : tensor<4x32x4096x2x32xf4E2M1FN> to tensor<1x1x64x2x32xf4E2M1FN>
    %extracted_slice_2 = tensor.extract_slice %17[%arg0, %arg1, %arg2, 0] [1, 1, 64, 2] [1, 1, 1, 1] : tensor<4x32x4096x2xf8E8M0FNU> to tensor<1x1x64x2xf8E8M0FNU>
    %22 = tensor.empty() : tensor<1x1x64x2x32xf4E2M1FN>
    %23 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice : tensor<1x1x64x2x32xf4E2M1FN>) outs(%22 : tensor<1x1x64x2x32xf4E2M1FN>) -> tensor<1x1x64x2x32xf4E2M1FN>
    %24 = tensor.empty() : tensor<1x1x64x2xf8E8M0FNU>
    %25 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_2 : tensor<1x1x64x2xf8E8M0FNU>) outs(%24 : tensor<1x1x64x2xf8E8M0FNU>) -> tensor<1x1x64x2xf8E8M0FNU>
    %26 = tensor.empty() : tensor<1x1x64xf32>
    %27 = linalg.fill ins(%cst_0 : f32) outs(%26 : tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %28 = linalg.fill ins(%cst : f32) outs(%26 : tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
    %29 = tensor.empty() : tensor<1x1x64x2x32xf32>
    %30 = linalg.fill ins(%cst : f32) outs(%29 : tensor<1x1x64x2x32xf32>) -> tensor<1x1x64x2x32xf32>
    %31:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %27, %arg6 = %28, %arg7 = %30) -> (tensor<1x1x64xf32>, tensor<1x1x64xf32>, tensor<1x1x64x2x32xf32>) {
      %extracted_slice_4 = tensor.extract_slice %15[%arg0, %arg1, %arg4, 0, 0] [1, 1, 64, 2, 32] [1, 1, 1, 1, 1] : tensor<4x32x4096x2x32xf4E2M1FN> to tensor<1x1x64x2x32xf4E2M1FN>
      %33 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_4 : tensor<1x1x64x2x32xf4E2M1FN>) outs(%22 : tensor<1x1x64x2x32xf4E2M1FN>) -> tensor<1x1x64x2x32xf4E2M1FN>
      %extracted_slice_5 = tensor.extract_slice %18[%arg0, %arg1, %arg4, 0] [1, 1, 64, 2] [1, 1, 1, 1] : tensor<4x32x4096x2xf8E8M0FNU> to tensor<1x1x64x2xf8E8M0FNU>
      %34 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_5 : tensor<1x1x64x2xf8E8M0FNU>) outs(%24 : tensor<1x1x64x2xf8E8M0FNU>) -> tensor<1x1x64x2xf8E8M0FNU>
      %35 = tensor.empty() : tensor<1x1x64x64xf32>
      %36 = linalg.fill ins(%cst : f32) outs(%35 : tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf32>
      %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%23, %33, %25, %34 : tensor<1x1x64x2x32xf4E2M1FN>, tensor<1x1x64x2x32xf4E2M1FN>, tensor<1x1x64x2xf8E8M0FNU>, tensor<1x1x64x2xf8E8M0FNU>) outs(%36 : tensor<1x1x64x64xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>, promote_operands = [0, 1, 2, 3], subgroup_basis = [[1, 1, 2, 1, 1, 1], [0, 1, 2, 3, 4, 5]]}>} {
      ^bb0(%in: f4E2M1FN, %in_8: f4E2M1FN, %in_9: f8E8M0FNU, %in_10: f8E8M0FNU, %out: f32):
        %39 = arith.scaling_extf %in, %in_9 : f4E2M1FN, f8E8M0FNU to f32
        %40 = arith.scaling_extf %in_8, %in_10 : f4E2M1FN, f8E8M0FNU to f32
        %41 = arith.mulf %39, %40 : f32
        %42 = arith.addf %41, %out : f32
        linalg.yield %42 : f32
      } -> tensor<1x1x64x64xf32>
      %extracted_slice_6 = tensor.extract_slice %16[%arg0, %arg1, %arg4, 0, 0] [1, 1, 64, 2, 32] [1, 1, 1, 1, 1] : tensor<4x32x4096x2x32xf4E2M1FN> to tensor<1x1x64x2x32xf4E2M1FN>
      %extracted_slice_7 = tensor.extract_slice %19[%arg0, %arg1, %arg4, 0] [1, 1, 64, 2] [1, 1, 1, 1] : tensor<4x32x4096x2xf8E8M0FNU> to tensor<1x1x64x2xf8E8M0FNU>
      %38:3 = iree_linalg_ext.exp_reduction{indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>, #iree_linalg_ext.iterator_type<reduction>], exp_reduced_operands = [1, 2]} attributes {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>, reduction = [0, 0, 0, 0, 0, 64], subgroup_basis = [[1, 1, 2, 1, 1, 1], [0, 1, 2, 3, 4, 5]], workgroup = [1, 1, 64, 0, 0, 0]}>} ins(%37, %extracted_slice_6, %extracted_slice_7 : tensor<1x1x64x64xf32>, tensor<1x1x64x2x32xf4E2M1FN>, tensor<1x1x64x2xf8E8M0FNU>) outs(%arg5, %arg6, %arg7 : tensor<1x1x64xf32>, tensor<1x1x64xf32>, tensor<1x1x64x2x32xf32>) {
      ^bb0(%arg8: f32, %arg9: f4E2M1FN, %arg10: f8E8M0FNU, %arg11: f32, %arg12: f32, %arg13: f32):
        %39 = arith.mulf %arg8, %cst_1 : f32
        %result = arith.scaling_truncf %39, %scale:  f32, f8E8M0FNU to f4E2M1FN
        %40 = arith.scaling_extf %result, %scale : f4E2M1FN, f8E8M0FNU to f32
        %41 = arith.scaling_extf %arg9, %arg10 : f4E2M1FN, f8E8M0FNU to f32
        %42 = arith.addf %arg8, %arg12 : f32
        %43 = arith.mulf %40, %41 : f32
        %44 = arith.addf %43, %arg13 : f32
        iree_linalg_ext.yield %arg11, %42, %44 : f32, f32, f32
      } -> tensor<1x1x64xf32>, tensor<1x1x64xf32>, tensor<1x1x64x2x32xf32>
      scf.yield %38#0, %38#1, %38#2 : tensor<1x1x64xf32>, tensor<1x1x64xf32>, tensor<1x1x64x2x32xf32>
    }
    %extracted_slice_3 = tensor.extract_slice %arg3[%arg0, %arg1, %arg2, 0, 0] [1, 1, 64, 2, 32] [1, 1, 1, 1, 1] : tensor<4x32x4096x2x32xf32> to tensor<1x1x64x2x32xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%31#1, %31#2 : tensor<1x1x64xf32>, tensor<1x1x64x2x32xf32>) outs(%extracted_slice_3 : tensor<1x1x64x2x32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %33 = arith.divf %in_4, %in : f32
      linalg.yield %33 : f32
    } -> tensor<1x1x64x2x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %32 into %arg3[%arg0, %arg1, %arg2, 0, 0] [1, 1, 64, 2, 32] [1, 1, 1, 1, 1] : tensor<1x1x64x2x32xf32> into tensor<4x32x4096x2x32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %21, %13 : tensor<4x32x4096x2x32xf32> into memref<4x32x4096x2x32xf32, #amdgpu.address_space<fat_raw_buffer>>
  return
}
