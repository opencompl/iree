// -----// IR Dump After TileAndDistributeToWorkgroupsUsingForallOpPass (iree-codegen-tile-and-distribute-to-workgroups-using-forall-op) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x64xf32>>
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>> -> tensor<4096x64xf32>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>> -> tensor<4096x64xf32>
  %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : tensor<4096x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x64xf32>>
  return
}

// -----// IR Dump After BufferizeDispatchTensorLoadStorePass (iree-codegen-bufferize-dispatch-tensor-load-store) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>
  %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %7 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x64xf32>>
  %8 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %9 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %10 = iree_codegen.load_from_buffer %4 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %11 = tensor.empty() : tensor<4096x64xf32>
  %12 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %11) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %8[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %13 = tensor.empty() : tensor<64x4096xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %9 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%13 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %16 = arith.mulf %in, %in_1 : f32
      %17 = arith.addf %16, %out : f32
      linalg.yield %17 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14, %10 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %16 = arith.mulf %in, %in_1 : f32
      %17 = arith.addf %16, %out : f32
      linalg.yield %17 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %15 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %12, %6 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After CombineLayoutTransformationPass (iree-codegen-combine-layout-transformation) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After ConfigTrackingCanonicalizerPass (iree-codegen-config-tracking-canonicalize) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After CSE (cse) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After FuseTensorPadWithConsumerPass (iree-codegen-fuse-tensor-pad-with-consumer) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After ConcretizePadResultShapePass (iree-codegen-concretize-pad-result-shape) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After PropagateDispatchSizeBoundsPass (iree-codegen-propagate-dispatch-size-bounds) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After DecomposeExpReductionPass (iree-linalg-ext-decompose-exp-reduction) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x4096xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %5 : tensor<64x64xf32>, tensor<4096x64xf32>) outs(%9 : tensor<64x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    } -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

test-matmul.mlir:49:13: warning: tiling is not thread safe at axis #2
  %result = linalg.generic  {
            ^
test-matmul.mlir:49:13: note: see current operation:
%11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %6 : tensor<64x4096xf32>, tensor<4096x64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
^bb0(%in: f32, %in_1: f32, %out: f32):
  %12 = arith.mulf %in, %in_1 : f32
  %13 = arith.addf %12, %out : f32
  linalg.yield %13 : f32
} -> tensor<64x64xf32>
// -----// IR Dump After LLVMCPUTileRootAndFuseProducerConsumerPass (iree-llvmcpu-tile-root-and-fuse-producer-consumer) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = scf.forall (%arg2, %arg3) = (0, 0) to (64, 4096) step (8, 32) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %extracted_slice_1 = tensor.extract_slice %5[%arg3, 0] [32, 64] [1, 1] : tensor<4096x64xf32> to tensor<32x64xf32>
      %10 = tensor.empty() : tensor<64x32xf32>
      %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_1 : tensor<64x64xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x32xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %13 = arith.mulf %in, %in_4 : f32
        %14 = arith.addf %13, %out : f32
        linalg.yield %14 : f32
      } -> tensor<64x32xf32>
      %extracted_slice_2 = tensor.extract_slice %6[%arg3, %arg2] [32, 8] [1, 1] : tensor<4096x64xf32> to tensor<32x8xf32>
      %extracted_slice_3 = tensor.extract_slice %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
      %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11, %extracted_slice_2 : tensor<64x32xf32>, tensor<32x8xf32>) outs(%extracted_slice_3 : tensor<64x8xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %13 = arith.mulf %in, %in_4 : f32
        %14 = arith.addf %13, %out : f32
        linalg.yield %14 : f32
      } -> tensor<64x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %12 into %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After ConvertAttentionToOnlineAttentionPass (iree-linalg-ext-convert-attention-to-online-attention) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = scf.forall (%arg2, %arg3) = (0, 0) to (64, 4096) step (8, 32) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %extracted_slice_1 = tensor.extract_slice %5[%arg3, 0] [32, 64] [1, 1] : tensor<4096x64xf32> to tensor<32x64xf32>
      %10 = tensor.empty() : tensor<64x32xf32>
      %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_1 : tensor<64x64xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x32xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %13 = arith.mulf %in, %in_4 : f32
        %14 = arith.addf %13, %out : f32
        linalg.yield %14 : f32
      } -> tensor<64x32xf32>
      %extracted_slice_2 = tensor.extract_slice %6[%arg3, %arg2] [32, 8] [1, 1] : tensor<4096x64xf32> to tensor<32x8xf32>
      %extracted_slice_3 = tensor.extract_slice %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
      %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11, %extracted_slice_2 : tensor<64x32xf32>, tensor<32x8xf32>) outs(%extracted_slice_3 : tensor<64x8xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %13 = arith.mulf %in, %in_4 : f32
        %14 = arith.addf %13, %out : f32
        linalg.yield %14 : f32
      } -> tensor<64x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %12 into %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After LLVMCPUTileRootAndFuseProducerConsumerPass (iree-llvmcpu-tile-root-and-fuse-producer-consumer) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = scf.forall (%arg2, %arg3) = (0, 0) to (64, 4096) step (8, 32) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %extracted_slice_1 = tensor.extract_slice %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
      %10 = scf.for %arg5 = %c0 to %c32 step %c16 iter_args(%arg6 = %extracted_slice_1) -> (tensor<64x8xf32>) {
        %11 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg5, %arg3]
        %extracted_slice_2 = tensor.extract_slice %5[%11, 0] [16, 64] [1, 1] : tensor<4096x64xf32> to tensor<16x64xf32>
        %12 = tensor.empty() : tensor<64x16xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_2 : tensor<64x64xf32>, tensor<16x64xf32>) outs(%12 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %16 = arith.mulf %in, %in_4 : f32
          %17 = arith.addf %16, %out : f32
          linalg.yield %17 : f32
        } -> tensor<64x16xf32>
        %14 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg5, %arg3]
        %extracted_slice_3 = tensor.extract_slice %6[%14, %arg2] [16, 8] [1, 1] : tensor<4096x64xf32> to tensor<16x8xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %extracted_slice_3 : tensor<64x16xf32>, tensor<16x8xf32>) outs(%arg6 : tensor<64x8xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %16 = arith.mulf %in, %in_4 : f32
          %17 = arith.addf %16, %out : f32
          linalg.yield %17 : f32
        } -> tensor<64x8xf32>
        scf.yield %15 : tensor<64x8xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After DecomposeWinogradTransformPass (iree-linalg-ext-decompose-winograd) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = scf.forall (%arg2, %arg3) = (0, 0) to (64, 4096) step (8, 32) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %extracted_slice_1 = tensor.extract_slice %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
      %10 = scf.for %arg5 = %c0 to %c32 step %c16 iter_args(%arg6 = %extracted_slice_1) -> (tensor<64x8xf32>) {
        %11 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg5, %arg3]
        %extracted_slice_2 = tensor.extract_slice %5[%11, 0] [16, 64] [1, 1] : tensor<4096x64xf32> to tensor<16x64xf32>
        %12 = tensor.empty() : tensor<64x16xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_2 : tensor<64x64xf32>, tensor<16x64xf32>) outs(%12 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %16 = arith.mulf %in, %in_4 : f32
          %17 = arith.addf %16, %out : f32
          linalg.yield %17 : f32
        } -> tensor<64x16xf32>
        %14 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg5, %arg3]
        %extracted_slice_3 = tensor.extract_slice %6[%14, %arg2] [16, 8] [1, 1] : tensor<4096x64xf32> to tensor<16x8xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %extracted_slice_3 : tensor<64x16xf32>, tensor<16x8xf32>) outs(%arg6 : tensor<64x8xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %16 = arith.mulf %in, %in_4 : f32
          %17 = arith.addf %16, %out : f32
          linalg.yield %17 : f32
        } -> tensor<64x8xf32>
        scf.yield %15 : tensor<64x8xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After DecomposeAttentionPass (iree-linalg-ext-decompose-attention) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %9 = scf.forall (%arg2, %arg3) = (0, 0) to (64, 4096) step (8, 32) shared_outs(%arg4 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %extracted_slice_1 = tensor.extract_slice %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
      %10 = scf.for %arg5 = %c0 to %c32 step %c16 iter_args(%arg6 = %extracted_slice_1) -> (tensor<64x8xf32>) {
        %11 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg5, %arg3]
        %extracted_slice_2 = tensor.extract_slice %5[%11, 0] [16, 64] [1, 1] : tensor<4096x64xf32> to tensor<16x64xf32>
        %12 = tensor.empty() : tensor<64x16xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_2 : tensor<64x64xf32>, tensor<16x64xf32>) outs(%12 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %16 = arith.mulf %in, %in_4 : f32
          %17 = arith.addf %16, %out : f32
          linalg.yield %17 : f32
        } -> tensor<64x16xf32>
        %14 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg5, %arg3]
        %extracted_slice_3 = tensor.extract_slice %6[%14, %arg2] [16, 8] [1, 1] : tensor<4096x64xf32> to tensor<16x8xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %extracted_slice_3 : tensor<64x16xf32>, tensor<16x8xf32>) outs(%arg6 : tensor<64x8xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %16 = arith.mulf %in, %in_4 : f32
          %17 = arith.addf %16, %out : f32
          linalg.yield %17 : f32
        } -> tensor<64x8xf32>
        scf.yield %15 : tensor<64x8xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After ForallToForPass (iree-codegen-forall-to-for) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = iree_codegen.load_from_buffer %0 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = tensor.empty() : tensor<4096x64xf32>
  %8 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %7) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %4[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %c8 = arith.constant 8 : index
    %c32_3 = arith.constant 32 : index
    %9 = scf.for %arg2 = %c0_1 to %c64 step %c8 iter_args(%arg3 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %10 = scf.for %arg4 = %c0_2 to %c4096 step %c32_3 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
        %extracted_slice_4 = tensor.extract_slice %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
        %11 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %extracted_slice_4) -> (tensor<64x8xf32>) {
          %12 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %extracted_slice_5 = tensor.extract_slice %5[%12, 0] [16, 64] [1, 1] : tensor<4096x64xf32> to tensor<16x64xf32>
          %13 = tensor.empty() : tensor<64x16xf32>
          %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_5 : tensor<64x64xf32>, tensor<16x64xf32>) outs(%13 : tensor<64x16xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %17 = arith.mulf %in, %in_7 : f32
            %18 = arith.addf %17, %out : f32
            linalg.yield %18 : f32
          } -> tensor<64x16xf32>
          %15 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %extracted_slice_6 = tensor.extract_slice %6[%15, %arg2] [16, 8] [1, 1] : tensor<4096x64xf32> to tensor<16x8xf32>
          %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14, %extracted_slice_6 : tensor<64x16xf32>, tensor<16x8xf32>) outs(%arg7 : tensor<64x8xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [0, 8, 32], vector_reduction = [0, 0, 16]>} {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %17 = arith.mulf %in, %in_7 : f32
            %18 = arith.addf %17, %out : f32
            linalg.yield %18 : f32
          } -> tensor<64x8xf32>
          scf.yield %16 : tensor<64x8xf32>
        }
        %inserted_slice = tensor.insert_slice %11 into %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
        scf.yield %inserted_slice : tensor<64x64xf32>
      }
      scf.yield %10 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %9 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %8, %3 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After GenericVectorizationPass (iree-codegen-generic-vectorization) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %5[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %11 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
        %extracted_slice_1 = tensor.extract_slice %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
        %12 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %extracted_slice_1) -> (tensor<64x8xf32>) {
          %13 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %extracted_slice_2 = tensor.extract_slice %6[%13, 0] [16, 64] [1, 1] : tensor<4096x64xf32> to tensor<16x64xf32>
          %14 = tensor.empty() : tensor<64x16xf32>
          %15 = vector.transfer_read %extracted_slice[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x64xf32>
          %16 = vector.transfer_read %extracted_slice_2[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<16x64xf32>, vector<16x64xf32>
          %17 = vector.transfer_read %14[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
          %18 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %16, %17 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %extracted_slice_3 = tensor.extract_slice %7[%19, %arg2] [16, 8] [1, 1] : tensor<4096x64xf32> to tensor<16x8xf32>
          %20 = vector.transfer_read %extracted_slice_3[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<16x8xf32>, vector<16x8xf32>
          %21 = vector.transfer_read %arg7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x8xf32>, vector<64x8xf32>
          %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %20, %21 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          %23 = vector.transfer_write %22, %arg7[%c0, %c0] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x8xf32>
          scf.yield %23 : tensor<64x8xf32>
        }
        %inserted_slice = tensor.insert_slice %12 into %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
        scf.yield %inserted_slice : tensor<64x64xf32>
      }
      scf.yield %11 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %5[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %11 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
        %extracted_slice_1 = tensor.extract_slice %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
        %12 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %extracted_slice_1) -> (tensor<64x8xf32>) {
          %13 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %extracted_slice_2 = tensor.extract_slice %6[%13, 0] [16, 64] [1, 1] : tensor<4096x64xf32> to tensor<16x64xf32>
          %14 = tensor.empty() : tensor<64x16xf32>
          %15 = vector.transfer_read %extracted_slice[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x64xf32>
          %16 = vector.transfer_read %extracted_slice_2[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<16x64xf32>, vector<16x64xf32>
          %17 = vector.transfer_read %14[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
          %18 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %16, %17 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %extracted_slice_3 = tensor.extract_slice %7[%19, %arg2] [16, 8] [1, 1] : tensor<4096x64xf32> to tensor<16x8xf32>
          %20 = vector.transfer_read %extracted_slice_3[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<16x8xf32>, vector<16x8xf32>
          %21 = vector.transfer_read %arg7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x8xf32>, vector<64x8xf32>
          %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %20, %21 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          %23 = vector.transfer_write %22, %arg7[%c0, %c0] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x8xf32>
          scf.yield %23 : tensor<64x8xf32>
        }
        %inserted_slice = tensor.insert_slice %12 into %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
        scf.yield %inserted_slice : tensor<64x64xf32>
      }
      scf.yield %11 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After CSE (cse) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %5[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      %11 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
        %extracted_slice_1 = tensor.extract_slice %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
        %12 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %extracted_slice_1) -> (tensor<64x8xf32>) {
          %13 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %extracted_slice_2 = tensor.extract_slice %6[%13, 0] [16, 64] [1, 1] : tensor<4096x64xf32> to tensor<16x64xf32>
          %14 = tensor.empty() : tensor<64x16xf32>
          %15 = vector.transfer_read %extracted_slice[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x64xf32>
          %16 = vector.transfer_read %extracted_slice_2[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<16x64xf32>, vector<16x64xf32>
          %17 = vector.transfer_read %14[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
          %18 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %16, %17 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %extracted_slice_3 = tensor.extract_slice %7[%13, %arg2] [16, 8] [1, 1] : tensor<4096x64xf32> to tensor<16x8xf32>
          %19 = vector.transfer_read %extracted_slice_3[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<16x8xf32>, vector<16x8xf32>
          %20 = vector.transfer_read %arg7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x8xf32>, vector<64x8xf32>
          %21 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %19, %20 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          %22 = vector.transfer_write %21, %arg7[%c0, %c0] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x8xf32>
          scf.yield %22 : tensor<64x8xf32>
        }
        %inserted_slice = tensor.insert_slice %12 into %arg5[0, %arg2] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
        scf.yield %inserted_slice : tensor<64x64xf32>
      }
      scf.yield %11 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After OptimizeTensorInsertExtractSlicesPass (iree-codegen-optimize-tensor-insert-extract-slices) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = tensor.empty() : tensor<64x16xf32>
    %11 = vector.transfer_read %5[%arg0, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<64x64xf32>
    %12 = vector.transfer_read %10[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
    %13 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice) -> (tensor<64x64xf32>) {
      %14 = vector.transfer_read %arg3[%c0, %arg2], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x8xf32>
      %15 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %14) -> (vector<64x8xf32>) {
        %17 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %arg5) -> (vector<64x8xf32>) {
          %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %19 = vector.transfer_read %6[%18, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x64xf32>
          %20 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %19, %12 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %21 = vector.transfer_read %7[%18, %arg2], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x8xf32>
          %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %21, %arg7 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          scf.yield %22 : vector<64x8xf32>
        }
        scf.yield %17 : vector<64x8xf32>
      }
      %16 = vector.transfer_write %15, %arg3[%c0, %arg2] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x64xf32>
      scf.yield %16 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = tensor.empty() : tensor<64x16xf32>
    %11 = vector.transfer_read %5[%arg0, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<64x64xf32>
    %12 = vector.transfer_read %10[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
    %13 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice) -> (tensor<64x64xf32>) {
      %14 = vector.transfer_read %arg3[%c0, %arg2], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x8xf32>
      %15 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %14) -> (vector<64x8xf32>) {
        %17 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %arg5) -> (vector<64x8xf32>) {
          %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %19 = vector.transfer_read %6[%18, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x64xf32>
          %20 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %19, %12 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %21 = vector.transfer_read %7[%18, %arg2], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x8xf32>
          %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %21, %arg7 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          scf.yield %22 : vector<64x8xf32>
        }
        scf.yield %17 : vector<64x8xf32>
      }
      %16 = vector.transfer_write %15, %arg3[%c0, %arg2] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x64xf32>
      scf.yield %16 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After CSE (cse) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = tensor.empty() : tensor<64x16xf32>
    %11 = vector.transfer_read %5[%arg0, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<64x64xf32>
    %12 = vector.transfer_read %10[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
    %13 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice) -> (tensor<64x64xf32>) {
      %14 = vector.transfer_read %arg3[%c0, %arg2], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x8xf32>
      %15 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %14) -> (vector<64x8xf32>) {
        %17 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %arg5) -> (vector<64x8xf32>) {
          %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %19 = vector.transfer_read %6[%18, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x64xf32>
          %20 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %19, %12 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %21 = vector.transfer_read %7[%18, %arg2], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x8xf32>
          %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %21, %arg7 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          scf.yield %22 : vector<64x8xf32>
        }
        scf.yield %17 : vector<64x8xf32>
      }
      %16 = vector.transfer_write %15, %arg3[%c0, %arg2] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x64xf32>
      scf.yield %16 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

test-matmul.mlir:16:1: error: One or more operations with large vector sizes (32768 bytes) were found:

func.func @no_peel_static_matmul() attributes {hal.executable.target = #executable_target_system_elf_x86_64_, translation_info = #translation} {
^
test-matmul.mlir:45:13: note:   %20 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %19, %12 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>

    %sum  = arith.addf %mul, %s : f32
            ^
test-matmul.mlir:63:13: note:   %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %21, %arg7 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>

    %sum  = arith.addf %mul, %s : f32
            ^
// -----// IR Dump After LLVMCPUVerifyVectorSizeLegalityPass Failed (iree-llvmcpu-verify-vector-size-legality) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = tensor.empty() : tensor<64x16xf32>
    %11 = vector.transfer_read %5[%arg0, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<64x64xf32>
    %12 = vector.transfer_read %10[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
    %13 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice) -> (tensor<64x64xf32>) {
      %14 = vector.transfer_read %arg3[%c0, %arg2], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x8xf32>
      %15 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %14) -> (vector<64x8xf32>) {
        %17 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %arg5) -> (vector<64x8xf32>) {
          %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %19 = vector.transfer_read %6[%18, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x64xf32>
          %20 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %19, %12 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %21 = vector.transfer_read %7[%18, %arg2], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x8xf32>
          %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %21, %arg7 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          scf.yield %22 : vector<64x8xf32>
        }
        scf.yield %17 : vector<64x8xf32>
      }
      %16 = vector.transfer_write %15, %arg3[%c0, %arg2] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x64xf32>
      scf.yield %16 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----// IR Dump After LLVMCPULowerExecutableTargetPass Failed (iree-llvmcpu-lower-executable-target) //----- //
func.func @no_peel_static_matmul() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {native_vector_size = 64 : i64}>, translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %0 = ub.poison : f32
  %c8 = arith.constant 8 : index
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(2) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(3) : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  %5 = iree_codegen.load_from_buffer %1 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %6 = iree_codegen.load_from_buffer %2 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %7 = iree_codegen.load_from_buffer %3 : memref<4096x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4096x64xf32>
  %8 = tensor.empty() : tensor<4096x64xf32>
  %9 = scf.forall (%arg0) = (0) to (4096) step (64) shared_outs(%arg1 = %8) -> (tensor<4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<4096x64xf32> to tensor<64x64xf32>
    %10 = tensor.empty() : tensor<64x16xf32>
    %11 = vector.transfer_read %5[%arg0, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<64x64xf32>
    %12 = vector.transfer_read %10[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x16xf32>, vector<64x16xf32>
    %13 = scf.for %arg2 = %c0 to %c64 step %c8 iter_args(%arg3 = %extracted_slice) -> (tensor<64x64xf32>) {
      %14 = vector.transfer_read %arg3[%c0, %arg2], %0 {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x8xf32>
      %15 = scf.for %arg4 = %c0 to %c4096 step %c32 iter_args(%arg5 = %14) -> (vector<64x8xf32>) {
        %17 = scf.for %arg6 = %c0 to %c32 step %c16 iter_args(%arg7 = %arg5) -> (vector<64x8xf32>) {
          %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg4]
          %19 = vector.transfer_read %6[%18, %c0], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x64xf32>
          %20 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %19, %12 : vector<64x64xf32>, vector<16x64xf32> into vector<64x16xf32>
          %21 = vector.transfer_read %7[%18, %arg2], %0 {in_bounds = [true, true]} : tensor<4096x64xf32>, vector<16x8xf32>
          %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %20, %21, %arg7 : vector<64x16xf32>, vector<16x8xf32> into vector<64x8xf32>
          scf.yield %22 : vector<64x8xf32>
        }
        scf.yield %17 : vector<64x8xf32>
      }
      %16 = vector.transfer_write %15, %arg3[%c0, %arg2] {in_bounds = [true, true]} : vector<64x8xf32>, tensor<64x64xf32>
      scf.yield %16 : tensor<64x64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg1[%arg0, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<4096x64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %9, %4 : tensor<4096x64xf32> into memref<4096x64xf32, #hal.descriptor_type<storage_buffer>>
  return
}
