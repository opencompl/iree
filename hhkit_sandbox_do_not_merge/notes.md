Problem we are trying to solve:

- The other elementwise are not getting fused into the for loop. Why?
  - Q: In which pass is the fusion done?
    - compiler/src/iree/compiler/Codegen/LLVMCPU/LLVMCPUTileRootAndFuseProducerConsumer.cpp
        tileRootAndFuseProducerConsumer
  - Q: Why does fusion stop happening?
- Try fusing a matmul

- compiler/src/iree/compiler/Codegen/Common/TileAndFuseUtils.cpp
    collectTiledAndFusedOps

CPULinalgExtTileAndVectorize:

- TileAndDistributeToWorkgroupsUsingForallOpPass
- BufferizeDispatchTensorLoadStorePass
- CombineLayoutTransformationPass
- ConfigTrackingCanonicalizerPass
- CSE
- FuseTensorPadWithConsumerPass
- ConcretizePadResultShapePass
- PropagateDispatchSizeBoundsPass
- DecomposeExpReductionPass
- LLVMCPUTileRootAndFuseProducerConsumerPass
- ConvertAttentionToOnlineAttentionPass
- LLVMCPUTileRootAndFuseProducerConsumerPass <- probably here?
- DecomposeWinogradTransformpass
- DecomposeAttentionPass
- ForallToForPass
- GenericVectorizationPass
- Canonicalizer
- CSE
- OptimizeTensorInsertExtractSlicesPass
- Canonicalizer
- CSE
- LLVMCPUVerifyVectorSizeLegalityPass
- LLVMCPULowerExecutableTargetPass
