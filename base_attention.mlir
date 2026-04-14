func.func @attention(%q: tensor<{B}x{H}x{N}x{D}xf16>, %k: tensor<{B}x{H}x{N}x{D}xf16>, %v: tensor<{B}x{H}x{N}x{D}xf16>) -> tensor<{B}x{H}x{N}x{D}xf16> {
    %scale = arith.constant 1.0: f32

    %score_init = tensor.empty() : tensor<{B}x{H}x{N}x{D}xf16>
    %s = iree_linalg_ext.attention
        {
            indexing_maps = [
                affine_map<(d, d0, d1, d2, d3, d4) -> (d, d0, d1, d2)>, 
                affine_map<(d, d0, d1, d2, d3, d4) -> (d, d0, d3, d2)>, 
                affine_map<(d, d0, d1, d2, d3, d4) -> (d, d0, d3, d4)>, 
                affine_map<(d, d0, d1, d2, d3, d4) -> ()>,
                affine_map<(d, d0, d1, d2, d3, d4) -> (d, d0, d1, d4)>
            ]
        } 
        ins(%q, %k, %v, %scale: tensor<{B}x{H}x{N}x{D}xf16>, tensor<{B}x{H}x{N}x{D}xf16>, tensor<{B}x{H}x{N}x{D}xf16>, f32)
        outs(%score_init: tensor<{B}x{H}x{N}x{D}xf16>) {
        ^bb0(%arg4: f32):
            iree_linalg_ext.yield %arg4 : f32
        } -> tensor<{B}x{H}x{N}x{D}xf16>
    return %s : tensor<{B}x{H}x{N}x{D}xf16>
}