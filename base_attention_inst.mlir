func.func @attention(%q: tensor<2x16x4096x128xf16>, %k: tensor<2x16x4096x128xf16>, %v: tensor<2x16x4096x128xf16>) -> tensor<2x16x4096x128xf16> {
    %scale = arith.constant 1.0: f16

    %score_init = tensor.empty() : tensor<2x16x4096x128xf16>
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
        ins(%q, %k, %v, %scale: tensor<2x16x4096x128xf16>, tensor<2x16x4096x128xf16>, tensor<2x16x4096x128xf16>, f16)
        outs(%score_init: tensor<2x16x4096x128xf16>) {
        ^bb0(%arg4: f16):
            iree_linalg_ext.yield %arg4 : f16
        } -> tensor<2x16x4096x128xf16>
    return %s : tensor<2x16x4096x128xf16>
}