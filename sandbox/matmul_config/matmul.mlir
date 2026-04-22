func.func @attention(%Q: tensor<4x32x4096x64xf16>, %K: tensor<4x32x4096x64xf16>) -> tensor<4x32x4096x4096xf32> {
    %cst0 = arith.constant 0.0 : f32
    %S_empty = tensor.empty() : tensor<4x32x4096x4096xf32>
    %S_fill  = linalg.fill ins(%cst0 : f32)
                            outs(%S_empty : tensor<4x32x4096x4096xf32>)
                            -> tensor<4x32x4096x4096xf32>

    %S = linalg.generic  {
        indexing_maps = [
        affine_map<(Z, H, N1, N2, D) -> (Z, H, N1, D)>,
        affine_map<(Z, H, N1, N2, D) -> (Z, H, N2, D)>,
        affine_map<(Z, H, N1, N2, D) -> (Z, H, N1, N2)>
        ],
        iterator_types = ["parallel", "parallel",  "parallel", "parallel", "reduction"]
    }
    ins(%Q, %K : tensor<4x32x4096x64xf16>, tensor<4x32x4096x64xf16>)
    outs(%S_fill : tensor<4x32x4096x4096xf32>)
    {
    ^bb0(%q : f16, %k : f16, %s : f32):
    %q_ext = arith.extf %q : f16 to f32
    %k_ext = arith.extf %k : f16 to f32
    %mul  = arith.mulf %q_ext, %k_ext : f32
    %sum  = arith.addf %mul, %s : f32
    linalg.yield %sum : f32
    } -> tensor<4x32x4096x4096xf32>
    return %S: tensor<4x32x4096x4096xf32>
}
