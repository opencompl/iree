func.func @attention(%Q: tensor<4x32x4096x64xf16>, %K: tensor<4x32x4096x64xf16>, %V: tensor<4x32x4096x64xf16>) -> tensor<4x32x4096x64xf32> {
    %cst0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index

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

    %red_empty = tensor.empty() : tensor<4x32x4096x64xf32>
    %max_empty = tensor.empty() : tensor<4x32x4096xf32>

    %max_el = arith.constant -3.40282347E+38 : f32
    %max_init = linalg.fill ins(%max_el : f32)
                            outs(%max_empty : tensor<4x32x4096xf32>)
                            -> tensor<4x32x4096xf32>

    %sum_empty = tensor.empty() : tensor<4x32x4096xf32>
    %sum_el = arith.constant 0.000000e+00 : f32
    %sum_init = linalg.fill ins(%sum_el : f32)
                            outs(%sum_empty : tensor<4x32x4096xf32>)
                            -> tensor<4x32x4096xf32>
    %acc_init = linalg.fill ins(%sum_el : f32)
                            outs(%red_empty : tensor<4x32x4096x64xf32>)
                            -> tensor<4x32x4096x64xf32>

    %MAX, %SUM, %PV = iree_linalg_ext.exp_reduction {
        indexing_maps = [
            affine_map<(B, H, M, N, K2) -> (B, H, M, K2)>,
            affine_map<(B, H, M, N, K2) -> (B, H, K2, N)>,
            affine_map<(B, H, M, N, K2) -> (B, H, M)>,
            affine_map<(B, H, M, N, K2) -> (B, H, M)>,
            affine_map<(B, H, M, N, K2) -> (B, H, M, N)>
        ],
        iterator_types = [
            #iree_linalg_ext.iterator_type<parallel>,
            #iree_linalg_ext.iterator_type<parallel>,
            #iree_linalg_ext.iterator_type<parallel>,
            #iree_linalg_ext.iterator_type<parallel>,
            #iree_linalg_ext.iterator_type<reduction>
        ],
        exp_reduced_operands = [1, 2]
    }
    ins(%S, %V : tensor<4x32x4096x4096xf32>, tensor<4x32x4096x64xf16>)
    outs(%max_init, %sum_init, %acc_init : tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>)
    {
    ^bb0(%ex : f32, %v : f16, %m : f32, %sum : f32, %acc : f32):
        %trunc = arith.truncf %ex : f32 to f16
        %ex_ext = arith.extf %trunc : f16 to f32
        %v_ext = arith.extf %v : f16 to f32
        %nsum = arith.addf %ex_ext, %sum : f32
        %mul  = arith.mulf %ex_ext, %v_ext : f32
        %nacc = arith.addf %mul, %acc : f32
        iree_linalg_ext.yield %m, %nsum, %nacc : f32, f32, f32
    } -> tensor<4x32x4096xf32>, tensor<4x32x4096xf32>, tensor<4x32x4096x64xf32>

    %result = linalg.generic {
                indexing_maps = [
                affine_map<(B, H, M, N) -> (B, H, M)>,
                affine_map<(B, H, M, N) -> (B, H, M, N)>
                ],
                iterator_types = ["parallel",  "parallel", "parallel", "parallel"]
            }
            ins(%SUM : tensor<4x32x4096xf32>)
            outs(%PV : tensor<4x32x4096x64xf32>) {
    ^bb0(%sum : f32, %pv : f32):
    %out = arith.divf %pv, %sum : f32
    linalg.yield %out : f32
    } -> tensor<4x32x4096x64xf32>

    return %result: tensor<4x32x4096x64xf32>
}
