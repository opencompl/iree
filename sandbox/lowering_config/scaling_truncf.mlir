#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @helpme(%46 : tensor<1x1x64x64xf32>) -> (tensor<1x1x64x64xf4E2M1FN>, tensor<1x1x64x64xf8E8M0FNU>)
{
  %47 = tensor.empty() : tensor<1x1x64x64xf4E2M1FN>
  %48 = tensor.empty() : tensor<1x1x64x64xf8E8M0FNU>
  %49:2 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46 : tensor<1x1x64x64xf32>) outs(%47, %48 : tensor<1x1x64x64xf4E2M1FN>, tensor<1x1x64x64xf8E8M0FNU>) {
    ^bb0(%in: f32, %out: f4E2M1FN, %out_9: f8E8M0FNU):
      %result, %scale = iree_linalg_ext.scaling_truncf %in : f32 to f4E2M1FN, f8E8M0FNU
      linalg.yield %result, %scale : f4E2M1FN, f8E8M0FNU
    } -> (tensor<1x1x64x64xf4E2M1FN>, tensor<1x1x64x64xf8E8M0FNU>)
  return %49#0, %49#1: tensor<1x1x64x64xf4E2M1FN>, tensor<1x1x64x64xf8E8M0FNU>

  // decompose to:
  // linalg.generic
  //
}
