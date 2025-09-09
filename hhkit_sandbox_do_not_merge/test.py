import torch
import torch.nn as nn
import torch.nn.functional as F


class MatmulModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        x = q @ k
        return x


torch_model = MatmulModel()
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(4096, 4096), torch.randn(4096, 4096))
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
onnx_program.save("matmul.onnx")
