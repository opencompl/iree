import iree.runtime as ireert
import iree.compiler as comp
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


def is_hip_cdna2():
    return False
    # target = triton.runtime.driver.active.get_current_target()
    # return target.backend == 'hip' and target.arch == 'gfx90a'


def main():
    flatbuffer = comp.compile_file(
        "e2e.mlir",
        target_backends=["llvm-cpu"],
        extra_args=["--iree-hal-target-device=local", "--iree-llvmcpu-target-cpu=host"],
    )
    module = ireert.load_vm_flatbuffer(flatbuffer, backend="llvm-cpu")

    shape = (20, 4096, 64)
    # q = torch.full(shape, 3, dtype=torch.float32)
    # k = torch.full(shape, 3, dtype=torch.float32)
    # v = torch.full(shape, 3, dtype=torch.float32)
    q = torch.rand(shape)
    k = torch.rand(shape)
    v = torch.rand(shape)

    # ireert.load_vm_module(modules)
    # print(dir(modules))
    iree_output = module.attention(q, k, v)
    iree_output = torch.from_numpy(iree_output.to_host())
    # torch_output = torch.softmax(q@k.transpose(1,-1), dim=-1) @ v
    torch_output = scaled_dot_product_attention(q, k, v, scale=1)

    rtol = 1e-2 if is_hip_cdna2() else 0
    if torch.allclose(iree_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ ExpReduction and Torch match")
    else:
        print("❌ ExpReduction and Torch differ")


if __name__ == "__main__":
    main()
