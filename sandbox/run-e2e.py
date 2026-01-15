import iree.runtime as ireert
import iree.compiler as comp
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


def is_hip_cdna2():
    return False
    # target = triton.runtime.driver.active.get_current_target()
    # return target.backend == 'hip' and target.arch == 'gfx90a'


def torch_test(q, k, v):
    ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    p = torch.matmul(q, k.transpose(2, 3))
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    out = torch.matmul(p, v)
    return out


def main():
    # shape = (4, 32, 4096, 64)
    shape = (4, 32, 8192, 64)
    kv_dict = {
        "BATCH": shape[0],
        "HEADS": shape[1],
        "NCTX": shape[2],
        "HEADDIM": shape[3],
    }

    instanced_file = "e2e-instanced.mlir"
    template = open("e2e-template.mlir", "r").read()
    for key, val in kv_dict.items():
        template = template.replace(key, str(val))
    with open("e2e-instanced.mlir", "w") as f:
        f.write(template)

    flatbuffer = comp.compile_file(
        instanced_file,
        target_backends=["llvm-cpu"],
        extra_args=["--iree-hal-target-device=local", "--iree-llvmcpu-target-cpu=host"],
    )
    module = ireert.load_vm_flatbuffer(flatbuffer, backend="llvm-cpu")
    device = "cuda"

    q = torch.randn(shape, dtype=torch.float32)
    k = torch.randn(shape, dtype=torch.float32)
    v = torch.randn(shape, dtype=torch.float32)

    # print(dir(modules))
    iree_output = module.attention(q, k, v)
    iree_output = torch.from_numpy(iree_output.to_host())
    # torch_output = torch_test(q, k, v).cpu()
    torch_output = scaled_dot_product_attention(q, k, v, scale=1)
    # triton_output = attention(q.to(device=device),k.to(device=device),v.to(device=device), False, 1)

    print(iree_output[0][0][0])
    print(torch_output[0][0][0])

    rtol = 1e-2 if is_hip_cdna2() else 0
    if torch.allclose(iree_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ ExpReduction and Torch match")
    else:
        print("❌ ExpReduction and Torch differ")


if __name__ == "__main__":
    main()
