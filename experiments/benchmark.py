import torch
import time

from torch_baseline.fused_block import FusedBlockTorch
from triton_kernel.driver import fused_block_triton

def benchmark():
    torch.manual_seed(0)
    M, N = 128, 4096
    device = "cuda"

    x = torch.randn((M, N), device=device)
    w1 = torch.randn((N, N), device=device)
    w2 = torch.randn((N, N), device=device)

    # Pytorch baseline
    model = FusedBlockTorch(N).to(device)
    model.linear1.weight.data.copy_(w1)
    model.linear1.bias.data.zero_()
    model.linear2.weight.data.copy_(w2)
    model.linear2.bias.data.zero_()

    # warmup
    for _ in range(10):
        _ = model(x)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize()
    torch_time = time.time() - t0

    # Triton kernel
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = fused_block_triton(x, w1, w2)
    torch.cuda.synchronize()
    triton_time = time.time() - t0

    print(f"Pytorch baseline: {torch_time:.4f}s")
    print(f"Triton kernel: {triton_time:.4f}s")
