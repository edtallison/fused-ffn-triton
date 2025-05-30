import torch
import time

from torch_baseline.fused_block import FusedBlockTorch
from triton_kernel.driver import fused_block_triton

def benchmark():
    torch.manual_seed(0)
    M, N = 128, 4096
    device = "cuda"

    x = torch.randn((M, N), device=device)
    w = torch.randn((N, N), device=device)

    print(f"x shape: {x.shape}")              # should be (128, 4096)
    print(f"w shape: {w.shape}")              # should be (4096, 4096)



    # Pytorch baseline
    model = FusedBlockTorch(N).to(device)

    print(f"model.linear.weight shape: {model.linear.weight.data.shape}")  # should be (4096, 4096)
    model.linear.weight.data.copy_(w)
    model.linear.bias.data.zero_()

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
        _ = fused_block_triton(x, w)
    torch.cuda.synchronize()
    triton_time = time.time() - t0

    print(f"Pytorch baseline: {torch_time:.4f}s")
    print(f"Triton kernel: {triton_time:.4f}s")
