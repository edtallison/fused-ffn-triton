import torch

from triton_kernel.fused_kernel import fused_kernel

def fused_block_triton(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Runs the fused LayerNorm -> Linear -> GELU kernel using Triton.

    Args:
        x: input tensor of shape (M, N)
        w1: weight tensor of shape (N, N)
        w2: weight tensor of shape (N, N)
        eps: epsilon for LayerNorm stability
    Returns:
        y: output tensor of shape (M, N)
    """
    assert x.is_cuda and w1.is_cuda and w2.is_cuda, "x, w1, and w2 must be on CUDA"
    M, N = x.shape
    y = torch.empty_like(x)

    grid = (M,) # one program per row

    fused_kernel[grid](
        x_ptr=x, w1_ptr=w1, w2_ptr=w2, y_ptr=y,
        stride_xm=x.stride(0), stride_xn=x.stride(1),
        stride_w1m=w1.stride(0), stride_w1n=w1.stride(1),
        stride_w2m=w2.stride(0), stride_w2n=w2.stride(1),
        stride_ym=y.stride(0), stride_yn=y.stride(1),
        M=M, N=N, eps=eps
    )

    return y
