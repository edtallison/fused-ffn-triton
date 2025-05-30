import torch

from triton_kernel.fused_kernel import fused_kernel

def fused_block_triton(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Runs the fused LayerNorm -> Linear -> GELU kernel using Triton.

    Args:
        x: input tensor of shape (M, N)
        w: weight tensor of shape (N, N)
        eps: epsilon for LayerNorm stability
    Returns:
        y: output tensor of shape (M, N)
    """
    assert x.is_cuda and w.is_cuda, "x and w must be on CUDA"
    M, N = x.shape
    y = torch.empty_like(x)

    grid = (M, N)

    fused_kernel[grid](
        x_ptr=x, w_ptr=w, y_ptr=y,
        stride_xm=x.stride(0), stride_xn=x.stride(1),
        stride_wm=w.stride(0), stride_wn=w.stride(1),
        stride_ym=y.stride(0), stride_yn=y.stride(1),
        M=M, N=N, eps=eps
    )

    return y
