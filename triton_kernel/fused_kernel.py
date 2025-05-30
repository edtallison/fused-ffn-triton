import triton
import triton.language as tl
import torch

@triton.jit
def fused_kernel(
    x_ptr, w1_ptr, w2_ptr, y_ptr, # pointers to input, weight, output in GPU memory
    stride_xm, stride_xn,
    stride_w1m, stride_w1n,
    stride_w2m, stride_w2n,
    stride_ym, stride_yn,
    M: tl.constexpr, N: tl.constexpr,
    eps: tl.constexpr,
):
    # which row we are operating on
    row = tl.program_id(0)

    # load the row
    offs_x = row * stride_xm
    idxs = tl.arange(0, N)
    x_row = tl.load(x_ptr + offs_x + idxs * stride_xn, mask=idxs < N, other=0.0)

    # layernorm
    mean = tl.sum(x_row, axis=0) / N
    var = tl.sum((x_row - mean) * (x_row - mean), axis=0) / N
    x_norm = (x_row - mean) / tl.sqrt(var + eps)

    # linear1
    offs_w1 = row * stride_w1m
    w1_row = tl.load(w1_ptr + offs_w1 + idxs * stride_w1n, mask=idxs < N, other=0.0)
    hidden = tl.sum(x_norm * w1_row, axis=0) # dot product

    # gelu
    gelu = 0.5 * hidden * (1.0 + tl.erf(hidden / tl.sqrt(2.0)))

    # linear2
    offs_w2 = row * stride_w2m
    w2_row = tl.load(w2_ptr + offs_w2 + idxs * stride_w2n, mask=idxs < N, other=0.0)
    out = tl.sum(gelu * w2_row, axis=0)

    # store result
    offs_y = row * stride_ym
    tl.store(y_ptr + offs_y + idxs * stride_yn, out, mask=idxs < N)
