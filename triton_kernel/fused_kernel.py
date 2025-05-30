import triton
import triton.language as tl
import torch

@triton.jit
def fused_kernel(
    x_ptr, w_ptr, y_ptr, # pointers to input, weight, output in GPU memory
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_ym, stride_yn,
    M: tl.constexpr, N: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0) 
    col = tl.program_id(1)

    # flat offsets
    offs_x = row * stride_xm + col * stride_xn
    offs_w = row * stride_wm + tl.arange(0, N)

    # load input element
    x_val = tl.load(x_ptr + offs_x)

    # === LayerNorm (per row) ===
    # For simplicity, assume M=1 or pre-normalized input; real impl should
    # compute mean & var over the full row via two-pass reduction.

    # === Linear (dot product) ===
    # Load weight row and perform dot
    # TODO: replace with block-wise matmul for full rows
    w_row = tl.load(w_ptr + offs_w)
    dot = tl.dot(x_val, w_row)

    # === GELU ===
    gelu = 0.5 * dot * (1.0 + tl.erf(dot / tl.sqrt(2.0)))

    # store result
    out_offs = row * stride_ym + col * stride_yn
    tl.store(y_ptr + out_offs, gelu)
