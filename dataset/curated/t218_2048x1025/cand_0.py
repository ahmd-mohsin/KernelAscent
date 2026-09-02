import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 218
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    eps,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, N)

    x = tl.load(X_ptr + row * stride_x + cols).to(tl.float32)

    # RMS norm in fp32 (matches reference _xf math)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # cast normalized value back to bf16 (matches .to(x.dtype))
    xn = (x * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols).to(tl.float32)

    # bf16 elementwise ops in PyTorch compute in fp32 then round to bf16
    y = (xn.to(tl.float32) * w).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.1772).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.3155).to(tl.bfloat16)

    # exact GELU with fp32 internal math (matches PyTorch opmath for bf16)
    yf = y.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + cols, g.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (same as reference)
        x = x @ self.W0

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.reshape(-1, N)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        rows = x2.shape[0]

        out = torch.empty_like(x2)
        _rms_gelu_kernel[(rows,)](
            x2, self.rms1_w, out,
            x2.stride(0), out.stride(0),
            1e-6,
            N=N,
            num_warps=4,
        )
        return out.reshape(orig_shape)
