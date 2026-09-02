import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 720
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x, stride_o,
    eps, scale,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)

    # Load row (bf16 -> fp32)
    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    # RMSNorm in fp32, round to bf16 (matches .to(x.dtype))
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    xn_bf16 = (x * inv).to(tl.bfloat16)

    # Multiply by weight: bf16 op done at fp32 opmath, rounded back to bf16
    w = tl.load(W_ptr + offs).to(tl.float32)
    y_bf16 = (xn_bf16.to(tl.float32) * w).to(tl.bfloat16)

    # Softmax: fp32 accumulation, bf16 output (matches PyTorch bf16 softmax)
    yf = y_bf16.to(tl.float32)
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    s = tl.sum(e, axis=0)
    p_bf16 = (e / s).to(tl.bfloat16)

    # relu(relu(p)) is a no-op on softmax output (>= 0); final scale in fp32 opmath
    out = tl.maximum(p_bf16.to(tl.float32), 0.0) * scale
    tl.store(Out_ptr + row * stride_o + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        x = x @ self.W0

        x = x.contiguous()
        out = torch.empty_like(x)
        M_rows, N = x.shape
        grid = (M_rows,)
        _fused_rms_softmax_kernel[grid](
            x, self.rms1_w, out,
            x.stride(0), out.stride(0),
            1e-6, 1.1833,
            N=N,
            num_warps=4,
        )
        return out
