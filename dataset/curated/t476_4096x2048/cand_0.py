import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 476
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_scale_softmax2_ln_kernel(
    X_ptr, G_ptr, B_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + cols).to(tl.float32)

    # scale (bf16 rounding like reference: bf16 * float scalar -> bf16)
    x = (x * 1.0685).to(tl.bfloat16).to(tl.float32)

    # softmax #1 (fp32 accumulate, bf16 output like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # softmax #2
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 internals, eps=1e-5, biased variance)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G_ptr + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores)
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        _fused_scale_softmax2_ln_kernel[(Mrows,)](
            y, self.ln4_g, self.ln4_b, out,
            y.stride(0), out.stride(0),
            N=N, BLOCK=512,
            num_warps=4,
        )
        return out
