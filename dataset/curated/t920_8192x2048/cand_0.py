import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 920
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_epilogue(X_ptr, B_ptr, W_ptr, Y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    ptr = X_ptr + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # exact GELU in fp32, round to fp16 (matches PyTorch half GELU)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # bias add (fp32 compute, fp16 rounding)
    b = tl.load(B_ptr + offs).to(tl.float32)
    s = (g + b).to(tl.float16).to(tl.float32)

    # scalar scale (fp32 compute, fp16 rounding)
    m = (s * 1.3289).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(m * m, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = (m * r).to(tl.float16).to(tl.float32)

    # weight multiply (fp32 compute, fp16 rounding)
    w = tl.load(W_ptr + offs).to(tl.float32)
    out = (y * w).to(tl.float16)
    tl.store(Y_ptr + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_epilogue[(Mrows,)](
            h, self.b2, self.rms4_w, y, N=N,
            num_warps=8,
        )
        return y


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
