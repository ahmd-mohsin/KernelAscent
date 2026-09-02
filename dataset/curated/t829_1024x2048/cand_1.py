import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 829
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _rms_softmax_kernel(
    X, W, Out,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.float16)  # cast to half like reference

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    v = (xn.to(tl.float32) * w).to(tl.float16)  # half*half via fp32 compute, cast back

    # softmax in fp32 (as PyTorch does for half inputs)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM (TF/HMMA on A100)
        Mrows, N = y.shape
        out = torch.empty((Mrows, N), device=y.device, dtype=torch.float16)
        BLOCK = triton.next_power_of_2(N)
        _rms_softmax_kernel[(Mrows,)](
            y, self.rms1_w, out,
            y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
