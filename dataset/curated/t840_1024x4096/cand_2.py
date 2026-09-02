import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 840
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _softmax_gelu_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's bf16 softmax)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), 0)
    p = e / s
    # round to bf16 (softmax output dtype), then upcast for gelu opmath
    p = p.to(tl.bfloat16).to(tl.float32)

    # exact gelu (erf) in fp32, then round to bf16
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # rmsnorm in fp32
    gm = tl.where(mask, g, 0.0)
    ms = tl.sum(gm * gm, 0) / N
    r = g * tl.math.rsqrt(ms + 1e-6)

    # cast to bf16 then multiply by bf16 weight (bf16 arithmetic, like PyTorch)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = r.to(tl.bfloat16) * w

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS bf16 tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape

        # fused softmax + gelu + rmsnorm + weight-scale
        y = torch.empty_like(h)
        _softmax_gelu_rms_kernel[(Mrows,)](
            h, self.rms3_w, y,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )

        # GEMM 2 + fused in-place relu
        out = y @ self.W4
        return torch.relu_(out)
