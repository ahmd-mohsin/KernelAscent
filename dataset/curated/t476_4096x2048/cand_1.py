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
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.0685 (bf16 elementwise op: fp32 opmath, round to bf16)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    # softmax #1 (fp32 compute, bf16 output rounding)
    xm = tl.where(mask, x, float('-inf'))
    m = tl.max(xm, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # softmax #2
    xm = tl.where(mask, x, float('-inf'))
    m = tl.max(xm, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 compute)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = x * 1.0685
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)

        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        N = h.shape[-1]
        rows = h.numel() // N
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_scale_softmax2_ln_kernel[(rows,)](
            h, self.ln4_g, self.ln4_b, out,
            N, 1.0685, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
