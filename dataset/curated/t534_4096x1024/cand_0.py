import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 534
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_bias_softmax_scale_ln(
    Y_ptr, Bias_ptr, G_ptr, B_ptr, O_ptr,
    stride_y, stride_o,
    N, scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_y + offs, mask=mask, other=-float('inf')).to(tl.float32)
    bias = tl.load(Bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = y + bias

    # softmax (fp32 accumulation, matches torch's half softmax internals)
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s) * scale

    # layernorm
    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = d * rstd * g + b

    tl.store(O_ptr + row * stride_o + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS tensor-core GEMM

        if not y.is_cuda:
            y = y + self.b1
            y = torch.softmax(y, dim=-1) * 1.0871
            return F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)

        y2 = y.view(-1, y.shape[-1])
        rows, N = y2.shape
        out = torch.empty_like(y2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_softmax_scale_ln[(rows,)](
            y2, self.b1, self.ln4_g, self.ln4_b, out,
            y2.stride(0), out.stride(0),
            N, 1.0871, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view_as(y)
