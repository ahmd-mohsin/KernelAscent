import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 743
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_relu_softmax_scale_ln(
    X_ptr, G_ptr, B_ptr, Out_ptr,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax in fp32 (mirrors PyTorch's fp32 accumulation for bf16)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s

    # round to bf16 like the reference (softmax output dtype)
    p = p.to(tl.bfloat16).to(tl.float32)

    # scale (fp32 compute, bf16 round, matching PyTorch scalar mul on bf16)
    p = p * scale
    p = p.to(tl.bfloat16).to(tl.float32)

    # layer norm in fp32
    mean = tl.sum(tl.where(mask, p, 0.0), 0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Out_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference path
            y = x @ self.W0
            y = torch.relu(y)
            y = torch.softmax(y, dim=-1)
            y = y * 1.1424
            return F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)

        h = torch.matmul(x, self.W0)  # (M, 2048) bf16, tensor-core GEMM
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_relu_softmax_scale_ln[(rows,)](
            h, self.ln4_g, self.ln4_b, out,
            N, 1e-5, 1.1424,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
