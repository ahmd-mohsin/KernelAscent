import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 705
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_bias_relu_softmax_ln(
    X, B1, B2, G, B, Out,
    N, stride, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    # match PyTorch bf16 elementwise semantics: fp32 compute, round to bf16 each op
    x = (x + b1).to(tl.bfloat16).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # softmax in fp32 (as PyTorch does for bf16), output rounded to bf16
    xs = tl.where(mask, x, float('-inf'))
    m = tl.max(xs, 0)
    e = tl.exp(xs - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # layer norm in fp32 (as PyTorch does for bf16)
    mean = tl.sum(tl.where(mask, sm, 0.0), 0) / N
    d = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Out + row * stride + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference path on CPU
            x = x @ self.W0
            x = x + self.b1
            x = x + self.b2
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
            return x

        # GEMM via cuBLAS tensor cores (same as reference matmul)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_relu_softmax_ln[(rows,)](
            h, self.b1, self.b2, self.ln5_g, self.ln5_b, out,
            N, h.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
