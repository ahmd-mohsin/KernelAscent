import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 382
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _softmax_ln_kernel(
    X, OUT, G, B,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch fp16 softmax internal compute)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    p = e / s

    # round to fp16 to match reference intermediate dtype, then back to fp32
    p = p.to(tl.float16).to(tl.float32)
    p = tl.where(mask, p, 0.0)

    # layer norm in fp32
    mean = tl.sum(p, 0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(OUT + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            h = torch.softmax(h, dim=-1)
            return F.layer_norm(h, (h.shape[-1],), self.ln2_g, self.ln2_b)

        h = h.contiguous()
        rows, N = h.shape[0], h.shape[-1]
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_ln_kernel[(rows,)](
            h, out, self.ln2_g, self.ln2_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
